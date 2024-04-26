from copy import copy

from transformers import T5Tokenizer, T5Model, T5Config
from transformers import T5ForConditionalGeneration, T5TokenizerFast
from allennlp.predictors import Predictor, TextClassifierPredictor
from allennlp_models.classification import StanfordSentimentTreeBankDatasetReader
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer

import torch
import os
import csv
import heapq
import sys
import operator
from tqdm import tqdm
import re
import nltk
import warnings
import argparse
import pandas as pd
import numpy as np
import random
import time
import logging
import json

# Local imports
from src.utils import *
from src.edit_finder import EditFinder, EditEvaluator, EditList
from src.editor import Editor, RaceEditor
from src.predictors.imdb.imdb_dataset_reader import ImdbDatasetReader
from src.predictors.newsgroups.newsgroups_dataset_reader import NewsgroupsDatasetReader
from src.predictors.race.race_dataset_reader import RaceDatasetReader

logger = logging.getLogger("my-logger")
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"), format=FORMAT)
logger.setLevel(logging.INFO)

def get_grad_sign_direction(grad_type, grad_pred):
    """ Helper function to get sign direction. When grad_type is signed, 
    determines whether to get most negative or positive gradient values. 
    This should depend on grad_pred, i.e. what label is being used to 
    compute gradients for masking. 
    
    During Stage Two, we want to mask tokens that push *away* from the contrast 
    label or *towards* the original label. 
    
    Sign direction plays no role if only gradient *magnitudes* are used 
        (i.e. if grad_type is not signed, but involves taking the l1/l2 norm.)
    """
    assert grad_pred in ["contrast", "original"]
    assert grad_type in ["integrated_l1", "integrated_signed", "normal_l1", 
            "normal_signed", "normal_l2", "integrated_l2"]
    if "signed" in grad_type and "contrast" in grad_pred:
        sign_direction = -1
    elif "signed" in grad_type and "original" in grad_pred:
        sign_direction = 1
    else:
        sign_direction = None
    return sign_direction

def load_editor_weights(editor_model, editor_path):
    """ Loads Editor weights from editor_path """

    if os.path.isdir(editor_path):
        editor_path = os.path.join(editor_path, "best.pth")
        if not os.path.exists(editor_path):
            raise NotImplementedError(f"If directory given for editor_path, \
                    it must contain a 'best.pth' file but found none in given \
                    dir. Please give direct path to file containing weights.")
    logger.info(f"Loading Editor weights from: {editor_path}")
    editor_model.load_state_dict(torch.load(editor_path),strict=False)
    return editor_model

def load_models(args):
    """ Loads Predictor and Editor by task and other args """

    logger.info("Loading models...")
    # predictor = load_predictor(args.meta.task)
    predictor = load_predictor_test()
    editor_tokenizer_wrapper = PretrainedTransformerTokenizer(
            't5-base', max_length=args.model.model_max_length)
    editor_tokenizer, editor_model = load_base_t5(
                       max_length=args.model.model_max_length)
    device = get_device()
    editor_model = load_editor_weights(editor_model, args.meta.editor_path)
    editor_model = editor_model.to(device)
    sign_direction = get_grad_sign_direction(
            args.mask.grad_type, args.misc.grad_pred) 
    
    masker = GradientMasker(args.search.max_mask_frac, 
            editor_tokenizer_wrapper, predictor, 
            args.model.model_max_length,
            grad_type=args.mask.grad_type, 
            sign_direction=sign_direction)

    if "race" in args.meta.task:
        editor = RaceEditor(editor_tokenizer_wrapper, editor_tokenizer, 
                editor_model, masker, 
                num_gens=args.generation.num_generations, 
                num_beams=args.generation.generation_num_beams, 
                grad_pred=args.misc.grad_pred, 
                generate_type=args.generation.generate_type, 
                length_penalty=args.generation.length_penalty, 
                no_repeat_ngram_size=args.generation.no_repeat_ngram_size, 
                top_p=args.generation.top_p,
                top_k=args.generation.top_k, 
                verbose=False, 
                editable_key="article")
    else:
        editor = Editor(editor_tokenizer_wrapper, editor_tokenizer, 
                editor_model, masker, 
                num_gens=args.generation.num_generations, 
                num_beams=args.generation.generation_num_beams, 
                grad_pred=args.misc.grad_pred, 
                generate_type=args.generation.generate_type, 
                no_repeat_ngram_size=args.generation.no_repeat_ngram_size, 
                top_p=args.generation.top_p, 
                top_k=args.generation.top_k, 
                length_penalty=args.generation.length_penalty, 
                verbose=False)
    logger.info("Done loading models.")
    return editor, predictor

def run_edit_test(args):

    """ Runs Stage 2 on test inputs by task. """

    task_dir = os.path.join(args.meta.results_dir, args.meta.task)
    stage_two_dir = os.path.join(task_dir, f"edits/{args.meta.stage2_exp}")
   
    if not os.path.exists(stage_two_dir):
        os.makedirs(stage_two_dir)

    logger.info(f"Task dir: {task_dir}")
    logger.info(f"Stage two dir: {stage_two_dir}")

    # Save args
    args_path = os.path.join(stage_two_dir, "stage_two_args.json")
    write_args(args_path, args)
    
    out_file = os.path.join(stage_two_dir, "edits.csv")
    meta_log_file = os.path.join(stage_two_dir, "meta_log.txt")

    meta_f = open(meta_log_file, 'w', 1)

    # Load models and Edit objects 
    editor, predictor = load_models(args)
    dr = get_dataset_reader(args.meta.task, predictor)
    edit_evaluator = EditEvaluator()
    edit_finder = EditFinder(predictor, editor, 
            beam_width=args.search.beam_width, 
            max_mask_frac=args.search.max_mask_frac,
            search_method=args.search.search_method,
            max_search_levels=args.search.max_search_levels)

    # Get inputs
    inputs = dr.get_inputs('test')
    print(type(inputs))
    print(inputs[0])
    if "race" not in args.meta.task:
        inputs = [x for x in inputs if len(x) > 0 and re.search('[a-zA-Z]', x)]

    np.random.seed(0)
    input_indices = np.array(range(len(inputs)))
    np.random.shuffle(input_indices)
    # Find edits and write to file
    with open(out_file, "w") as csv_file:
        fieldnames = ["data_idx", "sorted_idx", "orig_pred", "new_pred", 
                "contrast_pred", "orig_contrast_prob_pred", 
                "new_contrast_prob_pred", "orig_input", "edited_input", 
                "orig_editable_seg","masked_sentence","predicted_mask", "edited_editable_seg","orig_span",
                " ", "num_edit_rounds", "mask_frac",
                "duration", "error"]
        writer = csv.writer(csv_file, delimiter="\t")
        writer.writerow(fieldnames)
        count = 0
        loss_b =0
        loss_a =0

        device = get_device()
        for idx, i in tqdm(enumerate(input_indices), total=len(input_indices)):
            torch.cuda.empty_cache()
            inp = inputs[i]
            # inp = "With a catchy title like the Butcher of Plainfield this Ed Gein variation and Kane Hodder playing him will no doubt fly off the shelves for a couple of weeks.Most viewers will be bored silly with this latest take on the life of Ed Gien. The movie focuses on Ed’s rampage and gives us a(few)glimpses into his Psycosis and dwelling in Plainfeild.Its these scenes that give the movie a much needed jolt. What ruins this is the constant focus on other characters lives and focuses less on Eds.Big mistake here. Kane Hodder is a strange choice to play Gein,but He does pull it off quite well,and deserves more acting credits than he gets these days.Prascilla Barnes and Micahel Barryman also show up. 3/10"
            if idx >=1000:
                break
            # if "Fido is a story about more well mannered zombie" not in inp:
            #     continue
            logger.info(wrap_text(f"ORIGINAL INSTANCE ({i}): {inp}"))
            count +=1
            # reo =  re.findall(r'\d+/10',inp)
            # # print(reo)
            # temp_inp = copy(inp)
            # temp_label = "<pad><extra_id_0>"
            # temp_label_2 = "<pad><extra_id_0>"
            # for i,r in enumerate(reo):
            #     # print(r)
            #
            #     rate = int(r[0:-3])
            #     temp_label_2 = temp_label_2 + f" {str(rate)}<extra_id_{str(i + 1)}>"
            #     rate = 10-rate
            #     str_replace = f"<extra_id_{str(i)}>/10"
            #
            #     temp_inp = temp_inp.replace(r, str_replace)
            #
            #     str_re = str(rate)
            #     temp_label = temp_label + f" {str_re}<extra_id_{str(i+1)}>"
            #     print(temp_inp)
            #     print(temp_label_2)
            #     print(temp_label)
            # input_ids = editor.tokenizer(
            #     temp_inp, return_tensors="pt"
            # ).input_ids.to(device)
            # labels = editor.tokenizer(temp_label, return_tensors="pt").input_ids.to(device)
            # loss_a += editor.editor_model(input_ids, labels=labels).loss.item()
            #
            # labels = editor.tokenizer(temp_label_2, return_tensors="pt").input_ids.to(device)
            # loss_b += editor.editor_model(input_ids, labels=labels).loss.item()
            # outputs = editor.editor_model.generate(input_ids, eos_token_id = editor.tokenizer(f"<extra_id_{str(i+1)}>")['input_ids'][0])
            # print(editor.tokenizer.decode([3,168,117,28,166]))
            # input_ids.detach()
            # labels.detach()
            # torch.cuda.empty_cache()
            # print(count)
            # print(loss_a)
            # print(loss_b)
        # print("Loss Before:" + str(loss_b/count))
        # print("Loss After:" + str(loss_a / count))

            start_time = time.time()
            error = False
            # try:
            edited_list = edit_finder.minimally_edit(inp,
                    max_edit_rounds=args.search.max_edit_rounds,
                    edit_evaluator=edit_evaluator)

            torch.cuda.empty_cache()
            sorted_list = edited_list.get_sorted_edits()
            # print("Bach:")
            # print(edited_list.beam)
            # print("Bach1:")
            # print(edited_list.successful_edits)
            # except Exception as e:
            #     logger.info("ERROR: ", e)
            #     error = True
            #     sorted_list = []

            end_time = time.time()

            duration = end_time - start_time
            for s_idx, s in enumerate(sorted_list):
                writer.writerow([i, s_idx, edited_list.orig_label,
                    s['edited_label'], edited_list.contrast_label,
                    edited_list.orig_contrast_prob, s['edited_contrast_prob'],
                    edited_list.orig_input, s['edited_input'],
                    edited_list.orig_editable_seg,s['masked_sentence'],s['predicted_mask'],
                    s['edited_editable_seg'], s['orig_span'], s['minimality'],
                    s['num_edit_rounds'], s['mask_frac'], duration, error])
                csv_file.flush()


            if sorted_list == []:
                writer.writerow([i, 0, edited_list.orig_label,
                    None, edited_list.contrast_label,
                    edited_list.orig_contrast_prob, None,
                    edited_list.orig_input, None,
                    edited_list.orig_editable_seg,
                    None, None, None, None, duration, error])
                csv_file.flush()
                meta_f.flush()




    # loss_orig = []
    # loss_gen = []
    # input_ids = editor.tokenizer(
    #     s['masked_sentence'], return_tensors="pt"
    # ).input_ids.to(device)
    # labels = editor.tokenizer(s['orig_span'], return_tensors="pt").input_ids.to(device)
    # loss_orig.append(editor.editor_model(input_ids, labels=labels).loss.item())
    #
    # labels = editor.tokenizer(s['predicted_mask'], return_tensors="pt").input_ids.to(device)
    # loss_gen.append(editor.editor_model(input_ids, labels=labels).loss.item())
    csv_file.close()
    meta_f.close()
    df = pd.read_csv(out_file,sep='	').groupby(["orig_input"]).first()
    loss_orig = []
    loss_gen = []
    for i in range(len(df)):
        s = df.iloc[i]
        # if s['minimality'] >0.05:
        #     continue
        input_ids = editor.tokenizer(
            s['masked_sentence'], return_tensors="pt"
        ).input_ids.to(device)
        print("masked_sentence:" + str(s['masked_sentence']))
        print("predicted_mask:" + str(s['predicted_mask']))
        print("orig_span:" + str(s['orig_span']))
        labels = editor.tokenizer(s['predicted_mask'], return_tensors="pt").input_ids.to(device)
        loss_gen.append(editor.editor_model(input_ids, labels=labels).loss.item())
        labels = editor.tokenizer(s['orig_span'], return_tensors="pt").input_ids.to(device)
        loss_orig.append(editor.editor_model(input_ids, labels=labels).loss.item())

    print("loss orig: " +str(np.mean(loss_orig)))
    print(loss_orig)
    print("loss gen: " + str(np.mean(loss_gen)))
    print(loss_gen)