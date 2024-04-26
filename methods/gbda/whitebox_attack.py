# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
import pandas as pd
from bert_score.utils import get_idf_dict
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, \
    RobertaForSequenceClassification
import transformers
import math
import argparse
import math
import jiwer
import numpy as np
from allennlp_utils import load_predictor
import os
import warnings
transformers.logging.set_verbosity(transformers.logging.ERROR)
import time 
import torch
import torch.nn.functional as F
import os
import traceback
import logging

from src.dataset import load_data
from src.utils import bool_flag, get_output_file, print_args, load_gpt2_from_dict
import os


def wer(x, y):
    x = " ".join(["%d" % i for i in x])
    y = " ".join(["%d" % i for i in y])

    return jiwer.wer(x, y)


def bert_score(refs, cands, weights=None):
    refs_norm = refs / refs.norm(2, -1).unsqueeze(-1)
    if weights is not None:
        refs_norm *= weights[:, None]
    else:
        refs_norm /= refs.size(1)
    cands_norm = cands / cands.norm(2, -1).unsqueeze(-1)
    cosines = refs_norm @ cands_norm.transpose(1, 2)
    # remove first and last tokens; only works when refs and cands all have equal length (!!!)
    cosines = cosines[:, 1:-1, 1:-1]
    R = cosines.max(-1)[0].sum(1)
    return R


def log_perplexity(logits, coeffs):
    shift_logits = logits[:, :-1, :].contiguous()
    shift_coeffs = coeffs[:, 1:, :].contiguous()
    shift_logits = shift_logits[:, :, :shift_coeffs.size(2)]
    return -(shift_coeffs * F.log_softmax(shift_logits, dim=-1)).sum(-1).mean()

def load_predictor_test():
    model_name = 'textattack/bert-base-uncased-imdb'
    from allennlp.data.tokenizers.pretrained_transformer_tokenizer import PretrainedTransformerTokenizer
    from allennlp.data.token_indexers.pretrained_transformer_indexer import PretrainedTransformerIndexer

    transformer_tokenizer = PretrainedTransformerTokenizer(model_name)
    token_indexer = PretrainedTransformerIndexer(model_name)

    from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
    from allennlp.modules.token_embedders.pretrained_transformer_embedder import PretrainedTransformerEmbedder

    token_embedder = BasicTextFieldEmbedder(
        {
            "tokens": PretrainedTransformerEmbedder(model_name)
        })
    from allennlp.modules.seq2vec_encoders.bert_pooler import BertPooler

    transformer_encoder = BertPooler(model_name)
    from allennlp.common import cached_transformers
    from allennlp.data.vocabulary import Vocabulary
    namespace = "tokens"
    tokenizer = cached_transformers.get_tokenizer(model_name)

    if hasattr(tokenizer, "_unk_token"):
        oov_token = tokenizer._unk_token
    elif hasattr(tokenizer, "special_tokens_map"):
        oov_token = tokenizer.special_tokens_map.get("unk_token")
    vocab = Vocabulary(non_padded_namespaces=[namespace], oov_token=oov_token)
    vocab_items = tokenizer.get_vocab().items()
    for word, idx in vocab_items:
        vocab._token_to_index[namespace][word] = idx
        vocab._index_to_token[namespace][idx] = word
    vocab._non_padded_namespaces.add(namespace)

    from allennlp.modules.seq2vec_encoders.bert_pooler import BertPooler

    transformer_encoder = BertPooler(model_name).cuda()

    from allennlp.models import BasicClassifier

    model = BasicClassifier(vocab=vocab,
                            text_field_embedder=token_embedder,
                            seq2vec_encoder=transformer_encoder,
                            dropout=0.1,
                            num_labels=2).cuda()
    from transformers.models.auto import AutoConfig, AutoModelForSequenceClassification
    from transformers.models.auto.tokenization_auto import AutoTokenizer

    # model_uri = 'nlptown/bert-base-multilingual-uncased-sentiment'

    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    classifier = AutoModelForSequenceClassification.from_pretrained(model_name, config=config).cuda()

    model._classification_layer.weight = classifier.classifier.weight
    model._classification_layer.bias = classifier.classifier.bias
    from allennlp.data.dataset_readers import TextClassificationJsonReader

    dataset_reader = TextClassificationJsonReader(token_indexers={"tokens": token_indexer},
                                                  tokenizer=transformer_tokenizer,
                                                  max_sequence_length=512)
    from allennlp.predictors import TextClassifierPredictor
    predictor = TextClassifierPredictor(model, dataset_reader)
    return predictor
def main(args):
    # print(os.environ['HUGGINGFACE_HUB_CACHE'])
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    pretrained = args.model.startswith('textattack') or args.model.startswith('roberta')
    output_file = get_output_file(args, args.model, args.start_index, args.start_index + args.num_samples)
    output_file = os.path.join(args.adv_samples_folder, output_file)
    print(f"Outputting files to {output_file}")
    if os.path.exists(output_file):
        print('Skipping batch as it has already been completed.')
        exit()
    
    # Load dataset
    import datasets
    dataset, num_labels = load_data(args)
    df = pd.read_csv("datasets/merge.csv", delimiter="\t")
    df = df.iloc[100:]
    df = df[["Orig_Inp", "Orig_Sent"]]
    df.rename(columns={"Orig_Inp": "text",
                       "Orig_Sent": "label"}, inplace=True)
    # df = pd.read_csv("datasets/test.tsv", delimiter="\t")
    # df = df[["Orig_Inp", "Orig_Sent"]]
    # df.rename(columns={"sentence1": "premise",
    #                    "sentence2": "hypothesis",
    #                    "gold_label": "label"},inplace=True)
    # df['label'] = df['label'].map({"entailment":1, "contradiction":0, "neutral": 2})
    df['label'] = df['label'].map({"Positive":1, "Negative":0})
    ds = datasets.Dataset.from_pandas(df)
    test_key = "validation_%s" % args.mnli_option
    dataset[test_key] = ds
    label_perm = lambda x: x
    if pretrained and args.model == 'textattack/bert-base-uncased-MNLI':
        label_perm = lambda x: (x + 1) % 3

    # Load tokenizer, model, and reference model

    if args.model == 'roberta':
        # model = load_predictor_test()
        tokenizer = AutoTokenizer.from_pretrained("./pretrained_model")
        model = RobertaForSequenceClassification.from_pretrained("./pretrained_model").cuda()
        # tokenizer = model._dataset_reader._tokenizer.tokenizer

    else:
        model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=num_labels).cuda()
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
        tokenizer.model_max_length = 512
    if not pretrained:
        # Load model to attack
        suffix = '_finetune' if args.finetune else ''
        model_checkpoint = os.path.join(args.result_folder, '%s_%s%s.pth' % (args.model.replace('/', '-'), args.dataset, suffix))
        print(model_checkpoint)
        # model_checkpoint = "../mice_orig/trained_predictors/imdb/model/state.pth"
        print('Loading checkpoint: %s' % model_checkpoint)
        model.load_state_dict(torch.load(model_checkpoint))
        # model = torch.load(model_checkpoint)
        tokenizer.model_max_length = 512
    if args.model == 'gpt2':
        tokenizer.padding_side = "right"
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    if args.model == 'roberta':
        # ref_model = AutoModelForCausalLM.from_pretrained("roberta-base", output_hidden_states=True).cuda()
        # ref_model = load_gpt2_from_dict("%s/transformer_wikitext-103.pth" % args.gpt2_checkpoint_folder,
        #                                 output_hidden_states=True).cuda()
        ref_model = AutoModelForCausalLM.from_pretrained(
            "/home/nguyenv9/transformers/examples/pytorch/language-modeling/tmp/gpt2-roberta",
            output_hidden_states=True).cuda()
    elif 'bert-base-uncase' in args.model:
        # for BERT, load GPT-2 trained on BERT tokenizer
        ref_model = load_gpt2_from_dict("%s/transformer_wikitext-103.pth" % args.gpt2_checkpoint_folder, output_hidden_states=True).cuda()
        # ref_model = AutoModelForCausalLM.from_pretrained("bert-base-uncased", output_hidden_states=True).cuda()
    else:
        # ref_model = load_gpt2_from_dict("%s/transformer_wikitext-103.pth" % args.gpt2_checkpoint_folder,
        #                                                                 output_hidden_states=True).cuda()
        # ref_model = AutoModelForCausalLM.from_pretrained("/groups/dso/bach/transformers/examples/pytorch/language-modeling/tmp/gpt2-roberta", output_hidden_states=True).cuda()
        ref_model = AutoModelForCausalLM.from_pretrained(args.model, output_hidden_states=True).cuda()
    with torch.no_grad():
        # if args.model == 'roberta':
        #     embeddings = model._model._text_field_embedder.token_embedder_tokens.transformer_model.embeddings.word_embeddings.weight.cuda()
        # else:
        embeddings = model.get_input_embeddings()(torch.arange(0, tokenizer.vocab_size).long().cuda())
        ref_embeddings = ref_model.get_input_embeddings()(torch.arange(0, tokenizer.vocab_size).long().cuda())
        
    # encode dataset using tokenizer
    # if args.dataset == "mnli":
    if "nli" in args.dataset:
        testset_key = "validation_%s" % args.mnli_option
        preprocess_function = lambda examples: tokenizer(
            examples['premise'], examples['hypothesis'], max_length=256, truncation=True)
    else:
        text_key = 'text' if (args.dataset in ["ag_news", "imdb", "yelp"]) else 'sentence'
        testset_key = 'test' if (args.dataset in ["ag_news", "imdb", "yelp"]) else 'validation'
        testset_key = "validation_%s" % args.mnli_option
        preprocess_function = lambda examples: tokenizer(examples[text_key], max_length=512, truncation=True)
    encoded_dataset = dataset.map(preprocess_function, batched=True)
        
    # Compute idf dictionary for BERTScore
    if args.constraint == "bertscore_idf":
        # if "nli" in args.dataset:
        if "nli" in args.dataset:
            idf_dict = get_idf_dict(dataset['train']['premise'] + dataset['train']['hypothesis'], tokenizer, nthreads=20)
        else:
            idf_dict = get_idf_dict(dataset['train'][text_key], tokenizer, nthreads=20)
        
    # if "nli" in args.dataset:
    if "nli" in args.dataset:
        adv_log_coeffs = {'premise': [], 'hypothesis': []}
        clean_texts = {'premise': [], 'hypothesis': []}
        adv_texts = {'premise': [], 'hypothesis': []}
    else:
        adv_log_coeffs, clean_texts, adv_texts = [], [], []
    clean_logits = []
    adv_logits = []
    token_errors = []
    times = []
    pred_clean_labels = []
    adv_labels = []
    
    assert args.start_index < len(encoded_dataset[testset_key]), 'Starting index %d is larger than dataset length %d' % (args.start_index, len(encoded_dataset[testset_key]))
    end_index = min(args.start_index + args.num_samples, len(encoded_dataset[testset_key]))
    adv_losses, ref_losses, perp_losses, entropies = torch.zeros(end_index - args.start_index, args.num_iters), torch.zeros(end_index - args.start_index, args.num_iters), torch.zeros(end_index - args.start_index, args.num_iters), torch.zeros(end_index - args.start_index, args.num_iters)
    current_results = pd.read_csv("results.csv")
    try:
        start1 = time.time()
        for idx in range(args.start_index, end_index):
            print(f"index {idx}" )
            input_ids = encoded_dataset[testset_key]['input_ids'][idx]
            if args.model in ['gpt2',"roberta", "roberta-large"]:
                token_type_ids = None
            else:
                token_type_ids = encoded_dataset[testset_key]['token_type_ids'][idx]

            label = label_perm(encoded_dataset[testset_key]['label'][idx])
            # if args.model in ["roberta"]:
            #     x = encoded_dataset[testset_key]['text'][idx]
            #     clean_logit = torch.tensor(model.predict_instance(model._dataset_reader.text_to_instance(x))['logits'])
            #     predicted_clean_label = model.predict_instance(model._dataset_reader.text_to_instance(x))['label']
            #     pred_clean_labels.append(predicted_clean_label)
            # else:
            clean_logit = model(input_ids=torch.LongTensor(input_ids).unsqueeze(0).cuda(),
                                 token_type_ids=(None if token_type_ids is None else torch.LongTensor(token_type_ids).unsqueeze(0).cuda()), return_dict=True).logits.data.cpu()
            pred_clean_labels.append(int(clean_logit.argmax()))
            print('LABEL')
            print(label)
            print('TEXT')
            print(tokenizer.decode(input_ids))
            print('LOGITS')
            print(clean_logit)

            forbidden = np.zeros(len(input_ids)).astype('bool')
            # set [CLS] and [SEP] tokens to forbidden
            forbidden[0] = True
            forbidden[-1] = True
            offset = 0 if args.model == 'gpt2' or args.model=="roberta" or args.model=="roberta-large"  else 1
            # if "nli" in args.dataset:
            if "nli" in args.dataset:
                # set either premise or hypothesis to forbidden
                premise_length = len(tokenizer.encode(encoded_dataset[testset_key]['premise'][idx]))
                input_ids_premise = input_ids[offset:(premise_length-offset)]
                input_ids_hypothesis = input_ids[premise_length:len(input_ids)-offset]
                if args.attack_target == "hypothesis":
                    forbidden[:premise_length] = True
                else:
                    forbidden[(premise_length-offset):] = True
            forbidden_indices = np.arange(0, len(input_ids))[forbidden]
            forbidden_indices = torch.from_numpy(forbidden_indices).cuda()
            token_type_ids_batch = (None if token_type_ids is None else torch.LongTensor(token_type_ids).unsqueeze(0).repeat(args.batch_size, 1).cuda())

            start_time = time.time()
            with torch.no_grad():
                orig_output = ref_model(torch.LongTensor(input_ids).cuda().unsqueeze(0), return_dict=True).hidden_states[args.embed_layer]
                if args.constraint.startswith('bertscore'):
                    if args.constraint == "bertscore_idf":
                        ref_weights = torch.FloatTensor([idf_dict[idx] for idx in input_ids]).cuda()
                        ref_weights /= ref_weights.sum()
                    else:
                        ref_weights = None
                elif args.constraint == 'cosine':
                    # GPT-2 reference model uses last token embedding instead of pooling
                    if args.model == 'gpt2' or 'bert-base-uncased' in args.model:
                        orig_output = orig_output[:, -1]
                    else:
                        orig_output = orig_output.mean(1)
                log_coeffs = torch.zeros(len(input_ids), embeddings.size(0))
                indices = torch.arange(log_coeffs.size(0)).long()
                log_coeffs[indices, torch.LongTensor(input_ids)] = args.initial_coeff
                log_coeffs = log_coeffs.cuda()
                log_coeffs.requires_grad = True

            optimizer = torch.optim.Adam([log_coeffs], lr=args.lr)
            start = time.time()
            for i in range(args.num_iters):
                optimizer.zero_grad()
                coeffs = F.gumbel_softmax(log_coeffs.unsqueeze(0).repeat(args.batch_size, 1, 1), hard=False) # B x T x V
                inputs_embeds = (coeffs @ embeddings[None, :, :]) # B x T x D
                # if args.model in ["roberta"]:
                #     inputs_embeds1 = model._model._seq2vec_encoder(inputs_embeds)
                #     pred = model._model._classification_layer(inputs_embeds1)
                # else:
                pred = model(inputs_embeds=inputs_embeds, token_type_ids=token_type_ids_batch, return_dict=True).logits
                if args.adv_loss == 'ce':
                    adv_loss = -F.cross_entropy(pred, label * torch.ones(args.batch_size).long().cuda())
                elif args.adv_loss == 'cw':
                    top_preds = pred.sort(descending=True)[1]
                    correct = (top_preds[:, 0] == label).long()
                    indices = top_preds.gather(1, correct.view(-1, 1))
                    adv_loss = (pred[:, label] - pred.gather(1, indices) + args.kappa).clamp(min=0).mean()

                # Similarity constraint
                ref_embeds = (coeffs @ ref_embeddings[None, :, :])
                pred = ref_model(inputs_embeds=ref_embeds, return_dict=True)
                if args.lam_sim > 0:
                    output = pred.hidden_states[args.embed_layer]
                    if args.constraint.startswith('bertscore'):
                        ref_loss = -args.lam_sim * bert_score(orig_output, output, weights=ref_weights).mean()
                    else:
                        if args.model == 'gpt2' or 'bert-base-uncased' in args.model:
                            output = output[:, -1]
                        else:
                            output = output.mean(1)
                        cosine = (output * orig_output).sum(1) / output.norm(2, 1) / orig_output.norm(2, 1)
                        ref_loss = -args.lam_sim * cosine.mean()
                else:
                    ref_loss = torch.Tensor([0]).cuda()

                # (log) perple                                                                                                                                                                                                                                                                                                                                                                    xity constraint
                if args.lam_perp > 0:
                    perp_loss = args.lam_perp * log_perplexity(pred.logits, coeffs)
                else:
                    perp_loss = torch.Tensor([0]).cuda()

                # Compute loss and backward
                total_loss = adv_loss + ref_loss + perp_loss
                total_loss.backward()
                # adv_loss.backward()

                entropy = torch.sum(-F.log_softmax(log_coeffs, dim=1) * F.softmax(log_coeffs, dim=1))
                if i % args.print_every == 0:
                    print('Iteration %d: loss = %.4f, adv_loss = %.4f, ref_loss = %.4f, perp_loss = %.4f, entropy=%.4f, time=%.2f' % (
                        i+1, total_loss.item(), adv_loss.item(), ref_loss.item(), perp_loss.item(), entropy.item(), time.time() - start))

                # Gradient step
                log_coeffs.grad.index_fill_(0, forbidden_indices, 0)
                optimizer.step()

                # Log statistics
                adv_losses[idx - args.start_index, i] = adv_loss.detach().item()
                ref_losses[idx - args.start_index, i] = ref_loss.detach().item()
                perp_losses[idx - args.start_index, i] = perp_loss.detach().item()
                entropies[idx - args.start_index, i] = entropy.detach().item()
            times.append(time.time() - start_time)

            print('CLEAN TEXT')
            # if "nli" in args.dataset:
            if "nli" in args.dataset:
                clean_premise = tokenizer.decode(input_ids_premise)
                clean_hypothesis = tokenizer.decode(input_ids_hypothesis)
                clean_texts['premise'].append(clean_premise)
                clean_texts['hypothesis'].append(clean_hypothesis)
                print('%s %s' % (clean_premise, clean_hypothesis))
            else:
                clean_text = tokenizer.decode(input_ids[offset:(len(input_ids)-offset)])
                clean_texts.append(clean_text)
                print(clean_text)
            clean_logits.append(clean_logit)

            print('ADVERSARIAL TEXT')
            with (torch.no_grad()):
                for j in range(args.gumbel_samples):
                    adv_ids = F.gumbel_softmax(log_coeffs, hard=True).argmax(1)
                    # if "nli" in args.dataset:
                    if "nli" in args.dataset:
                        if args.attack_target == 'premise':
                            adv_ids_premise = adv_ids[offset:(premise_length-offset)].cpu().tolist()
                            adv_ids_hypothesis = input_ids_hypothesis
                        else:
                            adv_ids_premise = input_ids_premise
                            adv_ids_hypothesis = adv_ids[premise_length:len(adv_ids)-offset].cpu().tolist()
                        adv_premise = tokenizer.decode(adv_ids_premise)
                        adv_hypothesis = tokenizer.decode(adv_ids_hypothesis)
                        x = tokenizer(adv_premise, adv_hypothesis, max_length=256, truncation=True, return_tensors='pt')
                        token_errors.append(wer(input_ids_premise + input_ids_hypothesis, x['input_ids'][0]))
                    else:
                        adv_ids = adv_ids[offset:len(adv_ids)-offset].cpu().tolist()
                        adv_text = tokenizer.decode(adv_ids)
                        x = tokenizer(adv_text, max_length=256, truncation=True, return_tensors='pt')
                        token_errors.append(wer(adv_ids, x['input_ids'][0]))
                    if args.model == "roberta":
                        adv_logit = torch.tensor(model.predict_instance(model._dataset_reader.text_to_instance(adv_text))['logits'])
                        adv_label = model.predict_instance(model._dataset_reader.text_to_instance(adv_text))['label']

                    else:
                        adv_logit = model(input_ids=x['input_ids'].cuda(), attention_mask=x['attention_mask'].cuda(),
                                          token_type_ids=(x['token_type_ids'].cuda() if 'token_type_ids' in x else None), return_dict = True).logits.data.cpu()
                    if adv_logit.argmax() != label or j == args.gumbel_samples - 1:
                        # if "nli" in args.dataset:
                        if "nli" in args.dataset:
                            adv_texts['premise'].append(adv_premise)
                            adv_texts['hypothesis'].append(adv_hypothesis)
                            print('%s %s' % (adv_premise, adv_hypothesis))
                        else:
                            adv_texts.append(adv_text)
                            print(adv_text)
                        # adv_labels.append(adv_label)
                        adv_logits.append(adv_logit)
                        break

            # remove special tokens from adv_log_coeffs
            if "nli" in args.dataset:
                adv_log_coeffs['premise'].append(log_coeffs[offset:(premise_length-offset), :].cpu())
                adv_log_coeffs['hypothesis'].append(log_coeffs[premise_length:(log_coeffs.size(0)-offset), :].cpu())
            else:
                adv_log_coeffs.append(log_coeffs[offset:(log_coeffs.size(0)-offset), :].cpu()) # size T x V
            adv_labels.append(int(adv_logit.argmax()))
            print('')
            print('CLEAN LOGITS')
            print(clean_logit) # size 1 x C
            print('ADVERSARIAL LOGITS')
            print(adv_logit)   # size 1 x C
            # pd.DataFrame({
            #     'orig_premise': clean_texts['premise'],
            #     'orig_hypothesis': clean_texts['hypothesis'],
            #     'adv_premise': adv_texts['premise'],
            #     'adv_hypothesis': adv_texts['hypothesis'],
            # }).to_csv("snli_results_2.csv")
            # pd.DataFrame({
            #     'adv_texts': adv_texts,
            #     'clean_texts': clean_texts}).to_csv("snli_results_2.csv")
            # current_results.append(df, ignore_index=True)
            # current_results.to_csv("new_results.csv")
    except Exception as e:
        # pd.DataFrame({
        #     'adv_texts': adv_texts,
        #     'clean_texts': clean_texts}).to_csv("results.csv")
        logging.error(traceback.format_exc())
    stop1 = time.time()
    duration = stop1 - start1
    print('Time: ', duration)
    print("Token Error Rate: %.4f (over %d tokens)" % (sum(token_errors) / len(token_errors), len(token_errors)))
    # pd.DataFrame({
    #     'orig_premise': clean_texts['premise'],
    #     'orig_hypothesis': clean_texts['hypothesis'],
    #     'adv_premise': adv_texts['premise'],
    #     'adv_hypothesis': adv_texts['hypothesis'],
    #     'orig_label': pred_clean_labels,
    #     'adv_label': adv_labels
    # }).to_csv(f"{args.dataset}_gbda_{args.attack_target}.csv")
    pd.DataFrame({
        'adv_texts': adv_texts,
        'clean_texts': clean_texts}).to_csv(f"{args.dataset}_gbda_gpt2.csv")
    # torch.save({
    #     'adv_log_coeffs': adv_log_coeffs,
    #     'adv_logits': torch.cat(adv_logits, 0), # size N x C
    #     'adv_losses': adv_losses,
    #     'pred_clean_labels': pred_clean_labels,
    #     'clean_logits': torch.cat(clean_logits, 0),
    #     'clean_texts': clean_texts,
    #     'entropies': entropies,
    #     'labels': list(map(label_perm, encoded_dataset[testset_key]['label'][args.start_index:end_index])),
    #     'perp_losses': perp_losses,
    #     'ref_losses': ref_losses,
    #     'times': times,
    #     'token_error': token_errors,
    # }, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="White-box attack.")

    # Bookkeeping
    parser.add_argument("--result_folder", default="result/", type=str,
        help="folder for loading trained models")
    parser.add_argument("--gpt2_checkpoint_folder", default="result/", type=str,
        help="folder for loading GPT2 model trained with BERT tokenizer")
    parser.add_argument("--adv_samples_folder", default="adv_samples_new/", type=str,
        help="folder for saving generated samples")
    parser.add_argument("--dump_path", default="", type=str,
        help="Path to dump logs")

    # Data 
    parser.add_argument("--data_folder", required=True, type=str,
        help="folder in which to store data")
    parser.add_argument("--dataset", default="dbpedia14", type=str,
        choices=["dbpedia14", "ag_news", "imdb", "yelp", "mnli", "snli"],
        help="classification dataset to use")
    parser.add_argument("--mnli_option", default="matched", type=str,
        choices=["matched", "mismatched"],
        help="use matched or mismatched test set for MNLI")
    parser.add_argument("--num_samples", default=1, type=int,
        help="number of samples to attack")

    # Model
    parser.add_argument("--model", default="gpt2", type=str,
        help="type of model")
    parser.add_argument("--finetune", default=False, type=bool_flag,
        help="load finetuned model")

    # Attack setting
    parser.add_argument("--start_index", default=0, type=int,
        help="starting sample index")
    parser.add_argument("--num_iters", default=100, type=int,
        help="number of epochs to train for")
    parser.add_argument("--batch_size", default=10, type=int,
        help="batch size for gumbel-softmax samples")
    parser.add_argument("--attack_target", default="premise", type=str,
        choices=["premise", "hypothesis"],
        help="attack either the premise or hypothesis for MNLI")
    parser.add_argument("--initial_coeff", default=15, type=int,
        help="initial log coefficients")
    parser.add_argument("--adv_loss", default="cw", type=str,
        choices=["cw", "ce"],
        help="adversarial loss")
    parser.add_argument("--constraint", default="bertscore_idf", type=str,
        choices=["cosine", "bertscore", "bertscore_idf"],
        help="constraint function")
    parser.add_argument("--lr", default=0.3, type=float,
        help="learning rate")
    parser.add_argument("--kappa", default=5, type=float,
        help="CW loss margin")
    parser.add_argument("--embed_layer", default=-1, type=int,
        help="which layer of LM to extract embeddings from")
    parser.add_argument("--lam_sim", default=1, type=float,
        help="embedding similarity regularizer")
    parser.add_argument("--lam_perp", default=1, type=float,
        help="(log) perplexity regularizer")
    parser.add_argument("--print_every", default=10, type=int,
        help="print loss every x iterations")
    parser.add_argument("--gumbel_samples", default=100, type=int,
        help="number of gumbel samples; if 0, use argmax")

    args = parser.parse_args()
    print_args(args)
    main(args)