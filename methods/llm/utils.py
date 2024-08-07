import re
import argparse
from classifier import GoodClassifier
import pandas as pd
import numpy as np
from sentence_transformers import util
def extract_answer(answer, pattern_search):
    start_edited_pattern = r'\<new\>(.*?)(?:\<\/new\>)'
    edited_match = re.search(start_edited_pattern, answer, re.DOTALL)
    if edited_match:
        edited_text = edited_match.group(1).strip()
        if "Negative:" in edited_text:
            edited_text = edited_text.split("Negative:",1)[1]
        if "Positive:" in edited_text:
            edited_text = edited_text.split("Positive:",1)[1]
            edited_text = edited_text.strip()
        # target_match = re.search(pattern_search, edited_text, re.DOTALL)
        # if target_match:
        #     contrast_text = target_match.group(1).strip()
        # else:
        #     contrast_text = edited_text
    else:
        # target_match = re.search(pattern_search, answer)
        # if target_match:
        #     contrast_text = target_match.group(1).strip()
        # else:
        print(answer)
        edited_text = None
    return edited_text
def get_args():
     # Create the argument parser
    parser = argparse.ArgumentParser(description="generate counterfactual")

    # Add positional arguments
    parser.add_argument("-task", required=True, help="Name of the task. Currently, only IMDB and SNLI are supported.", choices=['imdb', 'snli'])

    # Add optional arguments
    parser.add_argument("-batch_size", type=int, default=100, help="Batch size for evaluation.")
    parser.add_argument("-temperature", type=float, default=1.0, help="Temperature for evaluation.")
    parser.add_argument("-method", type=str, default="method1", help="Method", choices=['method1', 'method2', 'method3', 'method4'])
    # Parse the command line arguments
    args = parser.parse_args()
    return args
def create_prompt_snli(example, target_sentence, clf: GoodClassifier = None, closest_instances = None):
    #orig_pred,contrast_pred,orig_premise,orig_hypothesis,gen_premise,gen_hypothesis,pred_orig_labels,pred_gen_premise_labels,pred_gen_hypothesis_labels
    label_map = {"entailment":1, "contradiction":0, "neutral": 2}
    example_map = {
        "neutral":{
            "premise": "Seven people are riding bikes on a sandy track.",
            "hypothesis": "The people are racing."
        },
        "entailment": {
            "premise": "Seven people are racing bikes on a sandy track.",
            "hypothesis": "People are riding bikes on a track."
        },
        "contradiction":{
            "premise": "Seven people are repairing bikes on a sandy track.",
            "hypothesis": "People are walking on a sandy track."
        }
    }
    examples=""
    if len(closest_instances) > 0:
        

        for _,instance in closest_instances.iterrows():
            if clf is not None:
                important_words, masked_input = clf.get_important_tokens(instance['sentence1'], instance['sentence2'], target_sentence=target_sentence, contrast_label=label_map[instance[f'cf_{target_sentence}_label']], max_length=64)
                important_words = "\nHere are the ordered list of important words that affect the relation: " + important_words  + ". Please change them if it is possible."
                # masked_input = f"\nHere is the sentence that has replaced important words with <mask> :\n'{masked_input}'. \nPlease fill the <mask> to get the new sentence that achieve the target."
            examples += f"""
######Start Example#######
Original label: {instance.gold_label}
Premise: {instance['sentence1']}
Hypothesis: {instance['sentence2']}
Target relation: {instance[f'cf_{target_sentence}_label']}{important_words}{masked_input}
(Edited {target_sentence}): <new>{instance[f'cf_{target_sentence}']}</new>
######End Example#######
"""





    # example_map = {
    #     "original_neutral": "The little boy in jean shorts kicks the soccer ball. A little boy is playing soccer outside.",
    #     "hypothesis_entailment": "The little boy in jean shorts kicks the soccer ball. A little boy is playing soccer..",
    #     "hypothesis_contradiction": "The little boy in jean shorts kicks the soccer ball. A little boy is playing cricket.",
    #     "premise_contradiction": "The little boy in jean shorts kicks the soccer ball in the house.	A little boy is playing soccer outside.",
    #     "premise_entailment": "The little boy in jean shorts kicks the soccer ball in the garden. A little boy is playing soccer outside."
    #     ""
    #     }
    # orig_input = eval(example['orig_input'])
    sentence_1 = example['orig_premise']
    sentence_2 = example['orig_hypothesis']
    important_words = ""
    important_words_example = ""
    important_words_guide = ""
    orig_label = example['orig_pred']
    target_label = example['contrast_pred']
    if target_sentence == "premise":
        temp = f"""Premise: {example_map[orig_label]['premise']}\nHypothesis: {example_map["neutral"]['hypothesis']}"""
    else:
        temp = f"""Premise: {example_map["neutral"]['premise']}\nHypothesis: {example_map[orig_label]['hypothesis']}"""
    if clf is not None:
        important_words_guide = "You will be provided with an ordered list of important words identified by the classifier as influential in its prediction (First word is the most important). Use these words as a guide for your changes."
        important_words, masked_input = clf.get_important_tokens(sentence_1, sentence_2, target_sentence=target_sentence, contrast_label=label_map[target_label], max_length=64)
        important_words = "\nHere are the important words that affect the relation: " + important_words  + ". Please change them if it is possible."
        important_words_example, masked_input_example = clf.get_important_tokens(temp.split(".")[0]+".", temp.split(".")[1]+".", target_sentence=target_sentence, contrast_label=label_map[target_label], max_length=64)
        important_words_example = "\nHere are the ordered list of important words that affect the relation: " + important_words_example + ". Please change them if it is possible."
        masked_input_example = f"\nHere is the sentence that has replaced important words with <mask> :\n'{masked_input_example}'. \nPlease fill the <mask> to get the new sentence that achieve the target."
        # masked_input = f"\nHere is the sentence that has replaced important words with <mask> :\n'{masked_input}'. \nPlease fill the <mask> to get the new sentence that achieve the target."
#     template = f"""Request: Given two sentences (premise and hypothesis) and their original label classified by a classifier, determine whether they entail, contradict, or are neutral to each other. Change the {target_sentence} with minimal edits to achieve the {target_label} label from the original one. Do not make any unnecessary changes.
# {important_words_guide}
# Original label: {orig_label}
# {temp}
# Target label: {target_label}{important_words_example}
# (Edited {target_sentence}): <new>{example_map[target_label][target_sentence]}</new>
# ######End Example#######

# Request: Similarly, given two sentences (premise and hypothesis) and their original relationship, determine whether they entail, contradict, or are neutral to each other. Change the {target_sentence} with minimal edits to achieve the {target_label} relation from the original one.
# {important_words_guide}
# Original label: {orig_label}
# [Start Original Text]
# Premise: {sentence_1}
# Hypothesis: {sentence_2}
# [End Original Text]
# Target label: {target_label}{important_words}
# Do not make any unneccesary changes. Enclose the generated sentence within <new> tags. Do not add anything else. Make as few edits as possible. Give me only the sentence with tags.
# Your sentence:
# """
    template = f"""Request:  Given two sentences (premise and hypothesis) and their original relationship, determine whether they entail, contradict, or are neutral to each other. Change the {target_sentence} with minimal edits to achieve the {target_label} relation from the original one. Do not make any unnecessary changes.
{important_words_guide}
Original relation: {orig_label}
{temp}
Target relation: {target_label}{important_words_example}{masked_input_example}
(Edited {target_sentence}): <new>{example_map[target_label][target_sentence]}</new>
######End Example#######
{examples}
Request: Similarly, given two sentences (premise and hypothesis) and their original relationship, determine whether they entail, contradict, or are neutral to each other. Change the {target_sentence} with minimal edits to achieve the {target_label} relation from the original one.
{important_words_guide}
Original relation: {orig_label}
[Start Original Text]
Premise: {sentence_1}
Hypothesis: {sentence_2}
[End Original Text]
Target relation: {target_label}{important_words}{masked_input}
Do not make any unneccesary changes. Enclose the generated sentence within <new> tags. Do not add anything else. Make as few edits as possible. Give me only the sentence with tags.
Your sentence:"""
    template = [{
            "role": "user",
            "content": template
        }]
    return template
def create_prompt_snli_test(example, target_sentence, clf: GoodClassifier = None, closest_instances = None):
    #orig_pred,contrast_pred,orig_premise,orig_hypothesis,gen_premise,gen_hypothesis,pred_orig_labels,pred_gen_premise_labels,pred_gen_hypothesis_labels
    label_map = {"entailment":1, "contradiction":0, "neutral": 2}
    example_map = {
        "neutral":{
            "premise": "Seven people are riding bikes on a sandy track.",
            "hypothesis": "The people are racing."
        },
        "entailment": {
            "premise": "Seven people are racing bikes on a sandy track.",
            "hypothesis": "People are riding bikes on a track."
        },
        "contradiction":{
            "premise": "Seven people are repairing bikes on a sandy track.",
            "hypothesis": "People are walking on a sandy track."
        }
    }
    examples=""
    if len(closest_instances) > 0:
        

        for _,instance in closest_instances.iterrows():
            if clf is not None:
                important_words, masked_input = clf.get_important_tokens(instance['sentence1'], instance['sentence2'], target_sentence=target_sentence, contrast_label=label_map[instance[f'cf_{target_sentence}_label']], max_length=64)
                important_words = "\nHere are the ordered list of important words that affect the relation: " + important_words  + ". Please change them if it is possible."
                # masked_input = f"\nHere is the sentence that has replaced important words with <mask> :\n'{masked_input}'. \nPlease fill the <mask> to get the new sentence that achieve the target."
            if target_sentence == "premise":
                examples += f"""
    ######Start Example#######
    Original label: {instance.gold_label}
    Premise: {masked_input}
    Hypothesis: {instance['sentence2']}
    Target relation: {instance[f'cf_{target_sentence}_label']}
    Please fill the <mask> to get the new sentence that achieve the target.
    (Edited {target_sentence}): <new>{instance[f'cf_{target_sentence}']}</new>
    ######End Example#######
    """
            else:
                 examples += f"""
    ######Start Example#######
    Original label: {instance.gold_label}
    Premise: {instance['sentence1']}
    Hypothesis: {masked_input}
    Target relation: {instance[f'cf_{target_sentence}_label']}
    Please fill the <mask> to get the new sentence that achieve the target.
    (Edited {target_sentence}): <new>{instance[f'cf_{target_sentence}']}</new>
    ######End Example#######
    """               





    # example_map = {
    #     "original_neutral": "The little boy in jean shorts kicks the soccer ball. A little boy is playing soccer outside.",
    #     "hypothesis_entailment": "The little boy in jean shorts kicks the soccer ball. A little boy is playing soccer..",
    #     "hypothesis_contradiction": "The little boy in jean shorts kicks the soccer ball. A little boy is playing cricket.",
    #     "premise_contradiction": "The little boy in jean shorts kicks the soccer ball in the house.	A little boy is playing soccer outside.",
    #     "premise_entailment": "The little boy in jean shorts kicks the soccer ball in the garden. A little boy is playing soccer outside."
    #     ""
    #     }
    # orig_input = eval(example['orig_input'])
    sentence_1 = example['orig_premise']
    sentence_2 = example['orig_hypothesis']
    important_words = ""
    important_words_example = ""
    important_words_guide = ""
    orig_label = example['orig_pred']
    target_label = example['contrast_pred']
    if target_sentence == "premise":
        temp = f"""Premise: {example_map[orig_label]['premise']}\nHypothesis: {example_map["neutral"]['hypothesis']}"""
    else:
        temp = f"""Premise: {example_map["neutral"]['premise']}\nHypothesis: {example_map[orig_label]['hypothesis']}"""
    if clf is not None:
        important_words_guide = "You will be provided with an ordered list of important words identified by the classifier as influential in its prediction (First word is the most important). Use these words as a guide for your changes."
        important_words, masked_input = clf.get_important_tokens(sentence_1, sentence_2, target_sentence=target_sentence, contrast_label=label_map[target_label], max_length=64)
        important_words = "\nHere are the important words that affect the relation: " + important_words  + ". Please change them if it is possible."
        important_words_example, masked_input_example = clf.get_important_tokens(temp.split(".")[0]+".", temp.split(".")[1]+".", target_sentence=target_sentence, contrast_label=label_map[target_label], max_length=64)
        important_words_example = "\nHere are the ordered list of important words that affect the relation: " + important_words_example + ". Please change them if it is possible."
        masked_input_example = f"\nHere is the sentence that has replaced important words with <mask> :\n'{masked_input_example}'. \nPlease fill the <mask> to get the new sentence that achieve the target."
        # masked_input = f"\nHere is the sentence that has replaced important words with <mask> :\n'{masked_input}'. \nPlease fill the <mask> to get the new sentence that achieve the target."
#     template = f"""Request: Given two sentences (premise and hypothesis) and their original label classified by a classifier, determine whether they entail, contradict, or are neutral to each other. Change the {target_sentence} with minimal edits to achieve the {target_label} label from the original one. Do not make any unnecessary changes.
# {important_words_guide}
# Original label: {orig_label}
# {temp}
# Target label: {target_label}{important_words_example}
# (Edited {target_sentence}): <new>{example_map[target_label][target_sentence]}</new>
# ######End Example#######

# Request: Similarly, given two sentences (premise and hypothesis) and their original relationship, determine whether they entail, contradict, or are neutral to each other. Change the {target_sentence} with minimal edits to achieve the {target_label} relation from the original one.
# {important_words_guide}
# Original label: {orig_label}
# [Start Original Text]
# Premise: {sentence_1}
# Hypothesis: {sentence_2}
# [End Original Text]
# Target label: {target_label}{important_words}
# Do not make any unneccesary changes. Enclose the generated sentence within <new> tags. Do not add anything else. Make as few edits as possible. Give me only the sentence with tags.
# Your sentence:
# """
    template = f"""Request:  Given two sentences (premise and hypothesis) and their original relationship, determine whether they entail, contradict, or are neutral to each other. Change the {target_sentence} with minimal edits to achieve the {target_label} relation from the original one. Do not make any unnecessary changes.
{important_words_guide}
Original relation: {orig_label}
{temp}
Target relation: {target_label}{important_words_example}{masked_input_example}
(Edited {target_sentence}): <new>{example_map[target_label][target_sentence]}</new>
######End Example#######
{examples}
Request: Similarly, given two sentences (premise and hypothesis) and their original relationship, determine whether they entail, contradict, or are neutral to each other. Change the {target_sentence} with minimal edits to achieve the {target_label} relation from the original one.
{important_words_guide}
Original relation: {orig_label}
[Start Original Text]
Premise: {sentence_1}
Hypothesis: {sentence_2}
[End Original Text]
Target relation: {target_label}{important_words}{masked_input}
Do not make any unneccesary changes. Enclose the generated sentence within <new> tags. Do not add anything else. Make as few edits as possible. Give me only the sentence with tags.
Your sentence:"""
    template = [{
            "role": "user",
            "content": template
        }]
    return template

def create_prompt_imdb(example, clf: GoodClassifier = None):
    contrast_map = {"Positive": "Negative", "Negative": "Positive"}
    label_map = {0: "Negative", 1: "Positive"}
    example_map = {
        "Negative": "Long, boring, blasphemous. Never have I been so glad to see ending credits roll.",
        "Positive": "Long, fascinating, soulful. Never have I been so sad to see ending credits roll."
                   }
    orig_sent = example['pred_orig_labels']
    contrast_label = 1-int(orig_sent)
    orig_sent = label_map[orig_sent]
    important_tokens_example = ""
    important_tokens = ""
    important_sentence = ""
    if clf is not None:
        important_tokens_example = clf.get_important_tokens(example_map[orig_sent],contrast_label=contrast_label)
        important_tokens_example = f"Important words: {important_tokens_example}"
        important_tokens = clf.get_important_tokens(example['orig_text'],contrast_label=contrast_label)
        important_tokens = f"Important words: {important_tokens}"
        important_sentence ="You will be provided with an ordered list of important words identified by the classifier as influential in its prediction (First word is the most important). Use these words as a guide for your changes, ensuring that the text remains fluent. Avoid making any unnecessary changes."
    template = f"""Context:
We need to generate the counterfactual text for a movie review
Task:
Given a movie review with its original sentiment classified as either positive or negative by a classifier, your task is to modify the text with minimal edits to flip the sentiment prediction of the classifier. Please enclose the generated review within <new></new> tags\\
{important_sentence}
Example:
######Start Example#######
{orig_sent}: {example_map[orig_sent]}
Target sentiment: {contrast_map[orig_sent]}
{important_tokens_example}
{contrast_map[orig_sent]}: <new>{example_map[contrast_map[orig_sent]]}</new>
######End Example#######
Your turn:
[Start Original Text]
{orig_sent}: {example['orig_text']}
[End Original Text]
Target sentiment: {contrast_map[orig_sent]}
{important_tokens}
Do not make any unnecessary changes.  Make as few edits as possible but ensure the text remains fluent. Please enclose the generated review within <new></new> tags like the given example.
"""
    template = [{
        "role": "user",
        "content": template
    }]
    
    return template
# def create_prompt_snli_mask(example, target_sentence, clf: GoodClassifier = None, closest_instances = None):
#     label_map = {"entailment":1, "contradiction":0, "neutral": 2}
#     template = f"""Request:  Given two sentences (premise and hypothesis) and their original relationship, determine whether they entail, contradict, or are neutral to each other. Change the {target_sentence} with minimal edits to achieve the {target_label} relation from the original one. Do not make any unnecessary changes.
# {important_words_guide}
# Original relation: {orig_label}
# {temp}
# Target relation: {target_label}{important_words_example}
# (Edited {target_sentence}): <new>{example_map[target_label][target_sentence]}</new>
# ######End Example#######
# {examples}
# Request: Similarly, given two sentences (premise and hypothesis) and their original relationship, determine whether they entail, contradict, or are neutral to each other. Change the {target_sentence} with minimal edits to achieve the {target_label} relation from the original one.
# {important_words_guide}
# Original relation: {orig_label}
# [Start Original Text]
# Premise: {sentence_1}
# Hypothesis: {sentence_2}
# [End Original Text]
# Target relation: {target_label}{important_words}
# Do not make any unneccesary changes. Enclose the generated sentence within <new> tags. Do not add anything else. Make as few edits as possible. Give me only the sentence with tags.
# Your sentence:"""
#     template = [{
#             "role": "user",
#             "content": template
#         }]
#     return template

def load_data(task):
    if task == "imdb":
        df = pd.read_csv(f"results/imdb/expert/results_auto_eval.csv")
        df = df[['orig_text','pred_orig_labels']]
    else:
        df = pd.read_csv(f"results/snli/mice/results.csv")
        df = df[["orig_pred","pred_orig_labels","contrast_pred","orig_premise","orig_hypothesis",]]
        df['concat_sentence'] = df.apply(lambda x: x['orig_premise'] + " " + x['orig_hypothesis'] if x['orig_premise'][-1] == "." else x['orig_premise'] + ". " + x['orig_hypothesis'], axis=1)
    return df
def load_pattern_search(task, target_sentence = None):
    if task == "imdb":
        pattern_search = r'(?:Positive|Negative): (.*?)(?:\n|$)'
    else:
        pattern_search = rf"(?:\(Edited {target_sentence}\)): (.*?)(?:\n|$)"
    return pattern_search


def filter_correct_predict_dataset(data):
    """
    Input: dataframe with orig_text and gen_text
    Output: dataframe with filter instances that the f(gen_text) != f(orig_text)
    """
def load_embeddings(sbert_model, task="snli", target_sentence = "premise"):
    # dev_hypothesis = pd.read_csv("datasets/NLI/revised_hypothesis/dev.tsv", delimiter="\t")
    # dev_premise = pd.read_csv("datasets/NLI/revised_premise/dev.tsv", delimiter="\t")
    dev_pairs = pd.read_csv("methods/llm/dev_merge_preds.csv")
    dev_pairs = dev_pairs[dev_pairs['orig_preds'] != dev_pairs[f'{target_sentence}_preds']]
    # dev_pairs = dev_pairs.merge(dev_hypothesis)∂
    dev_pairs['concat_sentence'] = dev_pairs.apply(lambda x: x['sentence1'] + x['sentence2'], axis=1)
    dev_neutral_pairs = dev_pairs[dev_pairs['gold_label'] == "neutral"]
    dev_entailment_pairs = dev_pairs[dev_pairs['gold_label'] == "entailment"]
    dev_contradiction_pairs = dev_pairs[dev_pairs['gold_label'] == "contradiction"]
    neutral_embeddings = sbert_model.encode(dev_neutral_pairs['concat_sentence'].to_list())
    entailment_embeddings = sbert_model.encode(dev_entailment_pairs['concat_sentence'].to_list())
    contradiction_embeddings = sbert_model.encode(dev_contradiction_pairs['concat_sentence'].to_list())
    label_to_embeddings = {"neutral":neutral_embeddings,
                    "entailment":entailment_embeddings,
                    "contradiction":contradiction_embeddings}
    label_to_sentences= {"neutral":dev_neutral_pairs,
                        "entailment":dev_entailment_pairs,
                        "contradiction":dev_contradiction_pairs}
    return label_to_embeddings, label_to_sentences
# def extract_closest_instances(instance, label_to_embeddings, label_to_sentences, number=5, task="snli"):
#     if task == "snli":
#         find_closest_instance(sbert_model.encode(instance['concat_sentence']), dev_embeddings)
#         dev_embeddings = label_to_embeddings[instance['gold_label']]
#         dev_sentences = label_to_sentences[instance['gold_label']]
def find_closest_indices(example, embeddings, number = 5):
    """
    select the example (pair of original text and counterfactual text) in the training and dev set that has the closest semantic meaning to the current example
    input: pair [original_label, original_text, cf_label, cf_text]
    output: closest pair
    requirement: need to have the same label
    """
    cos_sim_scores = util.cos_sim(example, embeddings)
    score_sorted = np.argsort(cos_sim_scores)
    indices = score_sorted[0][-number:]
    return indices