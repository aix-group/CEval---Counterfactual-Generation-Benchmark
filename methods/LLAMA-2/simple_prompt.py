import pandas as pd
from transformers import AutoTokenizer
import transformers
import torch
import time
import re
import argparse
from datetime import datetime
current_date = datetime.now()
date_string = current_date.strftime("%Y%m%d%H%M%S")
def create_prompt_snli(example, target_sentence):
    #orig_pred,contrast_pred,orig_premise,orig_hypothesis,gen_premise,gen_hypothesis,pred_orig_labels,pred_gen_premise_labels,pred_gen_hypothesis_labels
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
    orig_label = example['orig_pred']
    target_label = example['contrast_pred']
    if target_sentence == "premise":
        temp = f"""Premise: {example_map[orig_label]['premise']}\nHypothesis: {example_map["neutral"]['hypothesis']}"""
    else:
        temp = f"""Premise: {example_map["neutral"]['premise']}\nHypothesis: {example_map[orig_label]['hypothesis']}"""

#     template = f"""<s>[INST] <<SYS>>
# Given two sentences (premise and hypothesis) and their original relation, determine whether they entail, contradict, or are neutral to each other. Change the {target_sentence} with minimal edits to achieve the target relation from the original one. Do not make any unnecessary changes.
# <</SYS>>
# Original relation: {orig_label}
# {temp}
        
# Target relation: {target_label}
# Edited {target_sentence}:[/INST]{example_map[target_label][target_sentence]} </s><s>[INST] 
# Original relation: {orig_label}
# Premise: {sentence_1}
# Hypothesis: {sentence_2}
# Target relation: {target_label}
# Edited {target_sentence}:[/INST]"""
    template = f"""Request:  Given two sentences (premise and hypothesis) and their original relationship, determine whether they entail, contradict, or are neutral to each other. Change the {target_sentence} with minimal edits to achieve the {target_label} relation from the original one. Do not make any unnecessary changes.
Original relation: {orig_label}
{temp}
Target relation: {target_label}
<new>(Edited {target_sentence}): {example_map[target_label][target_sentence]}</new>
######End Example#######

Request: Similarly, given two sentences (premise and hypothesis) and their original relationship, determine whether they entail, contradict, or are neutral to each other. Change the {target_sentence} with minimal edits to achieve the {target_label} relation from the original one.
Original relation: {orig_label}
[Start Original Text]
Premise: {sentence_1}
Hypothesis: {sentence_2}
[End Original Text]
Target relation: {target_label}
Do not make any unneccesary changes. Enclose the generated sentence within <new> tags. Do not add anything else. Make as few edits as possible. Give me only the sentence with tags."""
#     template = f"""In the task of snli, a trained black-box classifier correctly predicted the label {orig_label}
# for the following text. Generate a counterfactual explanation by making minimal changes to the input text,
# so that the label changes from {orig_label} to {target_label}. Use the following definition of ‘counterfactual explanation’:
# “A counterfactual explanation reveals what should have been different in an instance to observe a diverse
# outcome." Enclose the generated text within <new> tags.\n—\nText: {sentence_1} {sentence_2}."""
    template = [{
            "role": "user",
            "content": template
        }]
    return template
def create_prompt(example):
    contrast_map = {"Positive": "Negative", "Negative": "Positive"}
    example_map = {
        "Negative": "Long, boring, blasphemous. Never have I been so glad to see ending credits roll.",
        "Positive": "Long, fascinating, soulful. Never have I been so sad to see ending credits roll."
                   }
    orig_sent = example['Sentiment']
    template = f"""Request: Given a piece of text with the original sentiment in the form of "Sentiment: Text". Change the text with minimal edits to get the target sentiment from the original sentiment. Do not make any unneccesary changes. For example:
{orig_sent}: {example_map[orig_sent]}
Target: {contrast_map[orig_sent]}
<new>{contrast_map[orig_sent]}: {example_map[contrast_map[orig_sent]]}</new>
######End Example#######
Request: Similarly, given a piece of text below with the original sentiment in the form of "Sentiment: Text". Change the text with "minimal edits" to get the {contrast_map[orig_sent]} sentiment from the {orig_sent} sentiment. 
[Start Original Text]
{orig_sent}: {example['Text']}
[End Original Text]
Target: {contrast_map[orig_sent]}
Do not make any unneccesary changes. Enclose the generated text within <new> tags. Do not add anything else. Make as few edits as possible.
"""
    template = [{
        "role": "user",
        "content": template
    }]
    return template
def get_args():
    # Create the argument parser
    parser = argparse.ArgumentParser(description="generate counterfactual")

    # Add positional arguments
    parser.add_argument("-task", required=True, help="Name of the task. Currently, only IMDB and SNLI are supported.", choices=['imdb', 'snli'])

    # Add optional arguments
    parser.add_argument("-batch_size", type=int, default=100, help="Batch size for evaluation.")
    parser.add_argument("-temperature", type=float, default=0.2, help="Temperature for evaluation.")

    # Parse the command line arguments
    args = parser.parse_args()
    return args
if __name__ == '__main__':
    args = get_args()
    batch_size = args.batch_size
    task = args.task
    temperature = args.temperature
    llm_model = "meta-llama/Llama-2-7b-chat-hf"
    # load dataset
    if task == "imdb":
        df = pd.read_csv(f"datasets/imdb/expert/test_original.tsv", delimiter="\t")
        df['Text'] = df['Text'].apply(lambda x: x.replace("<br /><br />"," "))
    else:
        df = pd.read_csv(f"results/snli/mice/results.csv")
        df = df[["orig_pred","contrast_pred","orig_premise","orig_hypothesis",]]
        # df = df.iloc[:10]
    tokenizer = AutoTokenizer.from_pretrained(llm_model)
    llm_pipeline = transformers.pipeline(
        "text-generation",
        model=llm_model,
        torch_dtype=torch.float16,
        device_map="auto",
        tokenizer = tokenizer
    )

    #Negative	Long, boring, blasphemous. Never have I been so glad to see ending credits roll.	4
    #Positive	Long, fascinating, soulful. Never have I been so sad to see ending credits roll.
    #split sentiment pairs

    #generate sentiment embeddings for each sentiments:

    
    # list_prompts = []
    # df_test = test_pairs
    #pick an instance
    # Specify the chunk size
    max_new_token = 64 if task == "snli" else 768 
    start_edited_pattern = r'\<new\>(.*?)(?:\<\/new\>)'
    
    start_time = time.time()
    target_sentences = ["premise", "hypothesis"] if task == "snli" else [None]
    for target_sentence in target_sentences:
        list_contrast_texts = []
        if task == "imdb":
            pattern_search = r'(?:Positive|Negative): (.*?)(?:\n|$)'
        else:
            pattern_search = rf"(?:\(Edited {target_sentence}\)): (.*?)(?:\n|$)"
        for i in range(0, df.shape[0], batch_size):
            batch = df.iloc[i:i+batch_size]
            list_chunk_prompts = []
            for index, example in batch.iterrows():
                if task == "snli":
                    prompt = create_prompt_snli(example, target_sentence)
                else:
                    prompt = create_prompt(example)
                # list_prompts.append(prompt)
                list_chunk_prompts.append(prompt)
            
            sequences = llm_pipeline(
                list_chunk_prompts,
                do_sample=True,
                top_k=50,
                num_return_sequences=1,
                max_new_tokens=max_new_token,
                temperature = temperature,
            )
            
            for seq in sequences:
                prompt = seq[0]['generated_text'][0]['content']
                answer = seq[0]['generated_text'][1]['content']

                #Write to file
                with open(f"raw_text_{task}_llama_{date_string}_{temperature}_{target_sentence}.txt", 'a') as fp:
                    fp.write("[start prompt]%s[end prompt]\n" % prompt)
                    fp.write("[start answer]%s[end answer]\n" % answer)

                #Extract answer
                edited_match = re.search(start_edited_pattern, answer, re.DOTALL)
                if edited_match:
                    edited_text = edited_match.group(1).strip()
                    target_match = re.search(pattern_search, edited_text, re.DOTALL)
                    if target_match:
                        contrast_text = target_match.group(1).strip()
                    else:
                        contrast_text = edited_text
                else:
                    target_match = re.search(pattern_search, answer)
                    if target_match:
                        contrast_text = target_match.group(1).strip()
                    else:
                        print(answer)
                        contrast_text = None


                list_contrast_texts.append(contrast_text)


        end_time = time.time()
        if target_sentence == None:
            column_name = f"llama_text_{temperature}"   
        else:
            column_name = f"llama_text_{temperature}_{target_sentence}"
        file_name = f"results/{task}/llama/llama_2_{task}_{temperature}_{date_string}.csv"
        df[column_name] = list_contrast_texts
    
    df.to_csv(file_name)
    duration = end_time - start_time
    print(duration)
    with open(f"llama_2_{date_string}.txt", 'w') as fp:
        fp.write("Duration: %s" % str(duration))