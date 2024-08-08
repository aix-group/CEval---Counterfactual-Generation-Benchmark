import os
import argparse
import pandas as pd
import torch
from transformers import BertForSequenceClassification, AutoTokenizer
from metrics.auto_metrics import *
import torch.nn.functional as F

def load_classifier(model_name):
    classifier = BertForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return classifier, tokenizer

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def get_args():
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Benchmark arguments")

    # Add positional arguments
    parser.add_argument("-csv_path", required=True, help="Path to the CSV file containing the original and generated text.")
    parser.add_argument("-task", required=True, help="Name of the task. Currently, only IMDB and SNLI are supported.", choices=['imdb', 'snli'])

    # Add optional arguments
    parser.add_argument("-batch_size", type=int, default=100, help="Batch size for evaluation.")
    parser.add_argument("-return_csv", action="store_true", help="Whether to save the metrics to the original CSV file.")

    # Parse the command line arguments
    args = parser.parse_args()
    return args
if __name__ == '__main__':
    args = get_args()
    # Load model
    model_name = f"textattack/bert-base-uncased-{args.task}"
    classifier, tokenizer = load_classifier(model_name)

    # Load metrics
    metrics = Metrics()

    #Load data
    df = pd.read_csv(args.csv_path)
    df = df.dropna()
    # df = df[df['gen_text'] != "-1"]
    #Calculate metrics
    if args.task == "imdb":
        try:
            list_orig_texts = df['orig_text'].to_list()
            list_gen_texts = df['gen_text'].to_list()
        except KeyError as e:
            print(f"Error: {e} key not found. Make sure you provided the correct format of the CSV file. You need to have 2 columns: orig_text and gen_text with a delimiter as a comma.")
        pred_orig_labels = []
        pred_edit_labels = []
        contrast_class_ids = []
        orig_contrast_probs = []
        pred_probs  = [] 
        # Predict original text
        for b in batch(list_orig_texts, args.batch_size):
            inputs = tokenizer(b, return_tensors="pt", padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                logits = classifier(**inputs).logits
            probabilities = F.softmax(logits, dim=1)
            predicted_class_ids = torch.argmax(logits, axis=1).tolist()
            contrast_class_ids+=torch.argmin(logits, axis=1).tolist()
            orig_contrast_probs+=torch.min(probabilities, axis=1).values.tolist()
            pred_orig_labels.extend(predicted_class_ids)

        #Predict generated Text
        for b in batch(list_gen_texts, args.batch_size):
            inputs = tokenizer(b, return_tensors="pt", padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                logits = classifier(**inputs).logits
            probabilities = F.softmax(logits, dim=1)
            pred_probs+=probabilities
            
            predicted_class_ids = torch.argmax(logits, axis=1).tolist()
            pred_edit_labels.extend(predicted_class_ids)
        diff_probs =  [float(tensor[label]) - org_prob for org_prob, tensor, label in zip(orig_contrast_probs, pred_probs, contrast_class_ids)]
        predict_contrast_prob = [float(tensor[label]) for  tensor, label in zip(pred_probs, contrast_class_ids)]
        # pred_edit_labels.extend([i['label'] for i in predictor.predict_batch_instance(b)])
        df['pred_orig_labels'] = pred_orig_labels
        df['pred_gen_labels'] = pred_edit_labels
        df['orig_contrast_probs'] = orig_contrast_probs
        df['predict_contrast_prob'] = predict_contrast_prob
        df['distance'] = df.apply(lambda x: metrics.score_minimality(x['orig_text'], x['gen_text']), axis=1)
        if 'gen_text_2' in df:
            df['diversity'] = df.apply(lambda x: metrics.score_minimality(x['gen_text'], x['gen_text_2']), axis=1)
        df['perplexity'] = metrics.score_perplexity(list_gen_texts)['perplexities']
        df['diff_probs'] = diff_probs
        df_contrast = df[df['pred_orig_labels'] != df['pred_gen_labels']]
        flip_rate = len(df_contrast) / len(df)
        print(f"Flip Rate: {round(flip_rate, 2)}")
        for name, df_temp in zip(["all","contrast"],[df, df_contrast]):
            mean_distance = round(df_temp['distance'].mean(), 2)
            mean_perplexity = round(df_temp['perplexity'].mean(), 2)
            mean_diff_probs =  round(df_temp['diff_probs'].mean(), 2)
            
            print(f"Probability Change {name}: {mean_diff_probs}")  
            print(f"Token Distance {name}: {mean_distance}")
            print(f"Perplexity {name}: {mean_perplexity}") 
            if 'gen_text_2' in df:
                df_temp = df_temp[df_temp['gen_text_2'] != "-1"]
                df_temp = df_temp[df_temp['gen_text'] != "-1"]
                mean_diversity =  round(df_temp['diversity'].mean(), 2)
                print(f"Diversity {name}: {mean_diversity}")
            print("-----------")

            
    else:
        list_orig_premise = df['orig_premise'].to_list()
        list_orig_hypothesis = df['orig_hypothesis'].to_list()
        list_gen_hypothesis= df['gen_hypothesis'].to_list()
        list_gen_premise= df['gen_premise'].to_list()

        #comment
        # label_map = {'contradiction': 0, 'entailment': 1, 'neutral': 2}
        # list_contrast_labels = df_original['contrast_pred'].apply(lambda x: label_map[x])

        pred_orig_labels = []
        pred_edit_premise_labels = []
        pred_edit_hypothesis_labels = []

        pred_edit_premise_probs = []
        pred_edit_hypo_probs = []
        pred_orig_probs = []
        inputs = tokenizer(list_orig_premise,list_gen_hypothesis, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            logits = classifier(**inputs).logits
        predicted_class_ids = torch.argmax(logits, axis=1).tolist()
        pred_edit_hypo_probs = torch.max(F.softmax(logits, dim=1), axis=1).values.tolist()
        pred_edit_hypothesis_labels.extend(predicted_class_ids)
        inputs = tokenizer(list_gen_premise,list_orig_hypothesis, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            logits = classifier(**inputs).logits
        predicted_class_ids = torch.argmax(logits, axis=1).tolist()
        pred_edit_premise_probs = torch.max(F.softmax(logits, dim=1), axis=1).values.tolist()
        pred_edit_premise_labels.extend(predicted_class_ids)

        inputs = tokenizer(list_orig_premise,list_orig_hypothesis, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            logits = classifier(**inputs).logits
        predicted_class_ids = torch.argmax(logits, axis=1).tolist()
        pred_orig_labels.extend(predicted_class_ids)
        pred_orig_probs += F.softmax(logits, dim=1)
        diff_premise_probs =  [prob- float(tensor[label])  for tensor, prob, label in zip(pred_orig_probs, pred_edit_premise_probs, pred_edit_premise_labels)]
        diff_hypothesis_probs =  [prob- float(tensor[label])  for tensor, prob, label in zip(pred_orig_probs, pred_edit_hypo_probs, pred_edit_hypothesis_labels)]

        
        # orig_contrast_probs =  [float(tensor[label])  for tensor, label in zip(pred_orig_probs,  list_contrast_labels)]
        #plot
        # df['orig_contrast_probs'] = orig_contrast_probs
        # df['pred_edit_premise_probs'] = pred_edit_premise_probs
        # df['pred_edit_hypo_probs'] = pred_edit_hypo_probs

        df['pred_orig_labels'] = pred_orig_labels
        df['pred_gen_premise_labels'] = pred_edit_premise_labels
        df['pred_gen_hypothesis_labels'] = pred_edit_hypothesis_labels
        df['distance_premise'] = df.apply(lambda x: metrics.score_minimality(x['orig_premise'], x['gen_premise']), axis=1)
        df['distance_hypothesis'] = df.apply(lambda x: metrics.score_minimality(x['orig_hypothesis'], x['gen_hypothesis']), axis=1)
        list_combine = [f"{premise} {hypothesis}" for premise, hypothesis in zip(list_orig_premise, list_gen_hypothesis)]
        df['perplexity_gen_hypothesis'] = metrics.score_perplexity(list_combine)['perplexities']
        list_combine = [f"{premise} {hypothesis}" for premise, hypothesis in zip(list_gen_premise, list_orig_hypothesis)]
        df['perplexity_gen_premise'] = metrics.score_perplexity(list_combine)['perplexities']
        df['diff_premise_probs'] = diff_premise_probs
        df['diff_hypothesis_probs'] = diff_hypothesis_probs
        df_contrast_premise = df[df['pred_orig_labels'] != df['pred_gen_premise_labels']]
        df_contrast_hypothesis= df[df['pred_orig_labels'] != df['pred_gen_hypothesis_labels']]
        flip_rate_premise = len(df_contrast_premise) / len(df)
        flip_rate_hypothesis = len(df_contrast_hypothesis) / len(df)
        print(f"Flip Rate Premise: {round(flip_rate_premise, 2)}")
        print(f"Flip Rate Hypothesis: {round(flip_rate_hypothesis, 2)}")
        print(f"Flip Rate Both: {round((flip_rate_premise + flip_rate_hypothesis)/2*100, 2)}%")
        for name, df_temp in zip(["all","contrast"],[df, df_contrast_premise]):
            mean_distance_premise = round(df_temp['distance_premise'].mean(), 2)
            mean_perplexity_premise = round(df_temp['perplexity_gen_premise'].mean(), 2)
            mean_diff_premise_probs=  round(df_temp['diff_premise_probs'].mean(), 2)
            print(f"Probability Change Premise {name}: {mean_diff_premise_probs}")  
            print(f"Token Distance Premise {name}: {mean_distance_premise}")
            print(f"Perplexity Premise {name}: {mean_perplexity_premise}") 
            print("-----------")
        for name, df_temp in zip(["all","contrast"],[df, df_contrast_hypothesis]):
            mean_distance_hypothesis = round(df_temp['distance_hypothesis'].mean(), 2)
            mean_perplexity_hypothesis= round(df_temp['perplexity_gen_hypothesis'].mean(), 2)
            mean_diff_hypothesis_probs=  round(df_temp['diff_hypothesis_probs'].mean(), 2)
            print(f"Probability Change Hypothesis {name}: {mean_diff_hypothesis_probs}")  
            print(f"Token Distance Hypothesis {name}: {mean_distance_hypothesis}")
            print(f"Perplexity Hypothesis {name}: {mean_perplexity_hypothesis}")
            print("-----------")
        prob_all = round((df["diff_premise_probs"].mean()+ df["diff_hypothesis_probs"].mean())/2,2)
        pp_all = round((df["perplexity_gen_premise"].mean()+ df["perplexity_gen_hypothesis"].mean())/2,2)
        dist_all = round((df["distance_premise"].mean()+ df["distance_hypothesis"].mean())/2,2)
        print(f"Probability changes both all: {prob_all}" )
        print(f"Perplexity changes both all: {pp_all}" )
        print(f"Distance both all: {dist_all}" )

    if args.return_csv:
        df.to_csv(f"{args.csv_path.split('.')[0]}_auto_eval.csv", index=False)