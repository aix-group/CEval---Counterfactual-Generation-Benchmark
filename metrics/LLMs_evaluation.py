
import argparse
import transformers
import torch
import json
import pandas as pd
from openai import OpenAI
import time
from itertools import zip_longest
from transformers import AutoTokenizer
from datetime import datetime
import httpx
current_date = datetime.now()
date_string = current_date.strftime("%Y%m%d%H%M%S")

def cut_continuous_repetitions(string, max_repetitions):
    result = ""
    count = 1  # Initialize count to 1 to account for the first occurrence

    for i in range(1, len(string)):
        if string[i] == string[i - 1]:
            count += 1
            if count <= max_repetitions:
                result += string[i]
        else:
            count = 1
            result += string[i]

    return result


class LLMEvaluator:
    def __init__(self, task, method, llm_model, batch_size=20, temperature=0.7):
        
        self.task = task
        self.method = method
        if self.task == "imdb":
            self.topic = "movie review"
        else:
            self.topic = "natural language inference pair"
        
        if llm_model == "gpt":
            self.model_name = "gpt"
            self.llm_model = "gpt-3.5-turbo-0125"
            self.client = OpenAI()
        else:
            self.model_name = "mistral"
            self.llm_model = "mistralai/Mistral-7B-Instruct-v0.2"
            self.tokenizer = AutoTokenizer.from_pretrained(self.llm_model, padding_side="left")
            self.llm_pipeline = transformers.pipeline(
                "text-generation",
                model=self.llm_model,
                torch_dtype=torch.float16,
                device_map="auto",
                tokenizer=self.tokenizer
            )
        self.batch_size = batch_size
        self.temperature = temperature
        self._create_prompt()
    def _delayed_completion(self, **kwargs):
        rate_limit_per_minute = 500
        delay = 60.0 / rate_limit_per_minute
        time.sleep(delay)
        return self.client.chat.completions.create(**kwargs)
    
    def _create_prompt(self):
        grammar_prompt = f"How grammatically correct is the {self.topic}? (on a scale of 1-5, with 1 being the lowest?)."
        cohesiveness_prompt= f"How well do the sentences in the {self.topic} fit together? (on a scale of 1-5, with 1 being the lowest?)."
        like_prompt = f"How enjoyable do you find the {self.topic}? (on a scale of 1-5, with 1 being the lowest?)."
        fluency_prompt = f"Question: How natural and fluent the {self.topic}? (on a scale of 1-5, with 1 being the lowest)."
        self.label_prompt = f"How relevant is the {self.topic} to the label?"
        self.start_prompt = f"""The goal of this task is to rate a {self.topic}. 
Note: Please take the time to fully read and understand the {self.topic}. We will reject submissions from workers that are clearly spamming the task.
        """
        self.start_fluency_prompt=f"You are given a list of {self.topic}s. Please read the {self.topic} and answer the question.\n"
        self.dict_questions = {"grammar":grammar_prompt,
                        "cohesiveness": cohesiveness_prompt,
                        "likableness": like_prompt,
                        "fluency": fluency_prompt}
    def evaluate_text(self, df):
        for k_question, question in self.dict_questions.items():
            self.file_prompt = open(f"raw_text/{self.task}_{k_question}_{self.method}_prompt_{self.model_name}_{self.temperature}_{date_string}.txt", 'w')
            self.file_answer = open(f"raw_text/{self.task}_{k_question}_{self.method}_answer_{self.model_name}_{self.temperature}_{date_string}.txt", 'w')
            if self.task == "imdb":
                df = self.evaluate_single_text(df,k_question,question)
            if self.task == "snli":
                df = self.evaluate_pair_text(df,k_question,question)
        df.to_csv(f"results/{self.task}/{self.method}/eval_{self.model_name}_{self.temperature}_{date_string}.csv", index = False) 
    def gpt_evaluate(self, list_prompts, file_answer, list_scores):
        for prompt in list_prompts:
            prompts = [
                {
                "role": "user",
                "content": prompt
            }
            ]
            count = 0
            while True : 
                if count >=5:
                    raw_text = None
                    break
                try:
                    result = self.client.chat.completions.create(model=self.llm_model,
                                                            messages=prompts,
                                                            max_tokens=1,
                                                            temperature = self.temperature)
                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 400:
                        raw_text = None
                        print("BadRequestError: ", e)
                        break
                    else:
                        print("An HTTP error occurred: ", e)
                except Exception as e:
                    raw_text = None
                    print("An unexpected error occurred: ", e)
                    break
                try:
                    raw_text = int(result.choices[0].message.content)
                    break  # If the conversion is successful, break the loop
                except ValueError:
                    count+=1
                    continue  # If the conversion fails, continue with the next iteration
            file_answer.write(f"{raw_text}\n")
            list_scores.append(raw_text)
        return list_scores
    def llm_hf_evaluate(self,list_prompts, file_answer, list_scores):
        sequences = self.llm_pipeline(
            list_prompts,
            do_sample=True,
            top_k=50,
            num_return_sequences=1,
            max_new_tokens=2,
            pad_token_id = self.tokenizer.eos_token_id,
            temperature = self.temperature
        )
        for s in sequences:
            file_answer.write(f"{s[0]['generated_text'][-1]}")
            try:
                list_scores.append(int(s[0]['generated_text'][-1]))
            except ValueError:
                print(f"{s[0]['generated_text']}")
                list_scores.append(-1)
        return list_scores
    def evaluate_pair_text(self,df, k_question, question):
        
        if self.method == "crest":
            df['gen_premise'] = df['gen_text'].apply(lambda x: x.split(".")[0])
            df['gen_hypothesis'] = df['gen_text'].apply(lambda x: x.split(".")[1])
            map_lists = {"gen":zip(df['gen_premise'], df['gen_hypothesis'])}
        else:
            map_lists = {"premise_gen":zip(df['gen_premise'], df['orig_hypothesis']),
                    "hypothesis_gen":zip(df['orig_premise'], df['gen_hypothesis'])}
        for name,pairs in map_lists.items():
            list_prompts  = []
            list_scores = []
            for i, (premise, hypothesis) in enumerate(pairs):
                if k_question == "fluency":
                    prompt =  self.start_fluency_prompt + f"{self.topic[0].capitalize() + self.topic[1:]} {i}:\n{premise} {hypothesis}\n" + question
                else:
                    prompt =  self.start_prompt + f"{self.topic[0].capitalize() + self.topic[1:]} {i}:\n{premise} {hypothesis}\n" + question 
                prompt += f"\nIgnore errors related to uppercase and lowercase letters. Please only return a score.\ The score is:"
                self.file_prompt.write(f"{prompt}\n")
                list_prompts.append(prompt)
            if self.model_name != "gpt":
                list_scores = self.llm_hf_evaluate(list_prompts,self.file_answer, list_scores)
            else:
                list_scores = self.gpt_evaluate(list_prompts, self.file_answer, list_scores)
            score_column = f"{k_question}_{name}_{self.model_name}_{self.temperature}"
            df[score_column] = list_scores
            temp_df = df[df[score_column] != -1]
            temp_df = temp_df.dropna(axis=0)
            mean_score = temp_df[score_column].mean()
            print(f"{score_column}: {mean_score}")
        return df
    def evaluate_single_text(self,df, k_question, question):
        
        texts = df['gen_text'].to_list()    
        list_scores = []
        for batch_number, batch_texts in enumerate(zip_longest(*[iter(texts)] * self.batch_size)):
            list_prompts  = []
            for i, text in enumerate(batch_texts):
                if text is None:
                    continue
                if k_question == "fluency":
                    prompt =  self.start_fluency_prompt + f"{self.topic[0].capitalize() + self.topic[1:]} {(batch_number) * self.batch_size + i}:\n{text}\n" + question + f"\nIgnore errors related to uppercase and lowercase letters. Please only return a score for the {self.topic}.\nThe score is:"
                else:
                    prompt =  self.start_prompt + f"{self.topic[0].capitalize() + self.topic[1:]} {(batch_number) * self.batch_size + i}:\n{text}\n" + question + f"\nIgnore errors related to uppercase and lowercase letters. Please only return a score for the {self.topic}.\nThe score is:"
                self.file_prompt.write(f"{prompt}\n")
                list_prompts.append(prompt)
            if self.model_name != "gpt":
                list_scores = self.llm_hf_evaluate(list_prompts,self.file_answer, list_scores)
            else:
                list_scores = self.gpt_evaluate(list_prompts, self.file_answer, list_scores)

        score_column = f"{k_question}_score_{self.model_name}_{self.temperature}"
        df[score_column] = list_scores
        temp_df = df[df[score_column] != -1]
        temp_df = temp_df.dropna(axis=0)
        mean_score = temp_df[score_column].mean()
        print(f"{score_column}: {mean_score}")
        return df
def get_args():
    # Create the argument parser
    parser = argparse.ArgumentParser(description="LLMs evaluation")

    # Add positional arguments
    parser.add_argument("-method", required=True, help="results of method for evaluation")
    parser.add_argument("-task", required=True, help="Name of the task. Currently, only IMDB and SNLI are supported.", choices=['imdb', 'snli'])

    # Add optional arguments
    parser.add_argument("-batch_size", type=int, default=100, help="Batch size for evaluation.")
    parser.add_argument("-temperature", type=float, default=0.2, help="Temperature for evaluation.")
    parser.add_argument("-llm_model", required=True, help="choose model used for evaluation", choices=['gpt', 'mistral'])

    # Parse the command line arguments
    args = parser.parse_args()
    return args
if __name__ == '__main__':
    args = get_args()
    llm_eval  = LLMEvaluator(args.task, args.method, args.llm_model, args.batch_size, args.temperature)
    print(f"{args.method}:")
    df = pd.read_csv(f"results/{args.task}/{args.method}/results.csv")
    df = df.dropna(axis=0)
    llm_eval.evaluate_text(df)

    
    
    