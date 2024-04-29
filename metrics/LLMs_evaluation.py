
import transformers
import torch
import json
import pandas as pd
from openai import OpenAI
import time
from itertools import zip_longest
from transformers import AutoTokenizer
from datetime import datetime
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
    def __init__(self, task, llm_model, list_methods, batch_size=20):
        
        self.task = task
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
            self.tokenizer = AutoTokenizer.from_pretrained(llm_model, padding_side="left")
            self.llm_pipeline = transformers.pipeline(
                "text-generation",
                model=llm_model,
                torch_dtype=torch.float16,
                device_map="auto",
                tokenizer=self.tokenizer
            )
        self.list_methods = list_methods
        self.batch_size = batch_size
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
        self.start_fluency_prompt=f"You are given a list of {self.topic}s. Please read the {self.topic} and answer the question.\n{self.topic[0].capitalize() + self.topic[1:]}:\n"
        self.dict_questions = {"grammar":grammar_prompt,
                        "cohesiveness": cohesiveness_prompt,
                        "likableness": like_prompt,
                        "fluency": fluency_prompt}
    def evaluate_pair_text(self):
        for method in self.list_methods:
            print(f"{method}:")

            df = pd.read_csv(f"results/{self.task}/{method}/results.csv")
            list_premise_gen = df['gen_premise'].to_list()
            list_hypothesis_gen = df['gen_hypothesis'].to_list()
            list_premise_orig = df['orig_premise'].to_list()
            list_hypothesis_orig = df['orig_hypothesis'].to_list()
            map_lists = {"premise_gen":zip(list_premise_gen, list_hypothesis_orig),
                       "hypothesis_gen":zip(list_premise_orig, list_hypothesis_gen)}
            for k_question in self.dict_questions:
                file_prompt = open(f"{self.task}_{k_question}_{method}_prompt_{self.model_name}_{date_string}.txt", 'a')
                file_answer = open(f"{self.task}_{k_question}_{method}_answer_{self.model_name}_{date_string}.txt", 'a')
                question = self.dict_questions[k_question]
                for name in map_lists.keys():
                    list_prompts  = []
                    list_scores = []
                    for i, (premise, hypothesis) in enumerate(map_lists[name]):
                        if k_question == "fluency":
                            prompt =  self.start_fluency_prompt + f"{self.topic[0].capitalize() + self.topic[1:]} {i}:\n{premise} {hypothesis}\n" + question
                        else:
                            prompt =  self.start_prompt + f"{self.topic[0].capitalize() + self.topic[1:]} {i}:\n{premise} {hypothesis}\n" + question 
                        prompt += f"\nIgnore errors related to uppercase and lowercase letters. Please only return a score.\nScore:"
                        file_prompt.write(f"{prompt}\n")
                        list_prompts.append(prompt)
                    if self.model_name != "gpt":
                        sequences = self.llm_pipeline(
                            list_prompts,
                            do_sample=True,
                            top_k=50,
                            num_return_sequences=1,
                            max_new_tokens=2,
                            pad_token_id = self.tokenizer.eos_token_id,
                            temperature = 0.7
                        )
                        for s in sequences:
                            file_answer.write(f"{s[0]['generated_text'].split('The score is:')[1]}")
                            try:
                                list_scores.append(int(s[0]['generated_text'][-1]))
                            except ValueError:
                                print(f"{s[0]['generated_text']}")
                                list_scores.append(-1)
                    else:
                        # stringifiedPromptsArray = json.dumps(list_prompts)
                        for prompt in list_prompts:
                            prompts = [
                                {
                                "role": "user",
                                "content": prompt
                            }
                            ]

                            print("ChatGPT: ")
                            result = self._delayed_completion(model=self.llm_model,
                                                                    messages=prompts,
                                                                    max_tokens=2)
                            raw_text = int(result.choices[0].message.content)
                            file_answer.write(f"{raw_text}\n")
                            list_scores.append(raw_text)
                    df[f"{k_question}_{name}_{self.model_name}"] = list_scores
            df.to_csv(f"results/{self.task}/{method}/results_llm_{date_string}.csv")
    def evaluate_long_text(self):
        for method in self.list_methods:
            
            print(f"{method}:")
            df = pd.read_csv(f"results/{self.task}/{method}/results.csv")[:5]
            
            texts = df['gen_text'].to_list()
            for k_question in self.dict_questions:
                # if method != "llama" or  k_question != "cohesiveness":
                #     continue
                list_scores = []
                file_prompt = open(f"{self.task}_{k_question}_{method}_prompt_{self.model_name}_{date_string}.txt", 'a')
                file_answer = open(f"{self.task}_{k_question}_{method}_answer_{self.model_name}_{date_string}.txt", 'a')
                question = self.dict_questions[k_question]
                
                
                for batch_number, batch_texts in enumerate(zip_longest(*[iter(texts)] * self.batch_size)):
                    text_list = ""
                    list_prompts  = []
                    for i, text in enumerate(batch_texts):
                        # if (batch_number) * self.batch_size + i <=399:
                        #     continue
                        if text is None:
                            continue
                        # text_list += f"{self.topic[0].capitalize() + self.topic[1:]} {(batch_number) * self.batch_size + i}:\n{text}\n"
                        if k_question == "fluency":
                            prompt =  self.start_fluency_prompt + f"{self.topic[0].capitalize() + self.topic[1:]} {(batch_number) * self.batch_size + i}:\n{text}\n" + question + f"Ignore errors related to uppercase and lowercase letters. Please only return a score for the {self.topic}. The score is:"
                        else:
                            prompt =  self.start_prompt + f"{self.topic[0].capitalize() + self.topic[1:]} {(batch_number) * self.batch_size + i}:\n{text}\n" + question + f"Ignore errors related to uppercase and lowercase letters. Please only return a score for the {self.topic}. The score is:"
                        file_prompt.write(f"{prompt}\n")
                        list_prompts.append(prompt)
                    # if text_list == "":
                    #     continue
                    # Example usage:
                    # input_text = "This is an example sentence."
                    # t)
                    # prompt = start_prompt + text_list + question
                        
                    
                    device = "cuda" # the device to load the model onto
                    
                    # max_repetitions = 6
                    # text_list = cut_continuous_repetitions(text_list, max_repetitions)
                    if self.model_name != "gpt":
                        sequences = self.llm_pipeline(
                            list_prompts,
                            do_sample=True,
                            top_k=50,
                            num_return_sequences=1,
                            max_new_tokens=2,
                            pad_token_id = self.tokenizer.eos_token_id,
                            temperature = 0.7
                        )
                        for s in sequences:
                            file_answer.write(f"{s[0]['generated_text'].split('The score is:')[1]}")
                            try:
                                list_scores.append(int(s[0]['generated_text'][-1]))
                            except ValueError:
                                list_scores.append(-1)
                    else:
                        for prompt in list_prompts:
                        # print(promptsArray)
                            if k_question == "fluency":
                                prompt =  self.start_fluency_prompt + text_list + question 
                            else:
                                prompt =  self.start_prompt + text_list + question
                            prompt += f"\nIgnore errors related to uppercase and lowercase letters. Please only return a score.\nScore:"
                        #     {
                        #     "role": "user",
                        #     "content": stringifiedPromptsArray
                        # }
                        # ]
                            prompts =  [
                                {
                                "role": "user",
                                "content": prompt
                            }
                            ]

                            results = self._delayed_completion(model=self.llm_model,
                                                                    messages=prompts,
                                                                    max_tokens=2,
                                                                    temperature=0.7)
                            raw_text = int(results.choices[0].message.content)
                            # batchCompletion = json.loads(stringifiedBatchCompletion.choices[0].message.content)
                            
                            file_answer.write(f"{raw_text}\n")

                            # Split the string into lines
                            # reviews_list = raw_text.split('\n')

                            # Extract the score from each line and store in a list
                            # scores_list = [int(review.split(': ')[1]) for review in reviews_list if review]
                            # scores_list = [int(review) for review in reviews_list if review]

                            # Add to final list
                            list_scores.append(raw_text )

                    # print(batchCompletion)
                    # completion = delayed_completion(
                    #     delay_in_seconds=delay,
                    #     model="gpt-3.5-turbo-1106",
                    #     messages=[
                    #         {"role": "user", "content": prompts}
                    #     ],
                    #     temperature = 1
                    # )
                    # file_answer.write(f"{completion.choices[0].message.content}\n")
                    # print(completion.choices[0].message.content)
                score_column = f"{k_question}_score_{self.model_name}"
                df[score_column] = list_scores
                temp_df = df[df[score_column] != -1]
                mean_score = temp_df[score_column].mean()
                print(f"{score_column}: {mean_score}")
            df.to_csv(f"results/{self.task}/{method}/results_llm_{date_string}.csv", index = False) 
if __name__ == '__main__':
    #create prompt for the dataset
    

    
    # Append text to the file
    # list_methods = ["crest"]
    list_methods = ["llama","human_crowd","mice"]
    # llm_eval  = LLMEvaluator("snli", "mistralai/Mistral-7B-Instruct-v0.2", list_methods, 100)
    llm_eval  = LLMEvaluator("imdb", "gpt", list_methods, 20)
    llm_eval.evaluate_long_text()
    # llm_eval.evaluate_pair_text()
    # evaluate_long_text(list_methods, batch_size=20, model="gpt")
    
    
    