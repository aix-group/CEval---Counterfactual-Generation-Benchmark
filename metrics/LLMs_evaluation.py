
import transformers
import torch
import json
import pandas as pd
from openai import OpenAI
import time
from itertools import zip_longest
from transformers import AutoTokenizer

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
        self.llm_model = llm_model
        if llm_model == "gpt":
            self.client = OpenAI()
        else:
            
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
        rate_limit_per_minute = 3
        delay = 60.0 / rate_limit_per_minute
        time.sleep(delay)
        return self.client.chat.completions.create(**kwargs)
    def _create_prompt(self):
        grammar_prompt = f"How grammatically correct is each {self.topic}? (on a scale of 1-5, with 1 being the lowest?)."
        cohesiveness_prompt= f"How well do the sentences in each {self.topic} fit together? (on a scale of 1-5, with 1 being the lowest?)."
        like_prompt = f"How enjoyable do you find each {self.topic}? (on a scale of 1-5, with 1 being the lowest?)."
        fluency_prompt = f"Question: How natural and fluent each {self.topic}? (on a scale of 1-5, with 1 being the lowest)."
        self.label_prompt = f"How relevant is each {self.topic} to the label?"
        self.start_prompt = f"""Please rate each {self.topic}
        The goal of this task is to rate each {self.topic}. 
        Note: Please take the time to fully read and understand the {self.topic}. We will reject submissions from workers that are clearly spamming the task.
        """
        self.start_fluency_prompt=f"You are given a list of {self.topic}s. Please read the {self.topic} and answer the question.\n{self.topic[0].capitalize() + self.topic[1:]}:\n"
        self.dict_questions = {"grammar":grammar_prompt,
                        "cohesiveness": cohesiveness_prompt,
                        "likableness": like_prompt,
                        "fluency": fluency_prompt}
    def evaluate_pair_text(self, list_methods):
        for method in list_methods:
            print(f"{method}:")

            df = pd.read_csv(f"results/{self.task}/{method}/results.csv")
            list_premise_gen = df['gen_premise'].to_list()
            list_hypothesis_gen = df['gen_hypothesis'].to_list()
            list_premise_orig = df['orig_premise'].to_list()
            list_hypothesis_orig = df['orig_hypothesis'].to_list()
            for k_question in self.dict_questions:
                list_scores = []
                file_prompt = open(f"{self.task}_{k_question}_{method}_prompt.txt", 'a')
                file_answer = open(f"{self.task}_{k_question}_{method}_answer.txt", 'a')
                question = self.dict_questions[k_question]
                list_prompts  = []
                for i, (premise, hypothesis) in enumerate(zip(list_premise_gen, list_hypothesis_orig)):
                    if k_question == "fluency":
                        prompt =  self.start_fluency_prompt + f"{self.topic[0].capitalize() + self.topic[1:]} {i}:\n{premise} {hypothesis}\n" + question + f". Ignore errors related to uppercase and lowercase letters. Please only return a score for each {self.topic}."
                    else:
                        prompt =  self.start_prompt + f"{self.topic[0].capitalize() + self.topic[1:]} {i}:\n{premise} {hypothesis}\n" + question + f". Ignore errors related to uppercase and lowercase letters. Please only return a score for each {self.topic}. "
                    file_prompt.write(f"{prompt}\n")
                    list_prompts.append(prompt)
                    
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
                df[f"{k_question}_premise_gen"] = list_scores
                list_prompts  = []
                list_scores = []
                for i, (premise, hypothesis) in enumerate(zip(list_premise_orig, list_hypothesis_gen)):
                    if k_question == "fluency":
                        prompt =  self.start_fluency_prompt + f"{self.topic[0].capitalize() + self.topic[1:]} {i}:\n{premise} {hypothesis}\n" + question + f". Ignore errors related to uppercase and lowercase letters. Please only return a score for each {self.topic}."
                    else:
                        prompt =  self.start_prompt + f"{self.topic[0].capitalize() + self.topic[1:]} {i}:\n{premise} {hypothesis}\n" + question + f". Ignore errors related to uppercase and lowercase letters. Please only return a score for each {self.topic}."
                    file_prompt.write(f"{prompt}\n")
                    list_prompts.append(prompt)
                if self.llm_model != "gpt":
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
                    stringifiedPromptsArray = json.dumps(list_prompts)

                    # print(promptsArray)

                    prompts = [
                        {
                        "role": "user",
                        "content": stringifiedPromptsArray
                    }
                    ]

                    batchInstruction = {
                        "role":
                        "system",
                        "content":
                        "Complete every element of the array. Reply with an array of all completions."
                    }

                    prompts.append(batchInstruction)
                    print("ChatGPT: ")
                    stringifiedBatchCompletion = self._delayed_completion(model="gpt-3.5-turbo-1106",
                                                            messages=prompts,
                                                            max_tokens=1000)
                    batchCompletion = json.loads(stringifiedBatchCompletion.choices[0].message.content)
                    list_scores.extend(batchCompletion)
                df[f"{k_question}_hypothesis_gen"] = list_scores

            df.to_csv(f"results/{self.task}/{method}/results_llm.csv")
    def evaluate_long_text(self):
        for method in self.list_methods:
            
            print(f"{method}:")
            df = pd.read_csv(f"results/{self.task}/{method}/results.csv")
            
            texts = df['gen_text'].to_list()
            for k_question in self.dict_questions:
                # if method != "llama" or  k_question != "cohesiveness":
                #     continue
                list_scores = []
                file_prompt = open(f"{self.task}_{k_question}_{method}_prompt.txt", 'a')
                file_answer = open(f"{self.task}_{k_question}_{method}_answer_10.txt", 'a')
                question = self.dict_questions[k_question]
                
                
                for batch_number, batch_texts in enumerate(zip_longest(*[iter(texts)] * self.batch_size)):
                    text_list = ""
                    list_prompts  = []
                    for i, text in enumerate(batch_texts):
                        # if (batch_number) * self.batch_size + i <=399:
                        #     continue
                        if text is None:
                            continue
                        text_list += f"{self.topic[0].capitalize() + self.topic[1:]} {(batch_number) * self.batch_size + i}:\n{text}\n"
                        if k_question == "fluency":
                            prompt =  self.start_fluency_prompt + f"{self.topic[0].capitalize() + self.topic[1:]} {(batch_number) * self.batch_size + i}:\n{text}\n" + question + f"Ignore errors related to uppercase and lowercase letters. Please only return a score for the {self.topic}. The score is:"
                        else:
                            prompt =  self.start_prompt + f"{self.topic[0].capitalize() + self.topic[1:]} {(batch_number) * self.batch_size + i}:\n{text}\n" + question + f"Ignore errors related to uppercase and lowercase letters. Please only return a score for the {self.topic}. The score is:"
                        file_prompt.write(f"{prompt}\n")
                        list_prompts.append(prompt)
                    if text_list == "":
                        continue
                    # Example usage:
                    # input_text = "This is an example sentence."
                    # t)
                    # prompt = start_prompt + text_list + question
                        
                    
                    device = "cuda" # the device to load the model onto
                    
                    max_repetitions = 6
                    text_list = cut_continuous_repetitions(text_list, max_repetitions)
                    if self.llm_model != "gpt":
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
                    # promptsArray = ["Hello world, from", "How are you B", "I am fine. W", "The  fifth planet from the Sun is "]
                    else:
                        stringifiedPromptsArray = json.dumps(list_prompts)

                        # print(promptsArray)
                        if k_question == "fluency":
                            prompt =  self.start_fluency_prompt + text_list + question + f". Ignore errors related to uppercase and lowercase letters. Please only return the score numer for each {self.topic} in the format '{self.topic}: score."
                        else:
                            prompt =  self.start_prompt + text_list + question + f". Ignore errors related to uppercase and lowercase letters. Please only return the score numer for each {self.topic} in the format '{self.topic}: score."
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
                        # batchInstruction = {
                        #     "role":
                        #     "system",
                        #     "content":
                        #     "Complete every element of the array. Reply with an array of all completions."
                        # }

                        # prompts.append(batchInstruction)
                        results = self._delayed_completion(model="gpt-3.5-turbo-1106",
                                                                messages=prompts,
                                                                max_tokens=200,
                                                                temperature=1.0)
                        raw_text = results.choices[0].message.content
                        # batchCompletion = json.loads(stringifiedBatchCompletion.choices[0].message.content)
                        
                        file_answer.write(f"{raw_text}\n")

                        # Split the string into lines
                        reviews_list = raw_text.split('\n')

                        # Extract the score from each line and store in a list
                        scores_list = [int(review.split(': ')[1]) for review in reviews_list if review]
                        # scores_list = [int(review) for review in reviews_list if review]

                        # Add to final list
                        list_scores.extend(scores_list)

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
                    
                df[f"{k_question}_score"] = list_scores
                temp_df = df[df[f"{k_question}_score"] != -1]
                mean_score = temp_df[f"{k_question}_score"].mean()
                print(f"{k_question}_score: {mean_score}")
            df.to_csv(f"results/{self.task}/{method}/results_llm.csv", index = False) 
# start_prompt = f"""Please rate the text of the {topic}
# The goal of this task is to rate the {topic}. 
# Note: Please take the time to fully read and understand the {topic}. We will reject submissions from workers that are clearly spamming the task.
# """
# task = "imdb"
# client = OpenAI()
# # model = "meta-llama/Llama-2-7b-chat-hf"
# topic = "movie review"

# # topic = "natural language inference pair"
# grammar_prompt = f"How grammatically correct is the text of the {topic}? (on a scale of 1-5, with 1 being the lowest?)."
# cohesiveness_prompt= f"How well do the sentences in the {topic} fit together? (on a scale of 1-5, with 1 being the lowest?)."
# like_prompt = f"How enjoyable do you find the {topic}? (on a scale of 1-5, with 1 being the lowest?)."
# label_prompt = f"How relevant is the {topic} to the label?"
# start_fluency_prompt=f"You are given a list of {topic}s. Please read the {topic} and answer the question.\n{topic[0].capitalize() + topic[1:]}:\n"
# fluency_prompt = f"Question: How natural and fluent is the text of the {topic}? (on a scale of 1-5, with 1 being the lowest)."
# dict_questions = {"grammar":grammar_prompt,
#                   "cohesiveness": cohesiveness_prompt,
#                   "likableness": like_prompt,
#                   "fluency": fluency_prompt}
# start_prompt = f"""Please rate the text of the {topic}
# The goal of this task is to rate the {topic}. 
# Note: Please take the time to fully read and understand the {topic}. We will reject submissions from workers that are clearly spamming the task.
# """
# task = "imdb"
# llm_model = "mistralai/Mistral-7B-Instruct-v0.2"
# tokenizer = AutoTokenizer.from_pretrained(llm_model, padding_side = "left")
# llm_pipeline = transformers.pipeline(
#     "text-generation",
#     model=llm_model,
#     torch_dtype=torch.float16,
#     device_map="auto",
#     tokenizer = tokenizer
# )
# tokenizer = None
# human = pd.read_csv("results/imdb/human_gold/results.csv")
# mice = pd.read_csv("results/imdb/mice/results.csv")
# llama = pd.read_csv("results/imdb/llama/results.csv")
# gbda = pd.read_csv("results/imdb/gbda/results.csv")
# cfgan = pd.read_csv("results/imdb/cf_gan/results.csv")

# prompt = ""
# for i, text in enumerate(human['gen_text'].to_list()[:50]):
#     prompt = prompt + f"Movie review {i+1}: " + text + "\n\n"

# tokenizer = AutoTokenizer.from_pretrained(model)
# pipeline = transformers.pipeline(
#     "text-generation",
#     model=model,
#     torch_dtype=torch.float16,
#     device_map="auto",
# )
# def create_prompt(dataset='imdb'):
#     dataset_topic_mapping = {"imdb":"movie review",
#                             "snli": "natural language inference"}
    
#     topc
#     imdb_prompt = 
# def process_function(text):
#     text = text.replace("<s> ","")
#     text = text.replace("<s>","")
#     text = text.replace("</s>","")
#     return text
# gbda[['orig_text', 'gen_text']] = gbda[['orig_text', 'gen_text']].applymap(process_function)
# for criterion, predefined_prompt in zip(["grammar", "cohesive", "like", "fluency"],[grammar_prompt, cohesiveness_prompt, like_prompt, fluency_prompt]):
#     for df in [human,mice,llama,gbda]:
#         list_text = df['gen_text'].to_list()
#         list_prompts = []
#         for text in list_text:
#             # text = str(human.iloc[1]['gen_text'])
#             if criterion == "fluency":
#                 prompt = start_fluency_prompt + text + f"\n(End of {topic})\n" + predefined_prompt
#             else:
#                 prompt = start_prompt + text + f"\n(End of {topic})\n" + predefined_prompt
#             list_prompts.append(prompt)
# list_prompts = list_prompts[:2]
# prompt = """'You are given movie reviews. Please read the movie reviews and answer the question.
# Movie review 1:
# i had quite severe hopes for the film, even though it got a bad review in the press. i was extremely supportive, and sat through the entire film. i felt quite ready by the end. although i am not in the least prudently particularly sensitive to colour good cinema - - i carol youghly noted some woody allen's'everything i ever wanted to do about sex,.... and michael hanneke's'funny games'- - i found the audiences'argument with this 15 - year @ old only to steal man's milk really sickening. and when the film factored in an " orgy " where the protagonist drinks from his mother's milk, as well as that of the woman he has been booing in for the whole film, i almost heaped with pleasure for the all perversion and diablo marip that it is. don't get me wrong, i see the vast majority of these cinema, as well as independently made films, so this flick does have touched me enormously. emphasis this film at all costs, it should be transferred to the annals of history as a staple in american cinema.

# Movie Review 2:
# woo disorders : * * * * * saturday night * * * * friday night * * * wednesday morning * * friday night * saturday morning a particularly notable movie, getting by on his ( now deceased ) heels rather than any actual acting talent, richard gere has always occupied a rather comfortable position in the american drama scene, being a regular target in leading man roles that still have a consistent presence here. but, he seems to have evolved more into these sort of threat to supernatural / bad acting. and as such hubbard seems to be more interested in his acting now. he has to work on some pretty matter here as paid, recent case worker max babbage, one such client assigned to a few hundred sex offenders in one year of the us, who along with his new wife cindy allthough ( claire danes ) must take to his latest case, delving into the death of a young woman while trying to forgive him for a mess he planned on months ago. this is a certain fall into the darker side of humanity, souring on film definitely not for the squeamish or those looking for light yourself. and as such it's a bad holiday, intimidating presence, unflinching and not amused by it's direct to dvd passage 

# Question: How natural and fluent is the text of the movie reviews? (on a scale of 1-5, with 1 being the lowest), give me only the score for each text'"""



# print(completion.choices[0].message)
#         sequences = pipeline(
#             list_prompts,
#             do_sample=True,
#             top_k=50,
#             num_return_sequences=1,
#             eos_token_id=tokenizer.eos_token_id,
#             max_length=1024,
#         )
#         df[criterion] = sequences
# human.to_csv("human_eval.csv")
# mice.to_csv("mice_eval.csv")
# llama.to_csv("llama_eval.csv")
# gbda.to_csv("gbda_eval.csv")


# for p in [grammar_prompt, cohesiveness_prompt, like_prompt]:
#     prompt = start_prompt + text + p
#     list_prompts.append(prompt)
# sequences = pipeline(
#     list_prompts,
#     do_sample=True,
#     top_k=50,
#     num_return_sequences=1,
#     eos_token_id=tokenizer.eos_token_id,
#     max_length=1024,
# )
# from transformers import GPT2Tokenizer

# def count_tokens(text, model_name="gpt2"):
#     tokenizer = GPT2Tokenizer.from_pretrained(model_name)
#     tokens = tokenizer.encode(text, add_special_tokens=True)
#     return len(tokens)
# def delayed_completion(delay_in_seconds: float = 1, **kwargs):
#     """Delay a completion by a specified amount of time."""

#     # Sleep for the delay
#     time.sleep(delay_in_seconds)

#     # Call the Completion API and return the result
#     return client.chat.completions.create(**kwargs)


# # Calculate the delay based on your rate limit
# def evaluate_pair_text(list_methods):
#     for method in list_methods:
#         df = pd.read_csv(f"results/{task}/{method}/results.csv")
#         list_premise_gen = df['gen_premise'].to_list()
#         list_hypothesis_gen = df['gen_hypothesis'].to_list()
#         list_premise_orig = df['orig_premise'].to_list()
#         list_hypothesis_orig = df['orig_hypothesis'].to_list()
#         for k_question in dict_questions:
#             list_scores = []
#             file_prompt = open(f"{task}_{k_question}_{method}_prompt.txt", 'a')
#             file_answer = open(f"{task}_{k_question}_{method}_answer.txt", 'a')
#             question = dict_questions[k_question]
#             list_prompts  = []
#             for i, (premise, hypothesis) in enumerate(zip(list_premise_gen, list_hypothesis_orig)):
#                 if k_question == "fluency":
#                     prompt =  start_fluency_prompt + f"{topic[0].capitalize() + topic[1:]} {i}:\n{premise} {hypothesis}\n" + question + f". Ignore errors related to uppercase and lowercase letters. Please only return a number. The score is:"
#                 else:
#                     prompt =  start_prompt + f"{topic[0].capitalize() + topic[1:]} {i}:\n{premise} {hypothesis}\n" + question + f". Ignore errors related to uppercase and lowercase letters. Please only return a number. The score is:"
#                 file_prompt.write(f"{prompt}\n")
#                 list_prompts.append(prompt)
                
#             sequences = llm_pipeline(
#                 list_prompts,
#                 do_sample=True,
#                 top_k=50,
#                 num_return_sequences=1,
#                 max_new_tokens=2,
#                 pad_token_id = tokenizer.eos_token_id,
#                 temperature = 0.7
#             )
#             for s in sequences:
#                 file_answer.write(f"{s[0]['generated_text'].split('The score is:')[1]}")
#                 try:
#                     list_scores.append(int(s[0]['generated_text'][-1]))
#                 except ValueError:
#                     print(f"{s[0]['generated_text']}")
#                     list_scores.append(-1)
#             df[f"{k_question}_premise_gen"] = list_scores
#             list_prompts  = []
#             list_scores = []
#             for i, (premise, hypothesis) in enumerate(zip(list_premise_orig, list_hypothesis_gen)):
#                 if k_question == "fluency":
#                     prompt =  start_fluency_prompt + f"{topic[0].capitalize() + topic[1:]} {i}:\n{premise} {hypothesis}\n" + question + f". Ignore errors related to uppercase and lowercase letters. Please only return a number. The score is:"
#                 else:
#                     prompt =  start_prompt + f"{topic[0].capitalize() + topic[1:]} {i}:\n{premise} {hypothesis}\n" + question + f". Ignore errors related to uppercase and lowercase letters. Please only return a number. The score is:"
#                 file_prompt.write(f"{prompt}\n")
#                 list_prompts.append(prompt)
            
#             sequences = llm_pipeline(
#                 list_prompts,
#                 do_sample=True,
#                 top_k=50,
#                 num_return_sequences=1,
#                 max_new_tokens=2,
#                 pad_token_id = tokenizer.eos_token_id,
#                 temperature = 0.7
#             )
#             for s in sequences:
#                 file_answer.write(f"{s[0]['generated_text'].split('The score is:')[1]}")
#                 try:
#                     list_scores.append(int(s[0]['generated_text'][-1]))
#                 except ValueError:
#                     list_scores.append(-1)
#             df[f"{k_question}_hypothesis_gen"] = list_scores

#         df.to_csv(f"results/{task}/{method}/results_llm.csv")            
                
#         # token_count = count_tokens(prompt)
#         # print(f"Token count: {token_count}")
#     # df['gen_text']
#     # grammar_prompt

# def evaluate_long_text(list_methods,batch_size = 20, model="gpt"):
#     for method in list_methods:
#         df = pd.read_csv(f"results/{task}/{method}/results.csv")
        
#         texts = df['gen_text'].to_list()
#         for k_question in dict_questions:
#             list_scores = []
#             file_prompt = open(f"{task}_{k_question}_{method}_prompt_{model}.txt", 'a')
#             file_answer = open(f"{task}_{k_question}_{method}_answer_{model}.txt", 'a')
#             question = dict_questions[k_question]
            
            
#             for batch_number, batch_texts in enumerate(zip_longest(*[iter(texts)] * batch_size)):
#                 text_list = ""
#                 list_prompts  = []
#                 for i, text in enumerate(batch_texts):
#                     if text is None:
#                         continue
#                     text_list += f"{topic[0].capitalize() + topic[1:]} {(batch_number - 1) * batch_size + i}:\n{text}\n"
#                     if k_question == "fluency":
#                         prompt =  start_fluency_prompt + f"{topic[0].capitalize() + topic[1:]} {(batch_number - 1) * batch_size + i}:\n{text}\n" + question + f". Ignore errors related to uppercase and lowercase letters. Please only return a number. The score is:"
#                     else:
#                         prompt =  start_prompt + f"{topic[0].capitalize() + topic[1:]} {(batch_number - 1) * batch_size + i}:\n{text}\n" + question + f". Ignore errors related to uppercase and lowercase letters. Please only return a number. The score is:"
#                     # file_prompt.write(f"{prompt}\n")
#                     list_prompts.append(prompt)
#                 # Example usage:
#                 # input_text = "This is an example sentence."
#                 # t)
#                 # prompt = start_prompt + text_list + question
                    
                
#                 device = "cuda" # the device to load the model onto

                
#                 if model != "gpt":
#                     sequences = llm_pipeline(
#                         list_prompts,
#                         do_sample=True,
#                         top_k=50,
#                         num_return_sequences=1,
#                         max_new_tokens=2,
#                         pad_token_id = tokenizer.eos_token_id,
#                         temperature = 0.7
#                     )
#                     for s in sequences:
#                         file_answer.write(f"{s[0]['generated_text'].split('The score is:')[1]}")
#                     try:
#                         list_scores.append(int(s[0]['generated_text'][-1]))
#                     except ValueError:
#                         list_scores.append(-1)
#                 # promptsArray = ["Hello world, from", "How are you B", "I am fine. W", "The  fifth planet from the Sun is "]
#                 else:
#                     stringifiedPromptsArray = json.dumps(list_prompts)

#                     # print(promptsArray)

#                     prompts = [
#                         {
#                         "role": "user",
#                         "content": stringifiedPromptsArray
#                     }
#                     ]

#                     batchInstruction = {
#                         "role":
#                         "system",
#                         "content":
#                         "Complete every element of the array. Reply with an array of all completions."
#                     }

#                     prompts.append(batchInstruction)
#                     print("ChatGPT: ")
#                     stringifiedBatchCompletion = delayed_completion(model="gpt-3.5-turbo-1106",
#                                                             messages=prompts,
#                                                             max_tokens=100)
#                     batchCompletion = json.loads(stringifiedBatchCompletion.choices[0].message.content)
#                     list_scores.extend(batchCompletion)
#                 # print(batchCompletion)
#                 # completion = delayed_completion(
#                 #     delay_in_seconds=delay,
#                 #     model="gpt-3.5-turbo-1106",
#                 #     messages=[
#                 #         {"role": "user", "content": prompts}
#                 #     ],
#                 #     temperature = 1
#                 # )
#                 # file_answer.write(f"{completion.choices[0].message.content}\n")
#                 # print(completion.choices[0].message.content)
                
#             df[f"{k_question}_score"] = list_scores

#         df.to_csv(f"results/imdb/{method}/results_llm.csv") 
if __name__ == '__main__':
    #create prompt for the dataset
    

    
    # Append text to the file
    list_methods = ["crest"]
    # list_methods = ["llama","human_crowd","mice"]
    llm_eval  = LLMEvaluator("snli", "mistralai/Mistral-7B-Instruct-v0.2", list_methods, 100)
    llm_eval.evaluate_long_text()
    # llm_eval.evaluate_pair_text()
    # evaluate_long_text(list_methods, batch_size=20, model="gpt")
    
    
    