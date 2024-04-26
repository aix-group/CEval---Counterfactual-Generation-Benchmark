import pandas as pd
from transformers import AutoTokenizer
import transformers
import torch
import time
import re
def create_prompt(example):
    contrast_map = {"Positive": "Negative", "Negative": "Positive"}
    example_map = {
        "Negative": "Long, boring, blasphemous. Never have I been so glad to see ending credits roll.",
        "Positive": "Long, fascinating, soulful. Never have I been so sad to see ending credits roll."
                   }
    orig_sent = example['Orig_Sent']
    template = """Request: """
    template = f"""Request: Given a piece of text with the original sentiment in the form of "Sentiment: Text". Change the text with minimal edits to get the target sentiment from the original sentiment. Do not make any unneccesary changes. For example:
[Start Original Text]
{orig_sent}: {example_map[orig_sent]}
[End Original Text]
Target: {contrast_map[orig_sent]}
[Start Edited Text]
{contrast_map[orig_sent]}: {example_map[contrast_map[orig_sent]]}
[End Edited Text]
######End Example#######
Now, similar to the example, given a piece of text below, please change the text with minimal edits to get the {contrast_map[orig_sent]} sentiment from the {orig_sent} sentiment. Do not make any unneccesary changes.
[Start Original Text]
{orig_sent}: {example['Orig_Inp']}
[End Original Text]
Target: {contrast_map[orig_sent]}
[Start Edited Text]
{contrast_map[orig_sent]}:"""
    return template
if __name__ == '__main__':

    llm_model = "meta-llama/Llama-2-7b-chat-hf"
    # llm_model = "tiiuae/falcon-40b-instruct"
    # sbert_model_name = "all-mpnet-base-v2"
    # dev_pairs = pd.read_csv("datasets/imdb/paired/dev_pairs.csv")
    # test_pairs = pd.read_csv("datasets/imdb/paired/test_paired.tsv", delimiter="\t")
    # train_pairs = pd.read_csv("datasets/imdb/paired/train_paired.tsv", delimiter="\t")
    # df_merge = pd.concat([train_pairs, test_pairs])
    # load dataset
    df = pd.read_csv("datasets/imdb/merge.csv", delimiter="\t")
    df = df.iloc[100:]
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

    list_contrast_texts = []
    list_prompts = []
    # df_test = test_pairs
    #pick an instance
    # Specify the chunk size
    chunk_size = 100
    start_edited_pattern = r'\[Start Edited Text\](.*?)(?:\[End Edited Text\]|$)'
    # Calculate the number of chunks
    num_chunks = len(df) // chunk_size + 1
    start_time = time.time()
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size
        chunk = df.iloc[start_idx:end_idx]
        list_chunk_prompts = []
        for index, example in chunk.iterrows():
            prompt = create_prompt(example)
            list_prompts.append(prompt)
            list_chunk_prompts.append(prompt)
        
        sequences = llm_pipeline(
            list_chunk_prompts,
            do_sample=True,
            top_k=50,
            num_return_sequences=1,
            max_new_tokens=768,
            eos_token_id=tokenizer.eos_token_id
        )
        # for seq in sequences:
        #     # print(seq[0]['generated_text']) 
        #     text = seq[0]['generated_text'].split("\n\n\n")[0]
        #     with open(r'raw_text_imdb_llm_2.txt', 'a') as fp:
        #         fp.write("[start]%s\n" % text)
        #     text_split = text.split("(Edit Text)\n")
        #     if len(text_split) == 3:
        #         contrast_text = text_split[2].split("\n\n")[0][14:]
        #     else:
        #         #raise error
        #         print("ERROR SUSPECT")
        #         print(text)
        #         contrast_text = text_split[2].split("\n\n")[0][14:]
        #     list_contrast_texts.append(contrast_text)
        for seq in sequences:
            # print(seq[0]['generated_text']) 
            text = seq[0]['generated_text']
            with open(r'raw_text_imdb_llm_2.txt', 'a') as fp:
                fp.write("[start]%s\n" % text)

        for seq in sequences:
            text = seq[0]['generated_text'].split("######End Example#######")[1]
            edited_match = re.search(start_edited_pattern, text, re.DOTALL)
            if edited_match:
                edited_text = edited_match.group(1).strip()
                target_match = re.search(r'(?:Positive|Negative):(.*?)(?:\\n|$)', edited_text, re.DOTALL)
                if target_match:
                    contrast_text = target_match.group(1).strip()
                else:
                    print(edited_text)
                    contrast_text = None
            else:
                print(text)
                contrast_text = None

            list_contrast_texts.append(contrast_text)


    end_time = time.time()
    df['llama_text'] = list_contrast_texts
    df.to_csv("imdb_llm_2.csv")
    duration = end_time - start_time
    print(duration)
    with open(r'duration_llama_2.txt', 'w') as fp:
        fp.write("Duration: %s" % str(duration))
    