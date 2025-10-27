from transformers import AutoTokenizer, set_seed
from prompts import PROMPT
from datasets import Dataset
import random
import json
import sys
from typing import List
import numpy as np

set_seed(42)
MAX_LENGTH=512*4

def build_messages(dataset: Dataset) -> Dataset:
    
    def preprocess_function(example):
        return {
            "messages": [
                {"role": "system", "content": "You are a seasoned Wikipedia fact-checker."},
                {"role": "user", "content": PROMPT.replace("{{claim}}", example["claim"])},
                {"role": "assistant", "content": json.dumps({"label": int(example["label"])})}
            ]
        }    
    data = list(dataset) 
    random.shuffle(data)

    dataset = dataset.map(preprocess_function, remove_columns=["claim", "label"])
    return dataset

def get_generation_tag(tokenizer):
    messages = ([{"role": "user","content":"test"}])
    no_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    with_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    if with_prompt.startswith(no_prompt):
        assistant_tag = with_prompt[len(no_prompt):]
    else:
        raise ValueError("Cannot identify the generation token.")
    
    assistant_tag_ids = tokenizer(assistant_tag, add_special_tokens=False)["input_ids"]

    return assistant_tag, assistant_tag_ids

def check_labels(tokenizer, tokenised_item):
    token_ids = np.array(tokenised_item['input_ids'])
    labels = np.array(tokenised_item['labels'])
    mask = labels != -100

    print(tokenizer.decode(token_ids[mask]))

def tokenize(example: dict, tokenizer, assistant_tag_ids: List[int]):

    def find_sublist_reverse(sub, lst):
        for i in range(len(lst) - len(sub), -1, -1):  # start from the back
            if lst[i:i+len(sub)] == sub:
                return i
        return -1

    text = tokenizer.apply_chat_template(
                                            example["messages"],
                                            tokenize=False,
                                            add_generation_prompt=False,
                                            enable_thinking=False
                                        )
    print(text)
    # text tokens
    text_tok = tokenizer(text, truncation=True, max_length=MAX_LENGTH)
    text_input_ids = text_tok['input_ids']

    # assistant tag tokens
    assistant_tag_index = find_sublist_reverse(assistant_tag_ids, text_input_ids)
    assert assistant_tag_index != -1, f"Could not find the assistant tag."
    assert (assistant_tag_index / len(text_input_ids)) > .8, f"Assistant tag not in the final 20% of the text."

    # generate completion mask
    # no need to shift lables, done by the model internally:
    # https://discuss.huggingface.co/t/where-does-the-transformers-do-the-target-text-shifting-in-causal-lm/32408
    labels = [-100]*len(text_input_ids)
    start = assistant_tag_index + len(assistant_tag_ids)
    labels[start:] = text_input_ids[start:]

    assert len(labels) == len(text_input_ids), "Labels length is incorrect."

    text_tok['labels'] =  labels
    return text_tok


def main():
    MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
    claims = [{'claim': "Test 1 claim", 'label': 1},
              {'claim': "Test 2 claim", 'label': 0}]

    hf_dataset = Dataset.from_list(claims)
    messages_ds = build_messages(hf_dataset)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID,             
                                              trust_remote_code=True)
    tokenizer.truncation_side = "left"

    assistant_tag, assistant_tag_ids = get_generation_tag(tokenizer)
    print(assistant_tag, assistant_tag_ids)

    tokenized_ds = messages_ds.map(tokenize,
                    fn_kwargs={"tokenizer": tokenizer,
                               "assistant_tag_ids": assistant_tag_ids}
                    )
    print(tokenized_ds['messages'][0])
    print(tokenized_ds[0])

    check_labels(tokenizer, tokenized_ds[0])

if __name__ == "__main__":
    main()