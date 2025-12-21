import os
import re
import json
import argparse
from concurrent.futures import ThreadPoolExecutor
from datasets import load_from_disk, DatasetDict, concatenate_datasets, Dataset
from openai import OpenAI
from prompts import (COT_SHORT_PROMPT, 
                     COT_PROMPT, 
                     COT_WIKI_SHORT_PROMPT, 
                     COT_WIKI_PROMPT
                    )
import random

BASE_DIR = os.getenv("BASE_WCD") 
DATA_DIR = os.path.join(BASE_DIR, "data/sets/main")
OUT_DIR = os.path.join(BASE_DIR, "data/sets/annotation")

def resample_train_data(train_ds: Dataset, total_size: int) -> Dataset:
    """
    Resample a dataset to total_size with 50/50 label balance.
    Returns a new Dataset; does not use or modify `self`.
    """
    assert total_size % 2 == 0, "Total size must be even."
    n_per_label = total_size // 2

    pos_all = [x for x in train_ds if x["label"] == 1]
    neg_all = [x for x in train_ds if x["label"] == 0]

    pos = pos_all[:min(len(pos_all), n_per_label)]
    neg = neg_all[:min(len(neg_all), n_per_label)]

    combined = pos + neg
    random.shuffle(combined)

    pos_count, neg_count = 0, 0
    for x in combined:
        if x["label"] == 1:
            pos_count += 1
        else:
            neg_count += 1
    assert pos_count == neg_count, "Dataset unbalanced after resampling."

    return Dataset.from_list(combined)
    

def get_dataset(lang: str,
                training_size: int, 
                smoke_test: bool) -> list[dict]:
    """
    Load full dataset and return as a single list.
    """
    ds = load_from_disk(os.path.join(DATA_DIR, lang))
    
    train = ds["train"].add_column("split", ["train"] * len(ds["train"]))

    if training_size != len(train):
        train = resample_train_data(train, training_size)

    dev   = ds["dev"].add_column("split", ["dev"] * len(ds["dev"]))
    test  = ds["test"].add_column("split", ["test"] * len(ds["test"]))

    if smoke_test:
        train = train.select(range(3))
        dev = dev.select(range(3))
        test = test.select(range(3))
        print("Smoke testing.")
    
    # combined = concatenate_datasets([train, dev, test])
    # return combined.to_list()
    return train.to_list(), dev.to_list(), test.to_list()


def format_prompt(ds: list,
                  prompt_template: dict,
                  lang: str,
                  context: bool,
                  smoke_test: bool):
    """
    Takes a ds as a list. Format the prompt of each item.
    """
    prompt_template = prompt_template[lang]
    for claim in ds:
        claim['prompt'] = prompt_template.format(**claim)
        if smoke_test:
            print(claim['prompt'])
    return ds

def query(claim: dict, 
          model: str, 
          client: OpenAI,
          smoke_test: bool) -> dict:
    """
    Takes a claim as a dict, queries model and returns the claim dict with category and explanation.
    """

    completion = client.chat.completions.create(
        model=model,
        temperature=0.1,
        messages=[{"role": "user", "content": claim['prompt']}]
    )

    r = retrieve_response(completion.choices[0].message.content)
    if r:
        if len(r) > 1:
            claim['reason'] = r[0]
            claim['explanation'] = r[0]
        else:
            claim['response'] = r[0]
    if smoke_test:
        print('Final response:')
        print(r)
    return claim

def retrieve_response(response: str) -> str:
    """
    Takes a response as a str. 
    Find the json key-value paris and returns them.
    Expected to find and return two: category and explanation.
    """
    # match = re.search(r"[```]?json\s*(\{.*?\})\s*[```]?", response, re.DOTALL)
    # match = re.search(r"\{\"rationale\":['\"](.*)[\"']\}", response, re.DOTALL)
    match = re.search(r"\{.*?\}", response, re.DOTALL)
    if match:
        json_str = match.group(0)
        try:
            data = json.loads(json_str)
            # return data["category"], data["explanation"]
            if len(data)>1:
                return [data["reason"], data["explanation"]]
            else:
                return [data["response"]]
            # return None, None
        except json.JSONDecodeError:
            print("Failed to decode JSON.")
            return None
    else:
        print("No JSON block found in response.")
        return None

def run(prompts: list[dict], 
        model: str, 
        client: OpenAI,
        smoke_test: bool) -> list[dict]:
    def process(claim):
        out = query(claim=claim, model=model, client=client, smoke_test=smoke_test)
        return out

    with ThreadPoolExecutor(max_workers=4) as executor:
        annotated_claims = list(executor.map(process, prompts))

    # rm prompt
    for x in annotated_claims:
        x.pop("prompt")
    return annotated_claims


def save_data(train: list[dict],
              dev: list[dict],
              test: list[dict],
              lang: str, 
              prompt_name: str
              ) -> None:
    """
    Converts a list of dicts into a DatasetDict with train/dev/test splits.
    Saves this in the path.
    """
    # train = [x for x in data if x["split"] == "train"]
    # dev   = [x for x in data if x["split"] == "dev"]
    # test  = [x for x in data if x["split"] == "test"]

    train_ds = Dataset.from_list(train)
    dev_ds   = Dataset.from_list(dev)
    test_ds  = Dataset.from_list(test)

    ds = DatasetDict({
        "train": train_ds,
        "dev": dev_ds,
        "test": test_ds,
    })

    out_path = os.path.join(OUT_DIR, f"{lang}_{prompt_name}")
    ds.save_to_disk(out_path)

def main():
    # argpaese
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--context", type=int, required=True)
    parser.add_argument("--training_size", type=int, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--smoke_test", type=int, required=True)
    args = parser.parse_args()
    args.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"]
            )
    if args.smoke_test:
        print("="*20)
        print("Running smoke test..")
    args.smoke_test = bool(args.smoke_test)
    args.context = bool(args.context)

    print("="*20)
    print(vars(args))

    if args.prompt == "cot_short":
        prompt_template = COT_SHORT_PROMPT
    if args.prompt == "cot":
        prompt_template = COT_PROMPT
    if args.prompt == "cot_wiki_short":
        prompt_template = COT_WIKI_SHORT_PROMPT
    if args.prompt == "cot_wiki":
        prompt_template = COT_WIKI_PROMPT

    
    # get data
    print("="*20)
    print("Getting data..")
    train, dev, test = get_dataset(lang=args.lang, 
                       training_size=args.training_size,       
                       smoke_test=args.smoke_test
                       )

    # get prompts
    print("="*20)
    print("Get prompts ...  ...")
    prompts = format_prompt(ds=train, 
                            prompt_template=prompt_template,
                            lang=args.lang, 
                            context=args.context, 
                            smoke_test=args.smoke_test
                            )
    
    # annotate claims
    print("="*20)
    print("Annotate data ...")
    annotated_claims = run(prompts=prompts, 
                           model=args.model, 
                           client=args.client,
                           smoke_test=args.smoke_test
                           )
    # save data
    save_data(train=annotated_claims, # this is train
              dev=dev,
              test=test,
              lang=args.lang, 
              prompt_name=args.prompt
              )

if __name__ == "__main__":
    main()