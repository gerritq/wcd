import os
import re
import json
import argparse
from concurrent.futures import ThreadPoolExecutor
from datasets import load_from_disk, DatasetDict, concatenate_datasets, Dataset
from openai import OpenAI
from typing import List

PROMPT = (
    "You are a Wikipedia citation reasoning assistant. "
    "Your task is to determine *why* a given claim does or does not require a citation. "
    "Select the most appropriate category from the list below and provide a concise explanation. "
    "Return your answer as valid JSON in a json block in the format:\n"
    "{{\"category\": your_category, \"explanation\": your_explanation}}\n\n"

    "=== Reasons why citations are needed ===\n"
    "• Quotation – The statement is a direct quotation or close paraphrase of a source.\n"
    "• Statistics – The statement contains statistics or quantitative data.\n"
    "• Controversial – The statement makes surprising or potentially controversial claims (e.g., conspiracy theories).\n"
    "• Opinion – The statement expresses a person’s subjective opinion or belief.\n"
    "• Private Life – The statement contains claims about a person’s private life (e.g., date of birth, relationship status).\n"
    "• Scientific – The statement includes technical or scientific claims.\n"
    "• Historical – The statement makes general or historical claims that are not common knowledge.\n"
    "• Other (Needs Citation) – The statement requires a citation for other reasons (please describe briefly).\n\n"

    "=== Reasons why citations are NOT needed ===\n"
    "• Common Knowledge – The statement only contains well-known or established facts.\n"
    "• Main Section – The statement appears in the lead section and is cited elsewhere in the article.\n"
    "• Plot – The statement describes the plot or characters of a book, film, or similar work that is the article’s subject.\n"
    "• Already Cited – The statement is already supported by a citation elsewhere in the paragraph or article.\n"
    "• Other (No Citation Needed) – The statement does not need a citation for other reasons (please describe briefly).\n\n"

    "Now, analyze the following claim:\n"
    "Claim: {claim}\n"
    "Label: {label}\n\n"
    "Please choose one of the categories above and explain why it applies."
    )

BASE_DIR = os.getenv("BASE_WCD") 
DATA_DIR = os.path.join(BASE_DIR, "data/sets/main")
OUT_DIR = os.path.join(BASE_DIR, "data/sets/annotation")

def get_dataset(lang: str, smoke_test: bool) -> list[dict]:
    """
    Load full dataset and return as a single list.
    """
    ds = load_from_disk(os.path.join(DATA_DIR, lang))
    
    train = ds["train"].add_column("split", ["train"] * len(ds["train"]))
    dev   = ds["dev"].add_column("split", ["dev"] * len(ds["dev"]))
    test  = ds["test"].add_column("split", ["test"] * len(ds["test"]))

    if smoke_test:
        train = train.select(range(3))
        dev = dev.select(range(3))
        test = test.select(range(3))
        print("Smoke testing.")
    
    combined = concatenate_datasets([train, dev, test])
    
    return combined.to_list()


def format_prompt(ds: List):
    """
    Takes a ds as a list. Format the prompt of each item.
    """
    for claim in ds:
        claim['prompt'] = PROMPT.format(**claim)
    return ds

def query(claim: dict, 
          model: str, 
          client: OpenAI) -> str:
    """
    Takes a claim as a dict, queries model and returns the claim dict with category and explanation.
    """

    completion = client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[{"role": "user", "content": claim['prompt']}]
    )

    claim['category'], claim['explanation'] = retrieve_response(completion.choices[0].message.content)
    return claim

def retrieve_response(response: str) -> tuple[str, str]:
    """
    Takes a response as a str. 
    Find the json key-value paris and returns them.
    Expected to find and return two: category and explanation.
    """
    match = re.search(r"```json\s*(\{.*?\})\s*```", response, re.DOTALL)
    if match:
        json_str = match.group(1)
        print(json_str)
        try:
            data = json.loads(json_str)
            return data["category"], data["explanation"]
        except json.JSONDecodeError:
            print("Failed to decode JSON.")
            return None, None
            
    else:
        print("No JSON block found in response.")
        return None, None

def run(prompts: list[dict], 
        model: str, 
        client: OpenAI) -> list[dict]:
    def process(claim):
        out = query(claim, model, client)
        return out

    with ThreadPoolExecutor(max_workers=4) as executor:
        annotated_claims = list(executor.map(process, prompts))

    # rm prompt
    for x in annotated_claims:
        x.pop("prompt")
    return annotated_claims


def save_data(data: list[dict], lang: str) -> DatasetDict:
    """
    Converts a list of dicts into a DatasetDict with train/dev/test splits.
    Saves this in the path.
    """
    train = [x for x in data if x["split"] == "train"]
    dev   = [x for x in data if x["split"] == "dev"]
    test  = [x for x in data if x["split"] == "test"]

    train_ds = Dataset.from_list(train)
    dev_ds   = Dataset.from_list(dev)
    test_ds  = Dataset.from_list(test)

    ds = DatasetDict({
        "train": train_ds,
        "dev": dev_ds,
        "test": test_ds,
    })

    out_path = os.path.join(OUT_DIR, f"{lang}")
    ds.save_to_disk(out_path)

def main():
    # argpaese
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--smoke_test", type=int, required=True)
    args = parser.parse_args()
    args.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"]
            )
    args.smoke_test = bool(args.smoke_test)

    # get data
    data = get_dataset(args.lang, args.smoke_test)
    
    # get prompts
    prompts = format_prompt(data)
    
    # annotate claims
    annotated_claims = run(prompts, 
                           args.model, 
                           args.client)
    
    save_data(annotated_claims, args.lang)

if __name__ == "__main__":
    main()