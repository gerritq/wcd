import os
import json
import re
import random
import argparse
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from datasets import load_from_disk
from openai import OpenAI
from prompts import SYSTEM_PROMPTS_SLM, SYSTEM_PROMPTS_LLM
from sklearn.metrics import accuracy_score, f1_score
from datetime import datetime
import time
from utils import (
                    MODEL_MAPPING, 
                    append_meta_file
)
from utils import MODEL_MAPPING

BASE_DIR = os.getenv("BASE_WCD")
DATA_DIR = os.path.join(BASE_DIR, "data/sets")
METRICS_DIR = os.path.join(BASE_DIR, "data/metrics/llm")
SHOTS_DIR = os.path.join(BASE_DIR, "data/sents/shots")

parser = argparse.ArgumentParser()
parser.add_argument("--lang", type=str, required=True)
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--shots", type=int, required=True)
parser.add_argument("--system", type=int, required=True)
parser.add_argument("--verbose", type=int, required=True)
parser.add_argument("--notes", type=str)
args = parser.parse_args()
args.shots = bool(args.shots)
args.system = bool(args.system)
args.verbose = bool(args.verbose)
args.model = MODEL_MAPPING[args.model]

def get_model_number(model_dir: str) -> int:
    '''Function is differnt than in utils, as we are not chekcing for dir but files'''
    model_names = [d for d in os.listdir(model_dir) if d.startswith("model_")] # and os.path.isdir(os.path.join(model_dir, d)
    
    numbers = []
    for name in model_names:
        
        num = int(name.split("_")[1].replace(".json",""))
        numbers.append(num)
        
    next_number = max(numbers) + 1 if numbers else 1
    return next_number


def format_messages(ds, args: dict) -> List[dict]:

    if args.verbose:
        PROMPT = SYSTEM_PROMPTS_LLM
    else:
        PROMPT = SYSTEM_PROMPTS_SLM

    if args.shots:
        def format_few_shot(example: dict, shots: List[Dict]):
                shots = [
                        f"Claim: {s['claim']}\nAnswer: <label>{s['label']}</label>" for s in shots
                        ]
                system_message = PROMPT[example['lang']]['system'] + "\n\nExamples:\n" + "\n\n".join(shots)
                return {"messages": [
                                    {"role": "system", "content": system_message},
                                    {"role": "user", "content": PROMPT[example['lang']]['user'].format(claim=example['claim'])}
                                ],
                        "label": int(example["label"]),
                        "claim": example["claim"]
                        }

        # load data and create messages
        shots_path = os.path.join(SHOTS_DIR, f"shots.json")
        with open(shots_path, "r", encoding="utf-8") as f:
            shots = json.load(f)
            shots = shots[args.lang]
        rows = [format_few_shot(ex, shots) for ex in ds]
    else:
        def format_zero_shot(example: dict):
            return {"messages": [
                                {"role": "system", "content": PROMPT[example['lang']]['system']},
                                {"role": "user", "content": PROMPT[example['lang']]['user'].format(claim=example['claim'])},
                                ],
                    "label": int(example["label"]),
                    "claim": example["claim"],
                    "lang": example["lang"]
                    }
            # load data and create messages
        rows = [format_zero_shot(ex) for ex in ds]
    
    # out
    random.shuffle(rows)
    return rows
    
def find_response(response: str) -> int:
    match = re.search(r"<label>\s*([01])\s*</label>", response, re.DOTALL | re.IGNORECASE)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            print("Could not convert label to int.")
            return None
    else:
        print("No <label>...</label> block found in response.")
        return None


def query(client, model: str, messages: List[Dict]) -> int:
        completion = client.chat.completions.create(
            model=model,
            temperature=0.0,
            messages=messages,
        )
        content = completion.choices[0].message.content
        return find_response(content)

def eval(results: List[Dict], 
         model_number: str,
         test_n: int,
         args) -> None:

    # get valids, ys, and preds
    valid = [r for r in results if r["pred"] is not None]
    y_true = [int(r["label"]) for r in valid]
    y_pred = [int(r["pred"]) for r in valid]

    # get scores
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    res = {'model_number': model_number, 
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'time_mins': args.time_mins,
            'model': args.model,
            'data': args.lang,
            'shots': args.shots,
            'system': args.system,
            'verbose': args.verbose,
            'test_n': test_n,
            'valid_n': len(valid),
            'notes': args.notes,
            "f1": f1,
            'accuracy': acc
            }

    metrics_path = os.path.join(METRICS_DIR, f"model_{model_number}.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

def main() -> None:
    start = time.time()
    # load data and get model number
    # ds = load_from_disk(os.path.join(DATA_DIR, args.lang))["test"].select(range(10))
    ds = load_from_disk(os.path.join(DATA_DIR, args.lang))["test"]
    model_number = get_model_number(METRICS_DIR)

    test_data = format_messages(ds, args)

    # test_data = test_data[:1]
    # print(test_data[:2])

    # initialise client
    client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"]
            )

    print(f"\nRUNNING MODEL={args.model} - LANGUAGE={args.lang}")
    def worker(example: Dict) -> Dict:
        try:
            # print("\n\n--- PROMPT ---")
            # for msg in example["messages"]:
            #     print(f"{msg['role'].upper()}: {msg['content']}")
            # print("--- END PROMPT ---\n")

            pred = query(client, args.model, example["messages"])
            return {
                "claim": example["claim"],
                "label": example["label"],
                "pred": pred,
            }
        except Exception as e:
            print(e)
            return {
                "claim": example["claim"],
                "label": example["label"],
                "pred": None,
                "error": str(e),
            }

    results = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = [ex.submit(worker, exm) for exm in test_data]
        for fut in as_completed(futures):
            results.append(fut.result())

    end = time.time()
    time_mins = (end - start) / 60.0
    args.time_mins = time_mins
    # eval
    eval(results, model_number, len(test_data), args)



if __name__ == "__main__":


    main()
