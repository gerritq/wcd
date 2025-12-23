import os
import json
import re
import random
import argparse
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from datasets import load_from_disk
from openai import OpenAI
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from datetime import datetime
import time
from argparse import Namespace
from prompts import INSTRUCT_PROMPT, VERBOSE_PROMPT
# --------------------------------------------------------------------------------------------------
# Some utils
# --------------------------------------------------------------------------------------------------


PROMPT_LANGUAGE_MAP = {
    "en": "English",
    "pt": "Portuguese",
    "de": "German",
    "ru": "Russian",
    "it": "Italian",
    "vi": "Vietnamese",
    "tr": "Turkish",
    "nl": "Dutch",
    "uk": "Ukrainian",
    "ro": "Romanian",
    "id": "Indonesian",
    "bg": "Bulgarian",
    "uz": "Uzbek",
    "no": "Norwegian",
    "az": "Azerbaijani",
    "mk": "Macedonian",
    "hy": "Armenian",
    "sq": "Albanian",
}


# PROMPT = {
#     "system": (
#         "You are a multilingual classifier. "
#         "Decide whether the given {lang} claim requires a citation. "
#         "Use 1 if the claim needs a citation. Use 0 if the claim does not need a citation.\n\n"
#         "Return only JSON in the format: {{\"label\": 0}} or {{\"label\": 1}}. "
#         "No explanations or extra text."
#     ),
#     "user_claim": "Claim: {claim}",
#     "user_context": (
#         "Section: {section}\n"
#         "Previous Sentence: {previous_sentence}\n"
#         "Claim: {claim}\n"
#         "Subsequent Sentence: {subsequent_sentence}"
#     ),
# }

# PROMPT_VERBOSE = {
#     "system": (
#         "You are a multilingual Wikipedia citation classifier. "
#         "You are provided with a {lang} claim and its context. "
#         "Your task is to analyze the claim and the context to decide whether the claim needs a citation. "
#         "On Wikipedia, there are various reasons why a claim may or may not require a citation. The reasons are listed below:\n\n"
#         "# Reasons why citations are needed (Label 1)\n"
#         "• Quotation – The statement is a direct quotation or close paraphrase of a source.\n"
#         "• Statistics – The statement contains statistics or quantitative data.\n"
#         "• Controversial – The statement makes surprising or potentially controversial claims.\n"
#         "• Opinion – The statement expresses a person’s subjective opinion or belief.\n"
#         "• Private Life – The statement contains claims about a person’s private life (e.g., date of birth, relationship status).\n"
#         "• Scientific – The statement includes technical or scientific claims.\n"
#         "• Historical – The statement makes general or historical claims that are not common knowledge.\n"
#         "• Other (Needs Citation) – The statement requires a citation for other reasons (briefly describe why).\n\n"
#         "# Reasons why citations are not needed (Label 0)\n"
#         "• Common Knowledge – The statement contains well-known or widely established facts.\n"
#         "• Plot – The statement describes the plot or characters of a book, film, or similar work that is the subject of the article.\n"
#         "• Other (No Citation Needed) – The statement does not require a citation for other reasons (briefly describe why).\n\n"
#         "Based on these reasons, think step-by-step to decide in which category the claim falls. "
#         "Return only JSON in the format: {{\"label\": 0}} or {{\"label\": 1}}. "
#         "No extra text."
#     ),
#     "user_claim": "Claim: {claim}",
#     "user_context": (
#         "Section: {section}\n"
#         "Previous Sentence: {previous_sentence}\n"
#         "Claim: {claim}\n"
#         "Subsequent Sentence: {subsequent_sentence}"
#     ),
# }



BASE_DIR = os.getenv("BASE_WCD")
DATA_DIR = os.path.join(BASE_DIR, "data/sets/main")
OUT_DIR = os.path.join(BASE_DIR, "data/exp1")
TEST_DIR = os.path.join(BASE_DIR, "data/exp1_test")
SHOTS_DIR = os.path.join(BASE_DIR, "data/sets/shots")

def get_model_number(model_dir: str) -> int:
    '''Function is differnt than in utils, as we are not chekcing for dir but files'''
    model_names = [d for d in os.listdir(model_dir) if d.startswith("meta_")] # and os.path.isdir(os.path.join(model_dir, d)
    
    numbers = []
    for name in model_names:
        
        num = int(name.split("_")[1].replace(".json",""))
        numbers.append(num)
        
    next_number = max(numbers) + 1 if numbers else 1
    return next_number


def format_messages(args: dict,
                    data: List[dict], 
                    prompt_template: str
                    ) -> List[dict]:
    
    lang = PROMPT_LANGUAGE_MAP[args.lang]
    system_message = prompt_template['system'].format(lang=lang)
    
    if args.shots:
        def format_few_shot(system_message: str, 
                            example: dict, 
                            shots: List[Dict]
                            ) -> dict:
                # select shots                  
                shots = [
                            (
                            f"Section: {s['section']}\n"
                            f"Previous sentence: {s['previous_sentence']}\n"
                            f"Claim: {s['claim']}\n"
                            f"Subsequent sentence: {s['claim']}\n"
                            f"Label: {{\"label\": {s['label']}}}"
                            ) for s in shots
                        ]

                system_message += "\n\nExamples:\n" + "\n\n".join(shots) 
                return {"messages": [
                                    {"role": "system", "content": system_message},
                                    {"role": "user", "content": prompt_template['user'].format(**example)}
                                ],
                        "label": int(example["label"]),
                        "claim": example["claim"],
                        "lang": example["lang"]
                        }

        # load data and create messages
        shots_path = os.path.join(SHOTS_DIR, f"shots.json")
        with open(shots_path, "r", encoding="utf-8") as f:
            shots = json.load(f)
            shots = shots[args.lang]
        rows = [format_few_shot(system_message, ex, shots) for ex in data]
    else:
        def format_zero_shot(system_message: str, example: dict):
            
            return {"messages": [
                                {"role": "system", "content": system_message},
                                {"role": "user", "content": prompt_template['user'].format(**example)},
                                ],
                    "label": int(example["label"]),
                    "claim": example["claim"],
                    "lang": example["lang"]
                    }
            # load data and create messages
        rows = [format_zero_shot(system_message, ex) for ex in data]
    
    # out
    random.shuffle(rows)
    return rows
    
def find_response(response: str) -> int:
    

    if '"label": 1' in response:
        return 1
    elif '"label": 0' in response:
        return 0
    else:
        print("No JSON label found in response.")
        return -1


def query(client, model: str, messages: List[Dict]) -> int:
        completion = client.chat.completions.create(
            model=model,
            temperature=0.1,
            messages=messages,
        )
        content = completion.choices[0].message.content
        return find_response(content)

def evaluate(results: List[Dict], 
             test_n: int,
             args: Namespace
             ) -> None:

    # get valids, ys, and preds
    valid = [r for r in results if r["pred"] is not None and r["pred"] != -1]
    y_true = [int(r["label"]) for r in valid]
    y_pred = [int(r["pred"]) for r in valid]

    # get scores
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    res = {"model_type": "icl",
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'time_mins': args.time_mins,
            'model_name': args.model,
            'lang': args.lang,
            'shots': args.shots,
            'context': args.context,
            'verbose': args.verbose,
            'test_n': test_n,
            'valid_n': len(valid),
            "test_metrics": {
                'accuracy': acc,
                'f1': f1,
                'tp': int(tp),
                'tn': int(tn),
                'fp': int(fp),
                'fn': int(fn),  
            }
    }
    if args.smoke_test:
        save_path = os.path.join(TEST_DIR, args.lang)
        meta_number = get_model_number(save_path)
        save_path = os.path.join(save_path, f"meta_{meta_number}.json")
    else:
        save_path = os.path.join(args.run_dir, f"meta_1.json")
    
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

def run(args: Namespace, data: list[dict]) -> None:
    start = time.time()

    print("="*20)
    print(f"RUNNING MODEL={args.model} | LANGUAGE={args.lang} | SHOTS={args.shots} | CONTEXT={args.context}")
    print("="*20)
    
    def worker(example: Dict) -> Dict:
        try:
            if args.smoke_test:
                print("\n\n--- PROMPT ---")
                for msg in example["messages"]:
                    print(f"{msg['role'].upper()}: {msg['content']}")
                print("--- END PROMPT ---\n")

            pred = query(client=args.client, 
                         model=args.model, 
                         messages=example["messages"]
                         )
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
        futures = [ex.submit(worker, exm) for exm in data]
        for fut in as_completed(futures):
            results.append(fut.result())

    end = time.time()
    time_mins = (end - start) / 60.0
    args.time_mins = time_mins
    # eval
    evaluate(results, len(data), args)
    
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--shots", type=int, required=True)
    parser.add_argument("--context", type=int, required=True)
    parser.add_argument("--run_dir", type=str, default="")
    parser.add_argument("--verbose", type=int, required=True)
    parser.add_argument("--smoke_test", type=int, required=True)
    args = parser.parse_args()
    
    assert args.shots in [0,1], "Shots must be 0 or 1"
    assert args.context in [0,1], "Context must be 0 or 1"
    assert args.smoke_test in [0,1], "Smoke test must be 0 or 1"
    assert args.verbose in [0,1], "Verbose must be 0 or 1"

    args.shots = bool(args.shots)
    args.smoke_test = bool(args.smoke_test)
    args.context = bool(args.context)
    args.verbose = bool(args.verbose)

    # Select the correct prompt user message
    
    if args.verbose:
        prompt_template = VERBOSE_PROMPT
    else:
        prompt_template = INSTRUCT_PROMPT
    
    if args.context:
        prompt_template['user'] = prompt_template['user_context']
    else:
        prompt_template['user'] = prompt_template['user_claim']

    # load data and get model number
    # ds = load_from_disk(os.path.join(DATA_DIR, args.lang))["test"].select(range(10))
    if args.smoke_test:
        print("="*20)
        print("RUNNING SMOKE TEST")
        print("="*20)
        data = load_from_disk(os.path.join(DATA_DIR, args.lang))["test"].select(range(3))
    else:
        data = load_from_disk(os.path.join(DATA_DIR, args.lang))["test"]
        
    # Format test data
    data = format_messages(args=args, 
                           data=data, 
                           prompt_template=prompt_template)

    # initialise client
    args.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"]
            )

    run(args=args, data=data)

if __name__ == "__main__":
    main()
