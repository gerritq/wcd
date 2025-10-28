import os
import json
import re
import time
import random
import torch
import evaluate
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import AutoTokenizer, set_seed, logging, AutoModelForCausalLM
from collections import defaultdict
from prompts import SYSTEM_PROMPTS_SLM
from typing import List
from utils import MODEL_MAPPING, get_model_number
import argparse
from sklearn.metrics import confusion_matrix
from datetime import datetime
from tqdm import tqdm

# Ignore "The following generation flags are not valid and may be ignored: ['temperature', 'top_p', 'top_k']. Set `TRANSFORMERS_VERBOSITY=info` for more details."
logging.set_verbosity_error()

acc_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

BASE_DIR = "/scratch/prj/inf_nlg_ai_detection/wcd"
SETS_DIR = os.path.join(BASE_DIR, "data/sets/main")
METRICS_DIR = os.path.join(BASE_DIR, "data/metrics/slm/icl")
SHOTS_DIR = os.path.join(BASE_DIR, "data/sets/shots")

SEED=42
MAX_LENGTH = 256*4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--lang", nargs="+", required=True)
parser.add_argument("--shots", type=int, required=True)
args = parser.parse_args()
args.shots = bool(args.shots)

print(f"Using device: {DEVICE}")

set_seed(SEED)

def load_test_sets(sets_dir: str):
    test_data = []
    for d in os.listdir(sets_dir):
        # print(d)
        path = os.path.join(sets_dir, d)
        if os.path.isdir(path):
            test_name = os.path.basename(path)
            ds = load_from_disk(path)
            if "test" in ds:
                test_data.append((test_name, ds["test"]))
            else:
                raise KeyError("Key test does not exist.")
    return test_data

def build_messages(dataset: Dataset, lang):
    data = list(dataset)
    
    messages, labels = [], []

    if args.shots:
        shots_path = os.path.join(SHOTS_DIR, "shots.json")
        with open(shots_path, "r", encoding="utf-8") as f:
            shots_all = json.load(f)
        lang_shots = shots_all[lang]
        shot_lines = [
            f"Claim: {s['claim']}\nAnswer: {SYSTEM_PROMPTS_SLM[lang]['assistant'].format(label=s['label'])}"
            for s in lang_shots
        ]
        system_content = SYSTEM_PROMPTS_SLM[lang]["system"] + "\n\nExamples:\n" + "\n\n".join(shot_lines)
    else:
        system_content = SYSTEM_PROMPTS_SLM[lang]["system"]
        
    for ex in data:
        user = SYSTEM_PROMPTS_SLM[lang]["user"].format(claim=ex["claim"])
        messages.append([
            {"role": "system", "content": system_content},
            {"role": "user", "content": user},
        ])
        labels.append(ex['label'])

    return messages, labels


    # def preprocess_function(example):
        
    #     if args.shots:
    #         shots_path = os.path.join(SHOTS_DIR, f"shots.json")
    #         with open(shots_path, "r", encoding="utf-8") as f:
    #             shots = json.load(f)
    #             shots = shots[example['lang']]
    #             shots = [f"Claim: {x['claim']}\nAnswer: {SYSTEM_PROMPTS_SLM[example['lang']]['assistant'].format(label=x['label'])}" x for x in shots]
    #         system_message = SYSTEM_PROMPTS_SLM[example['lang']]['system'] + "\n\nExamples:\n" + "\n\n".join(shots)
            
    #         x = {
    #         "messages": [
    #             {"role": "system", "content": system_message},
    #             {"role": "user", "content": SYSTEM_PROMPTS_SLM[example['lang']]['user'].format(claim=example['claim'])},
    #             {"role": "assistant", "content": SYSTEM_PROMPTS_SLM[example['lang']]['assistant'].format(label=example['label'])}
    #             ]
    #         }
    #     else:
    #         x = {
    #         "messages": [
    #             {"role": "system", "content": SYSTEM_PROMPTS_SLM[example['lang']]['system']},
    #             {"role": "user", "content": SYSTEM_PROMPTS_SLM[example['lang']]['user'].format(claim=example['claim'])},
    #             {"role": "assistant", "content": SYSTEM_PROMPTS_SLM[example['lang']]['assistant'].format(label=example['label'])}
    #             ]
    #         }
    #     print(x)
    #     return x
    
    # data = list(dataset) 
    # random.shuffle(data)

    # dataset = dataset.map(preprocess_function, remove_columns=["claim", "label"])
    # return dataset

def compute_metrics(preds, refs):
    
    acc = acc_metric.compute(predictions=preds, references=refs)["accuracy"]
    f1  = f1_metric.compute(predictions=preds, references=refs, average="binary")["f1"]

    tn, fp, fn, tp = confusion_matrix(refs, preds).ravel()

    return {
        "accuracy": acc,
        "f1": f1,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn)
    }

def predict(model, tokenizer, messages, batch_size, is_llama=False):
    preds = []
    for i in tqdm(range(0, len(messages), batch_size)):
        batch = messages[i:i+batch_size]
        chat_texts = []
        for msg in batch:
            # print(i, msg, "\n\n")
            chat = tokenizer.apply_chat_template(
                msg,
                tokenize=False,
                add_generation_prompt=True, # important to make it answer 
                enable_thinking=False
            )
            chat_texts.append(chat)
            # print(i, chat, "\n\n")

        enc = tokenizer(
            chat_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH
        ).to(model.device)

        terminators = [tokenizer.eos_token_id] if is_llama else None

        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=terminators,
            )

        for j in range(len(batch)):
            output_ids = out[j][len(enc["input_ids"][j]):].tolist()
            try:
                idx = len(output_ids) - output_ids[::-1].index(151668)  # </think> for Qwen
            except ValueError:
                idx = 0
            response = tokenizer.decode(output_ids[idx:], skip_special_tokens=True).strip()

            # print("answer", j, response)
            
            label = None
            match = re.search(r"<label>\s*([01])\s*</label>", response, re.DOTALL | re.IGNORECASE)
            if match:
                label = int(match.group(1))
            preds.append(label)
    return preds

def evaluation(model_name, lang, batch_size: int = 16):

    # Load tokeniser and model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    tokenizer.truncation_side = "left"

    model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                 device_map="auto", 
                                                 torch_dtype="auto", 
                                                 trust_remote_code=True)
    model.eval()

    # Get and prep data
    data_path = os.path.join(SETS_DIR, lang)
    test = load_from_disk(os.path.join(SETS_DIR, lang))['test']
    messages, labels = build_messages(test, lang)
    is_llama = True if "llama" in model_name.lower() else False

    # Run predictions
    start = time.time()
    preds = predict(model, 
                    tokenizer, 
                    messages, 
                    batch_size, 
                    is_llama=is_llama)

    # Zip predictions with the label and language
    valid = [(p, y) for p, y in zip(preds, labels) if p is not None]

    if valid:
        p_clean, y_clean = zip(*valid)
        metrics = compute_metrics(list(p_clean), list(y_clean))
    else:
        raise ValueError("No valid predictions.")

    end = time.time()

    meta = {"model": model_name,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "data": lang,
            "shots": args.shots,
            "n_total": len(labels),
            "n_scored": len(valid),
            "time_min": (end - start) / 60.0,
            "metrics": metrics
            }

    return meta

def main():
    start = time.time()

    for lang in args.lang:
        print(f"RUNNING {args.model} - Lang {lang} - SHOTS {args.shots}")
        
        model_name = MODEL_MAPPING[args.model]
        meta = evaluation(model_name, lang)

        model_number = get_model_number(METRICS_DIR)
        meta.update({"model_number": model_number})
            
        out_path = os.path.join(METRICS_DIR, f"model_{meta['model_number']}.json")
        with open(out_path, "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)    
    
if __name__ == "__main__":
    main()