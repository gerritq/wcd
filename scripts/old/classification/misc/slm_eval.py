import os
import json
import re
import time
import random
import torch
import evaluate
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import AutoTokenizer, set_seed, logging
from peft import AutoPeftModelForCausalLM
from collections import defaultdict
from prompts import SYSTEM_PROMPTS_SLM
from typing import List
from utils import MODEL_MAPPING
import argparse
from sklearn.metrics import confusion_matrix

# Ignore "The following generation flags are not valid and may be ignored: ['temperature', 'top_p', 'top_k']. Set `TRANSFORMERS_VERBOSITY=info` for more details."
logging.set_verbosity_error()

acc_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

BASE_DIR = "/scratch/prj/inf_nlg_ai_detection/wcd"
MODEL_DIR = os.path.join(BASE_DIR, "data/models/slm")
SETS_DIR = os.path.join(BASE_DIR, "data/sets/main")
METRICS_DIR = os.path.join(BASE_DIR, "data/metrics/slm")

SEED=42
MAX_LENGTH = 256*4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument("--model_number", type=str, required=True)
parser.add_argument("--languages", nargs="+", required=True)
parser.add_argument("--mode", type=str, required=True)
args = parser.parse_args()


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

# def build_messages(dataset: Dataset, system: bool) -> Dataset:

#     def preprocess_function(example):
#         if args.system:
#             PROMPT = SYSTEM_PROMPTS_SLM[example['lang']]
#         else:
#             # to do
#             pass
        
#         if system:
#             x = {
#             "messages": [
#                 {"role": "system", "content": PROMPT['system']},
#                 {"role": "user", "content": PROMPT['user'].format(claim=example['claim'])}
#             ]
#         }

#         else:
#             # to do
#             pass
#         return x
    
#     data = list(dataset) 
#     random.shuffle(data)

#     dataset = dataset.map(preprocess_function, remove_columns=["claim", "label"])
#     return dataset

def build_messages_labels(test_ds: Dataset) -> Dataset:
    claims = test_ds["claim"]
    labels = test_ds["label"]
    langs = test_ds["lang"]
    
    msgs = []
    for i in range(len(claims)):
        msg = {
                "messages": [
                        {"role": "system", "content": SYSTEM_PROMPTS_SLM[langs[i]]['system']},
                        {"role": "user", "content": SYSTEM_PROMPTS_SLM[langs[i]]['user'].format(claim=claims[i])},
                ]
            }
        msgs.append(msg)

    return msgs, labels, langs

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
    for i in range(0, len(messages), batch_size):
        batch = messages[i:i+batch_size]
        chat_texts = []
        for msg in batch:
            chat = tokenizer.apply_chat_template(
                msg["messages"],
                tokenize=False,
                add_generation_prompt=True, # important to make it answer 
                enable_thinking=False
            )
            chat_texts.append(chat)
            # print(chat)

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

def evaluation(model_path: str,
         meta: dict,
         test_data_name: str,
         batch_size: int = 16):

    # Load tokeniser and model
    tokenizer = AutoTokenizer.from_pretrained(meta['model'], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    tokenizer.truncation_side = "left"

    model = AutoPeftModelForCausalLM.from_pretrained(
        model_path, device_map="auto", torch_dtype="auto", trust_remote_code=True
        )
    model.eval()

    # Get and prep data
    test = load_from_disk(os.path.join(SETS_DIR, test_data_name))['test']
    # PROMPT = SYSTEM_PROMPTS_SLM[meta['data']]

    messages, labels, langs = build_messages_labels(test)
    is_llama = True if "llama" in meta['model'].lower() else False

    # Run predictions
    start = time.time()
    preds = predict(model, 
                    tokenizer, 
                    messages, 
                    batch_size, 
                    is_llama=is_llama)

    # Zip predictions with the label and language
    valid = [(p, y, l) for p, y, l in zip(preds, labels, langs) if p is not None]

    if valid:
        p_clean, y_clean, l_clean = zip(*valid)
        overall = compute_metrics(list(p_clean), list(y_clean))
    else:
        raise ValueError("No valid predictions.")

    # Results by language
    by_lang = {}
    if valid:
        bucket = defaultdict(lambda: {"preds": [], "refs": []})
        for p, y, l in zip(p_clean, y_clean, l_clean):
            bucket[l]["preds"].append(p)
            bucket[l]["refs"].append(y)
        for l, d in bucket.items():
            by_lang[l] = compute_metrics(d["preds"], d["refs"])
            
    end = time.time()

    meta['eval'][test_data_name] = {
                "test_data_name": test_data_name,
                "n_total": len(labels),
                "n_scored": len(valid),
                "time_min": (end - start) / 60.0,
                "accuracy": overall["accuracy"],
                "f1": overall["f1"],
                "by_lang": by_lang
    }

    return meta

def main():
    start = time.time()

    model_path = os.path.join(MODEL_DIR, args.model_number) 
    
    # Load meta
    with open(os.path.join(model_path, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)

    if not os.path.exists(model_path):
        print(f"Model dir does not exit {model_dir}. Skipping.")
        sys.exit(0)

    print(f"EVALUATION {args.model_number} - Model {meta['model']} - Train {meta['data']}")    
    
    if args.mode == "all":
        test_langs = args.languages
    elif args.mode == "self":
        test_langs = [meta['data']]
    elif args.mode == "ool":
        ool_langs = args.languages[:]
        ool_langs.remove(meta['data'])
        test_langs = ool_langs
    else:
        raise Exception(f"Incorrect mode: {args.mode}")
    
    print(f"\tEvaluation languages: {test_langs}")

    meta['eval'] = {}

    for lang in test_langs:
        
        print(f"\tEvaluating {lang}")
        # adds results to meta eval
        evaluation(model_path, meta, lang)
        
        
    # out_path = os.path.join(METRICS_DIR, f"{meta['data']}_2_{lang}_model{meta['model_number']}.json")
    out_path = os.path.join(METRICS_DIR, f"model_{meta['model_number']}_2.json")
    with open(out_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)    
    
            
if __name__ == "__main__":
    main()