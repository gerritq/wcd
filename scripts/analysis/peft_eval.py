import os
import json
import re
import time
import random
import torch
import evaluate
import sys
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import AutoTokenizer, set_seed
from peft import AutoPeftModelForCausalLM
from collections import defaultdict
from prompts import PROMPTS
from typing import List
from utils import MODEL_MAPPING
from tqdm import tqdm

acc_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

BASE_DIR = "/scratch/prj/inf_nlg_ai_detection/wcd"
MODEL_DIR = os.path.join(BASE_DIR, "data/models/slm")
SETS_DIR = os.path.join(BASE_DIR, "data/sets")
METRICS_DIR = os.path.join(BASE_DIR, "data/metrics/slm")

SEED=42
MAX_LENGTH = 256*4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REVERSE_MODEL_MAPPING = {v: k for k, v in MODEL_MAPPING.items()}

print(f"Using device: {DEVICE}")

set_seed(SEED)
random.seed(SEED)

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

def build_messages_labels(test_ds: Dataset, PROMPT: str, system: bool) -> Dataset:
    claims = test_ds["claim"]
    labels = test_ds["label"]
    langs = test_ds["lang"] if "lang" in test_ds.column_names else ["all"] * len(labels)
    
    msgs = []
    for c in claims:

            if system:
                msg = {
                        "messages": [
                            {"role": "system", "content": PROMPT},
                            {"role": "user", "content": f"Claim: {c}"},
                        ]
                    }
                msgs.append(msg)
            else:
                msg = {
                        "messages": [
                            {"role": "system", "content": "You are a seasoned Wikipedia fact-checker."},
                            {"role": "user", "content": PROMPT.replace("{{claim}}", c)},
                        ]
                    }
                msgs.append(msg)

    return msgs, labels, langs

def compute_metrics(preds, refs):
    acc = acc_metric.compute(predictions=preds, references=refs)["accuracy"]
    f1  = f1_metric.compute(predictions=preds, references=refs, average="binary")["f1"]
    return {"accuracy": acc, "f1": f1}


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
                max_new_tokens=64,
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
            content = tokenizer.decode(output_ids[idx:], skip_special_tokens=True).strip()

            lab = None
            try:
                obj = json.loads(content)
                lab = int(obj.get("label", None))
            except Exception:
                m = re.search(r'"label"\s*:\s*(\d)', content)
                if m:
                    lab = int(m.group(1))
            if lab not in (0, 1):
                lab = None
            preds.append(lab)
    return preds

def eval(model_path: str,
         model_hf_name: str, 
         train_data_name: str,
         test_data_name: str, 
         test_data: Dataset,
         PROMPT: str,
         system: bool,
         batch_size: int = 8):

    # tokeniser
    tokenizer = AutoTokenizer.from_pretrained(model_hf_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    tokenizer.truncation_side = "left"

    # load the model
    model = AutoPeftModelForCausalLM.from_pretrained(
        model_path, device_map="auto", torch_dtype="auto", trust_remote_code=True
        )
    model.eval()

    # gen data
    messages, labels, langs = build_messages_labels(test_data, PROMPT, system)
    is_llama = True if "llama" in model_hf_name.lower() else False

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

    res = {
        "model_path": model_path,
        "train_data_name": train_data_name,
        "test_data_name": test_data_name,
        "n_total": len(labels),
        "n_scored": len(valid),
        "time_min": (end - start) / 60.0,
        "accuracy": overall["accuracy"],
        "f1": overall["f1"],
        "by_lang": by_lang
    }

    return res

def main():
    # Get test data
    test_data = load_test_sets(SETS_DIR)
    
    # Get SLM dirs only
    model_paths = [os.path.join(MODEL_DIR, d) for d in os.listdir(MODEL_DIR) if any(x in d for x in ["model"])]
    for model_path in sorted(model_paths):

        model_number = os.path.basename(model_path)
        # load model meta file
        meta_path = os.path.join(model_path, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)


        model_hf_name = meta['model']
        model_name = REVERSE_MODEL_MAPPING[meta['model']]
        train_data_name = meta['data']
        system = meta['system']
        
        # cases for the ct_english. TO DO; this does not work bc english is the lang not en; change the data prep
        # if len(train_data_name.split("_") )> 1:
        #     train_data_name = train_data_name.split("_")[0]

        prompt_key = train_data_name
        if system:
            prompt_key = prompt_key + "_system"
        PROMPT = PROMPTS[prompt_key]
        
        # select specific models
        # if model_number not in ['model_1']:
        #     continue

        print(f"Running for model {model_number}: MODEL={model_name}, LANGUAGE+{train_data_name}")
        for test_data_name, test_ds in test_data:
            
            # select specific models
            # if test_data_name != "ct_english":
            #     continue
            if "en" != test_data_name:
                continue
                
            print(f"\tEval on {test_data_name}...")
            
            res = eval(model_path, 
                       model_hf_name, 
                       train_data_name,
                       test_data_name,
                       test_ds,
                       PROMPT,
                       system,
                       batch_size=8)

            # ssave
            metrics_path = os.path.join(METRICS_DIR, f"{train_data_name}_2_{test_data_name}_{model_number}.json")
            meta["eval_results"] = res
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)

            print(meta)
            
if __name__ == "__main__":
    main()