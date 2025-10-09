import os
import json
import argparse
import random
import evaluate
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
import re
from utils import MODEL_MAPPING
from prompts import PROMPT
import time
print(os.environ['HF_HOME'])

parser = argparse.ArgumentParser()
parser.add_argument("--lang", type=str, required=True)
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--epochs", type=int, default=5)
args = parser.parse_args()

BASE_DIR = os.getenv("BASE_WCD")
DATA_DIR = os.path.join(BASE_DIR, "data/sets")
OUTPUT_DIR = os.path.join(BASE_DIR, "data/scores")

MODEL_ID = MODEL_MAPPING[args.model]
set_seed(SEED)
random.seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

set_seed(SEED)
random.seed(SEED)

def format_prompt(text: str) -> str:
    return PROMPT.replace("{{claim}}", text)

def predict(model, tokenizer, texts, batch_size):
    preds = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        # Build chat messages and apply chat template per item
        chat_texts = []
        for t in batch:
            msg = [{"role": "system", "content": "You are a seasoned Wikipedia fact-checker."},
                   {"role": "user", "content": format_prompt(t)}]
            chat = tokenizer.apply_chat_template(
                msg,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            chat_texts.append(chat)


        print(f"\nChats:")
        for i, x in enumerate(chat_texts):
            print('\nChat', i, x, "===\n")

        enc = tokenizer(
            chat_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH
        ).to(model.device)

        # model specifics
        terminators = None
        if "llama" in args.model:
            terminators = [tokenizer.eos_token_id]

        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=terminators,
            )

        print("=== GENERATIONS ===")

        for j in range(len(batch)):
            output_ids = out[j][len(enc["input_ids"][j]):].tolist() 
            try:
                # rindex finding 151668 (</think>)
                index = len(output_ids) - output_ids[::-1].index(151668)
                print("Found index")
            except ValueError:
                print('did not find index')
                index = 0
            full_prompt = tokenizer.decode(out[j], skip_special_tokens=True).strip("\n")
            prompt = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
            content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

            print("======================================")
            print("======================================")
            print(f"Batch item {j}")
            print("Entire prompt:", full_prompt)
            print("Content:", content)
            print(output_ids)
            print("======================================")
            print("======================================")

            label = None
            # Numeric
            try:
                obj = json.loads(content)
                label = int(obj.get("label", None))
            except Exception:
                m = re.search(r'"label"\s*:\s*(\d)', content)
                if m:
                    label = int(m.group(1))
            if label not in (0, 1):
                lalabelb = None

            # yes/no
            # label = content.strip().lower()
            # if label in ['yes', 'no']:
            #     label = 1 if label == 'yes' else 0
            # else:
            #     label = None
            preds.append(label)
    return preds

def evaluate_set(model, tokenizer, ds):
    if "claim" in ds.column_names:
        texts = ds["claim"]
    elif "text" in ds.column_names:
        texts = ds["text"]
    elif "sentence" in ds.column_names:
        texts = ds["sentence"]
    else:
        texts = [x["claim"] for x in ds]
    labels = ds["label"]

    preds = predict(model, tokenizer, texts, args.batch_size)

    valid = [(p, l) for p, l in zip(preds, labels) if p is not None]
    if valid:
        preds_clean, labels_clean = zip(*valid)
        acc = evaluate.load("accuracy").compute(predictions=preds_clean, references=labels_clean)["accuracy"]
        f1 = evaluate.load("f1").compute(predictions=preds_clean, references=labels_clean, average="binary")["f1"]
    else:
        acc, f1 = 0.0, 0.0

    return acc, f1, len(valid), len(labels)

def save_jsonl(records, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main():

    start = time.time()
    ds_test = build_testset()

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16 if DEVICE == "cuda" else None,
        trust_remote_code=True
    )
    model.eval()

    print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")

    acc, f1, n_valid, n_total = evaluate_set(model, tokenizer, ds_test)

    end = time.time()
    result = {
        "data": args.data,
        "model": model_id,
        "n_requested": args.n,
        "n_total": n_total,
        "n_scored": n_valid,
        "accuracy": acc,
        "f1": f1,
        "time": (end-start) / 60,
        "cuda_max_memory_allocation": torch.cuda.max_memory_allocated()
    }

    out_path = os.path.join(BASE_DIR, f"data/scores/{args.data}_{args.model}.jsonl")
    save_jsonl([result], out_path)

    print(result)


if __name__ == "__main__":
    main()