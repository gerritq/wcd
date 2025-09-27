import os
import json
import argparse
import random
import evaluate
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
import re
import time
print(os.environ['HF_HOME'])

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, required=True)
parser.add_argument("--n", type=int, default=300)
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--batch_size", type=int, default=8)
args = parser.parse_args()

model_id = {
    "llama_3_70b": "meta-llama/Meta-Llama-3-70B-Instruct",
    "llama_3_8b": "meta-llama/Meta-Llama-3-8B-Instruct",
    "qwen8b": "Qwen/Qwen3-8B",
    "qwen3b": "Qwen/Qwen3-30B-A3B",
    "qwen06b": "Qwen/Qwen3-0.6B",
    "qwen32b": "Qwen/Qwen3-32B"
}.get(args.model)

BASE_DIR = "/scratch/prj/inf_nlg_ai_detection/wcd"
MAX_LENGTH = 128
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PROMPT = """Your task is to determine whether a claim needs a citation (label=1) or no citation (label=0).

Claim: ""{{claim}}""

Output the label in the following format:
{"label": <label>}

Do not add any other text.
# """

# PROMPT = """Do you think the following claim needs a citation (YES) or no citation (NO)?

# Claim: ""{{claim}}""

# Only answer with YES or NO. Not other text.
# """

if args.data not in ["pl_sents","hu_sents","pt_sents","en_sents","cn_fa","cn_fa_ss","cn_fa_ss_nl"]:
    raise ValueError(f"args.data '{args.data}' not available")

set_seed(SEED)
random.seed(SEED)

def load_data():
    if args.data in ["cn_fa","cn_fa_ss","cn_fa_ss_nl"]:
        with open(os.path.join(BASE_DIR, f"data/sets/{args.data}.jsonl"), "r", encoding="utf-8") as f:
            return [json.loads(l) for l in f]
    data = []
    with open(os.path.join(BASE_DIR, f"data/sets/{args.data}.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            obj["label"] = obj.pop("label_2")
            data.append(obj)
    return data

def build_testset():
    data = load_data()
    random.shuffle(data)
    k = args.n // 2
    pos = [x for x in data if x["label"] == 1][:k]
    neg = [x for x in data if x["label"] == 0][:k]
    data = pos + neg
    random.shuffle(data)
    return Dataset.from_list(data)

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