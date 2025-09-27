#!/usr/bin/env python
import os, json, re, time, random
import torch, evaluate
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from peft import AutoPeftModelForCausalLM

BASE_DIR = "/scratch/prj/inf_nlg_ai_detection/wcd"
FT_DIR   = os.path.join(BASE_DIR, "data/ft")
OUT_DIR  = os.path.join(BASE_DIR, "data/scores")
os.makedirs(OUT_DIR, exist_ok=True)

MAX_LENGTH = 128
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

set_seed(SEED)
random.seed(SEED)

def save_jsonl(records, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_texts_labels(test_ds: Dataset):
    print(test_ds.column_names)
    if "claim" in test_ds.column_names:
        texts = test_ds["claim"]
    elif "text" in test_ds.column_names:
        texts = test_ds["text"]
    elif "sentence" in test_ds.column_names:
        texts = test_ds["sentence"]
    elif "messages" in test_ds.column_names:
        texts = test_ds["messages"]
    else:
        texts = [x["claim"] for x in test_ds]
    labels = test_ds["label"]
    return texts, labels

def predict(model, tokenizer, texts, batch_size, is_llama=False):
    preds = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        chat_texts = []
        for msg in batch:
            chat = tokenizer.apply_chat_template(
                msg,
                tokenize=False,
                add_generation_prompt=True,
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

def eval(folder_path: str, batch_size: int = 8):
    # load dataset
    
    ds = load_from_disk(folder_path)    
    test_ds = ds["test"]

    tokenizer = AutoTokenizer.from_pretrained(folder_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    try:
        model = AutoPeftModelForCausalLM.from_pretrained(
            folder_path, device_map="auto", torch_dtype="auto", trust_remote_code=True
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            folder_path, device_map="auto", torch_dtype="auto", trust_remote_code=True
        )
    model.eval()

    texts, labels = build_texts_labels(test_ds)
    is_llama = True if "llama" in folder_path else False

    start = time.time()
    preds = predict(model, tokenizer, texts, batch_size, is_llama=is_llama)
    valid = [(p, l) for p, l in zip(preds, labels) if p is not None]
    if valid:
        p_clean, y_clean = zip(*valid)
        acc = evaluate.load("accuracy").compute(predictions=p_clean, references=y_clean)["accuracy"]
        f1 = evaluate.load("f1").compute(predictions=p_clean, references=y_clean, average="binary")["f1"]
    else:
        acc, f1 = 0.0, 0.0
    end = time.time()

    res = {
        "folder": folder_path,
        "n_total": len(labels),
        "n_scored": len(valid),
        "accuracy": acc,
        "f1": f1,
        "time_min": (end - start) / 60.0
    }
    return res

def main():
    folders = [os.path.join(FT_DIR, d) for d in os.listdir(FT_DIR) if os.path.isdir(os.path.join(FT_DIR, d))]
    for fld in sorted(folders):
        name = os.path.basename(fld)
        results_path = os.path.join(OUT_DIR, f"{name}.jsonl")

        if os.path.exists(results_path):
            print(f"Skipping {fld} (results already exist at {results_path})")
            continue
        
        print(f"\nRunning {fld}")
        res = eval(fld, batch_size=8)
        if res is not None:
            save_jsonl([res], results_path)
            print(res)

if __name__ == "__main__":
    main()