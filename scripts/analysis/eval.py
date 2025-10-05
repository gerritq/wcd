
# need to get messages
# do eval by lang


import os, json, re, time, random
import torch, evaluate
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from peft import AutoPeftModelForCausalLM
from collections import defaultdict
from prompts import PROMPT

acc_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

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

def compute_all_metrics(preds, refs, scores=None):
    acc = acc_metric.compute(predictions=preds, references=refs)["accuracy"]
    f1  = f1_metric.compute(predictions=preds, references=refs, average="binary")["f1"]
    return {"accuracy": acc, "f1": f1}

def build_texts_labels(test_ds: Dataset):
    print(test_ds.column_names)
    claims = test_ds["claim"]
    labels = test_ds["label"]
    langs = test_ds["lang"] if "lang" in test_ds.column_names else ["all"] * len(labels)
    
    # build messages
    messages = []
    for c in claims:
        user_msg = PROMPT.replace("{{claim}}", c)
        msg = [
            {"role": "system", "content": "You are a seasoned Wikipedia fact-checker."},
            {"role": "user", "content": user_msg}
        ]
        messages.append(msg)

    return messages, labels, langs

def predict(model, tokenizer, messages, batch_size, is_llama=False):
    preds = []
    for i in range(0, len(messages), batch_size):
        batch = messages[i:i+batch_size]
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

    messages, labels, langs = build_texts_labels(test_ds)
    is_llama = True if "llama" in folder_path else False

    start = time.time()
    preds = predict(model, tokenizer, messages, batch_size, is_llama=is_llama)

    # include lang
    valid = [(p, y, l) for p, y, l in zip(preds, labels, langs) if p is not None]

    if valid:
        p_clean, y_clean, l_clean = zip(*valid)
        overall = compute_all_metrics(list(p_clean), list(y_clean))
    else:
        overall = {"accuracy": 0.0, "f1": 0.0}

    by_lang = {}
    if valid:
        bucket = defaultdict(lambda: {"preds": [], "refs": []})
        for p, y, g in zip(p_clean, y_clean, l_clean):
            bucket[g]["preds"].append(p)
            bucket[g]["refs"].append(y)
        for g, d in bucket.items():
            by_lang[g] = compute_all_metrics(d["preds"], d["refs"])
            
    end = time.time()

    res = {
        "folder": folder_path,
        "n_total": len(labels),
        "n_scored": len(valid),
        "time_min": (end - start) / 60.0,
        "accuracy": overall["accuracy"],
        "f1": overall["f1"],
        "by_lang": by_lang
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