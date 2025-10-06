import os
import json
import argparse
import random
import torch
import time
import evaluate
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed
)
from utils import MODEL_MAPPING

parser = argparse.ArgumentParser()
parser.add_argument("--lang", type=str, required=True)
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--epochs", type=int, default=5)
args = parser.parse_args()

BASE_DIR = os.getenv("BASE_WCD")
DATA_DIR = os.path.join(BASE_DIR, "data/sets")
OUTPUT_DIR = os.path.join(BASE_DIR, "data/scores")

MAX_LENGTH = 256
SEED = 42

PROMPT = """Your task is to determine whether a claim needs a citation (label=1) or no citation (label=0).

Claim: "{{claim}}"

Output the label in the following format:
{"label": <label>}
"""

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

def build_dataset():
    data = load_data()
    random.shuffle(data)
    k = args.n // 2
    pos = [x for x in data if x["label"] == 1][:k]
    neg = [x for x in data if x["label"] == 0][:k]
    data = pos + neg
    random.shuffle(data)

    samples = []
    for x in data:
        claim = x["claim"] if "claim" in x else x.get("text", "")
        label = x["label"]
        user_msg = PROMPT.replace("{{claim}}", claim)
        messages = [
            {"role": "system", "content": "You are a seasoned Wikipedia fact-checker."},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": json.dumps({"label": label})}
        ]
        samples.append({"messages": messages, "label": label})
    return Dataset.from_list(samples)

def preprocess(dataset, tokenizer):
    def tokenize_fn(example):
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False
        )
        return tokenizer(text, truncation=True, max_length=MAX_LENGTH)
    return dataset.map(tokenize_fn, remove_columns=["messages"])

def save_jsonl(records, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main():
    start = time.time()


    ds = load_from_disk(os.path.join(DATA_DIR, args.lang))


    ds = build_dataset()
    split = ds.train_test_split(test_size=0.1, seed=SEED)
    train_ds, test_ds = split["train"], split["test"]
    
    save_dir = os.path.join(BASE_DIR, f"data/ft/ft_{args.model}_{args.data}")
    os.makedirs(save_dir, exist_ok=True)
    DatasetDict({"train": train_ds, "test": test_ds}).save_to_disk(save_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    train_ds = preprocess(train_ds, tokenizer)
    test_ds = preprocess(test_ds, tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        trust_remote_code=True
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=f"./tmp_{args.model}_{args.data}",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        eval_strategy="epoch",
        save_strategy="no",
        logging_dir="./logs",
        report_to="none",
        fp16=torch.cuda.is_available()
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=data_collator
    )

    trainer.train()

    out_dir = os.path.join(BASE_DIR, f"data/ft/ft_{args.model}_{args.data}")
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

if __name__ == "__main__":
    main()