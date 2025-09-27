import json
import random
import numpy as np
import evaluate
import torch
import wandb
import os
import argparse

from datasets import Dataset, ClassLabel, DatasetDict
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    set_seed, 
    TrainingArguments, 
    Trainer
)

# Settings
wandb.login()
wandb.init(project="wiki-cd")
os.environ["WANDB_PROJECT"]="wiki-cd"

parser = argparse.ArgumentParser()
parser.add_argument("--hp_search", action="store_true")
parser.add_argument("--data", type=str, required=True)
parser.add_argument("--n", type=int, default=6000)
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--notes", type=str, default="-")
args = parser.parse_args()

BASE_DIR="/scratch/prj/inf_nlg_ai_detection/wcd"
MAX_LENGTH = 256
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if args.data not in ["pl_sents","hu_sents", "pt_sents", "en_sents", "cn_fa", "cn_fa_ss", "cn_fa_ss_nl"]:
    raise ValueError(f"args.data '{args.data}' not available")

RUN_NAME=f"wcd_{args.data}"
wandb.log({"N": args.n,
          "data": args.data,
          "notes": args.notes})

print(" ===== RUN ===== ")
print(f" ===== Model: {args.model} ===== ")
print(f" ===== Data: {args.data} ===== ")
print(f" ===== N (total): {args.n} ===== ")
print(f" ===== HP Tuning: {args.hp_search} ===== ")
print(f" ===== Device: {DEVICE} ===== ")
print("")

def load_data():
     

    if args.data == "cn_fa_ss_nl":
        with open(os.path.join(BASE_DIR, "data/sets/cn_fa_ss_nl.jsonl"), "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
        return data

    if args.data == "cn_fa":
        with open(os.path.join(BASE_DIR, "data/sets/cn_fa.jsonl"), "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
        return data

    if args.data == "cn_fa_ss":
        with open(os.path.join(BASE_DIR, "data/sets/cn_fa_ss.jsonl"), "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
        return data
    
    if args.data in ["en_sents", "pt_sents", "hu_sents", "pl_sents"]:
        data=[]
        with open(os.path.join(BASE_DIR, f"data/sets/{args.data}.jsonl"), "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                obj["label"] = obj.pop("label_2")
                data.append(obj)
        return data

def generate_dataset():
    data = load_data()
    random.shuffle(data)

    set_n = args.n//2

    positives = [x for x in data if x['label'] == 1][:set_n]
    negatives = [x for x in data if x['label'] == 0][:set_n]

    # Combine and shuffle again
    data = positives + negatives
    print("N", len(data))
    random.shuffle(data)

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def tokenize(example):
        return tokenizer(
            example["claim"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
        )

    dataset = Dataset.from_list(data)

    split = dataset.train_test_split(test_size=0.2, seed=SEED)
    val_test = split["test"].train_test_split(test_size=0.5, seed=SEED)
    dataset_dict = DatasetDict({
        "train": split["train"],
        "val": val_test["train"],
        "test": val_test["test"]
    })

    dataset = dataset_dict.map(tokenize, batched=True)
    return dataset

def main():
    dataset = generate_dataset()

    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    def model_init():
        return AutoModelForSequenceClassification.from_pretrained(
            args.model, num_labels=2
        )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        acc = accuracy_metric.compute(predictions=predictions, references=labels)
        f1 = f1_metric.compute(predictions=predictions, references=labels, average="binary")

        return {
            "accuracy": acc["accuracy"],
            "f1": f1["f1"]
        }

    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        eval_strategy="epoch",
        report_to="wandb",
        run_name=RUN_NAME,
        )

    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
        compute_metrics=compute_metrics,
    )

    wandb.log({"train_size": len(dataset["train"])})
    
    if args.hp_search:
        def wandb_hp_space(trial):
            return {                
                "learning_rate": {"distribution": "uniform", "min": 1e-6, "max": 1e-4},
                "per_device_train_batch_size": {"values": [16, 32]},
                }

        best_trials = trainer.hyperparameter_search( 
            direction="minimize",
            backend="wandb",
            hp_space=wandb_hp_space,
            n_trials=5
        )

        print(best_trials)
    else:
        trainer.train()

if __name__ == "__main__":
    main()