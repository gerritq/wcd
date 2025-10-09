import json
import random
import numpy as np
import evaluate
import torch
import argparse
import os
from datasets import load_from_disk
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    set_seed, 
    TrainingArguments, 
    Trainer
)
from utils import MODEL_MAPPING
import time

parser = argparse.ArgumentParser()
parser.add_argument("--lang", type=str, required=True)
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--epochs", type=int, default=5)
args = parser.parse_args()

BASE_DIR = os.getenv("BASE_WCD")
DATA_DIR = os.path.join(BASE_DIR, "data/sets")
MODEL_DIR = os.path.join(BASE_DIR, "data/models")
METRICS_DIR = os.path.join(BASE_DIR, "data/metrics")

MODEL_ID = MODEL_MAPPING[args.model]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

set_seed(42)
random.seed(42)

accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

def tokenise_data(data, tokenizer):
    

    def tokenize(example):
        return tokenizer(
            example["claim"],
            padding="max_length",
            truncation=True,
        )
    dataset = data.map(tokenize, batched=True)
    return dataset

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    acc = accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="binary")

    return {
        "accuracy": acc["accuracy"],
        "f1": f1["f1"]
        }

def main():
    start = time.time()

    ds = load_from_disk(os.path.join(DATA_DIR, args.lang))
    train_tok = tokenise_data(ds["train"], tokenizer)
    test_tok = tokenise_data(ds["test"], tokenizer)

    print(f"\nRUNNING {args.lang}. Len data {len(ds['train']) + len(ds['test'])}")

    def model_init():
        return AutoModelForSequenceClassification.from_pretrained(
            MODEL_ID, 
            num_labels=2
        )

    training_args = TrainingArguments(
        output_dir=None,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        eval_strategy="epoch",
        logging_strategy="epoch",
        report_to="none"
        )

    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=test_tok,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # eval on test
    metrics = trainer.evaluate(eval_dataset=test_tok)
    
    # collect losses
    train_losses = []
    eval_losses = []
    
    print(trainer.state.log_history)

    for log in trainer.state.log_history:
        if "loss" in log:
            train_losses.append({"epoch": log.get("epoch"), "loss": log["loss"]})
        if "eval_loss" in log:
            eval_losses.append({"epoch": log.get("epoch"), "eval_loss": log["eval_loss"]})

    end = time.time()
    
    meta = {
        "data": args.lang,
        "model": MODEL_ID,
        "train_n": len(ds['train']),
        "test_n": len(ds['test']),
        "epochs": args.epochs,
        "batch_size": training_args.per_device_train_batch_size,
        "learning_rate": training_args.learning_rate,
        "train_losses": train_losses,
        "eval_losses": eval_losses,
        "time_mins": (end - start) / 60.0,
        "cuda_max_memory_allocation": torch.cuda.max_memory_allocated() / 1024**2
    }

    # save model
    model_dir = os.path.join(MODEL_DIR, f"{args.lang}_{args.model}")
    trainer.model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    with open(os.path.join(model_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # save scores
    with open(os.path.join(METRICS_DIR, f"{args.lang}_{args.model}.json"), "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()