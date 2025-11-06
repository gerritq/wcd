import os
import random
import evaluate
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import torch
import re
from tqdm import tqdm
from datasets import load_from_disk

# https://huggingface.co/collections/meta-llama/llama-33-67531d5c405ec5d08a852000
# https://huggingface.co/collections/Qwen/qwen3-67dd247413f0e2e4f653967f

accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

BASE_DIR = os.getenv("BASE_WCD")
DATA_DIR = os.path.join(BASE_DIR, "data/sets/main")

##########################################################################################
# Meta functions
##########################################################################################

MODEL_MAPPING =  {
    "mBert": "google-bert/bert-base-multilingual-uncased",
    "mDeberta-b": "microsoft/mdeberta-v3-base",
    "mDeberta-l": "microsoft/deberta-v3-large",
    "llama3_1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama3_3b": "meta-llama/Llama-3.2-3B-Instruct",
    "llama3_8b": "meta-llama/Llama-3.1-8B-Instruct",
    "llama3_70b": "meta-llama/Llama-3.3-70B-Instruct",
    "qwen3_06b": "Qwen/Qwen3-0.6B",
    "qwen3_4b": "Qwen/Qwen3-4B-Instruct-2507",
    "qwen3_8b": "Qwen/Qwen3-8B",
    "qwen3_30b": "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "qwen3_32b": "Qwen/Qwen3-32B",
    "gemma3_12b": "google/gemma-3-12b-it",
    "gpt_oss": "openai/gpt-oss-20b",
    "mistral_8b": "mistralai/Ministral-8B-Instruct-2410",
    "ds_llama": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "aya": "CohereLabs/aya-101",
    "gpt-4o-mini": "openai/gpt-4o-mini",
    "gemini-2.5-flash-lite": "google/gemini-2.5-flash-lite"
    }

def append_meta_file(meta: dict, model_dir: str):
    """appends to the meta file in the model dir; creates the file if it does not exist"""
    meta_path = os.path.join(model_dir, "meta_overview.jsonl")
    with open(meta_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")

def get_model_number(model_dir: str) -> int:
    """identifies the latest model number on the model doir"""
    model_names = [os.path.splitext(d)[0] for d in os.listdir(model_dir) if d.startswith("model_")]
    
    numbers = []
    for name in model_names:
        
        num = int(name.split("_")[1])
        numbers.append(num)
        
    next_number = max(numbers) + 1 if numbers else 1
    return next_number

##########################################################################################
# Code for PLMs
##########################################################################################

def tokenise_data(ds, tokenizer):
    """tokesnier for plms which support context"""
    
    def tokenize(example):
        return tokenizer(
            example['claim'],
            padding="max_length",
            truncation=True,
            max_length=512
        )
    return ds.map(tokenize, batched=False)
    
def get_data(args, tokenizer):
    """retrieve and tokenise data"""
    ds = load_from_disk(os.path.join(DATA_DIR, args.lang))
    train_tok = tokenise_data(ds["train"], tokenizer)
    dev_tok  = tokenise_data(ds["dev"], tokenizer)
    test_tok  = tokenise_data(ds["test"], tokenizer)

    return train_tok, dev_tok, test_tok

def compute_metrics(eval_pred):
    """evaluation function for plms"""
    logits, labels = eval_pred
    probs = np.exp(logits) / np.exp(logits).sum(-1, keepdims=True)  # softmax
    predictions = np.argmax(probs, axis=-1)

    acc = accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="binary")

    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()

    return {
        "accuracy": acc["accuracy"],
        "f1": f1["f1"],
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn)
    }


def collect_and_save_losses(history, model_dir):
    """inputs the trainer history and the model dir
    collects train and eval losses and saves a plot in the model dir"""

    def plot_loss_curves(train_losses, eval_losses, out_dir):

        out_path = os.path.join(out_dir, "loss_plot.pdf")

        train_epochs = [x["epoch"] for x in train_losses]
        train_vals = [x["loss"] for x in train_losses]
        eval_epochs = [x["epoch"] for x in eval_losses]
        eval_vals = [x["eval_loss"] for x in eval_losses]

        plt.figure(figsize=(6, 4))
        plt.plot(train_epochs, train_vals, marker="o", label="Train Loss", color="steelblue")
        plt.plot(eval_epochs, eval_vals, marker="s", label="Eval Loss", color="tomato")

        plt.title("Training and Evaluation Loss over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        
    train_losses, eval_losses = [], []
    for log in history:
        if "loss" in log:
            train_losses.append({"epoch": log.get("epoch"), "loss": log["loss"]})
        if "eval_loss" in log:
            eval_losses.append({"epoch": log.get("epoch"), "eval_loss": log["eval_loss"]})

    if train_losses and eval_losses:
        plot_loss_curves(train_losses, eval_losses, model_dir)



