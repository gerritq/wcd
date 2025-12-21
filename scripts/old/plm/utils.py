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
from transformers import BitsAndBytesConfig, PreTrainedTokenizerBase

from typing import Dict, List, Callable, Sequence, Tuple, Optional, Any
from torch.utils.data import Dataset
# https://huggingface.co/collections/meta-llama/llama-33-67531d5c405ec5d08a852000
# https://huggingface.co/collections/Qwen/qwen3-67dd247413f0e2e4f653967f

accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

BASE_DIR = os.getenv("BASE_WCD")
DATA_DIR = os.path.join(BASE_DIR, "data/sets/main")

MODEL_MAPPING =  {
    "mBert": "google-bert/bert-base-multilingual-uncased",
    "xlm-r-b": "FacebookAI/xlm-roberta-base",
    "xlm-r-l": "FacebookAI/xlm-roberta-large",
    "mDeberta-b": "microsoft/mdeberta-v3-base",
    "mDeberta-l": "microsoft/deberta-v3-large",
    }

# CONFIGS
BNB_CONFIG = BitsAndBytesConfig(
                            load_in_4bit=True, 
                            bnb_4bit_use_double_quant=True, 
                            bnb_4bit_quant_type="nf4", 
                            bnb_4bit_compute_dtype=torch.bfloat16
                            )

##########################################################################################
# Meta functions
##########################################################################################

def append_meta_file(meta: dict, model_dir: str) -> None:
    """
    Append to the meta file in the model_dir
    """
    meta_path = os.path.join(model_dir, "meta_overview.jsonl")
    with open(meta_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")

def get_model_number(model_dir: str) -> int:
    """
    Identifies the latest meta_<n>.json file in model_dir.
    Returns the next available number.
    """
    meta_files = [f for f in os.listdir(model_dir) if f.startswith("meta_") and f.endswith(".json")]

    numbers = []
    for fname in meta_files:
        num = int(fname.split("_")[1].split(".")[0])
        numbers.append(num)

    return max(numbers) + 1 if numbers else 1

def collect_and_save_losses(history, model_dir):
    """
    Takes the traine.state.history and model_dir.
    Outputs a train/dev loss plot in to model_dir.
    """

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


##########################################################################################
# Data functions
##########################################################################################

def tokenize_data(ds:Dataset, 
                  tokenizer: PreTrainedTokenizerBase,
                  context: bool,
                  max_length=512) -> Dataset:
    SEP = f" {tokenizer.sep_token} "
    def _tok(example):
        if context:
            parts = [
                example.get("section"),
                example.get("previous_sentence"),
                example["claim"],
                example.get("subsequent_sentence"),
            ]
            text = SEP.join(p for p in parts if p).strip()
        else:
            text = example["claim"].strip()
        # print("\n", text)
        enc = tokenizer(
            example['claim'],
            truncation=True,
            max_length=max_length,
            padding=False,      # collator will pad
        )
        enc['labels'] = example['label']
        return enc

    remove_columns = [x for x in ds.column_names]
    ds_tok = ds.map(_tok, batched=False, remove_columns=remove_columns)

    return ds_tok

def get_all_data_sets(path: str, lang: str) -> List[Dataset]:
    """
    Takes a path and language.
    Returns all three datasets.
    """
    ds = load_from_disk(os.path.join(path, lang))
    return ds['train'], ds['dev'], ds['test']


##########################################################################################
# Trainer
##########################################################################################


def compute_metrics(eval_pred):
    """
    Compute metrics function for PLMs for HF trainer.
    
    """
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