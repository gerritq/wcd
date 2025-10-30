import os
import random
import evaluate
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# https://huggingface.co/collections/meta-llama/llama-33-67531d5c405ec5d08a852000
# https://huggingface.co/collections/Qwen/qwen3-67dd247413f0e2e4f653967f

accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")
roc_auc_metric = evaluate.load("roc_auc")

MODEL_MAPPING =  {
    "mBert": "google-bert/bert-base-multilingual-uncased",
    "xlm-r-b": "FacebookAI/xlm-roberta-base",
    "xlm-r-l": "FacebookAI/xlm-roberta-large",
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

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = np.exp(logits) / np.exp(logits).sum(-1, keepdims=True)  # softmax
    predictions = np.argmax(probs, axis=-1)

    acc = accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="binary")
    auc = roc_auc_metric.compute(prediction_scores=probs[:, 1], references=labels)

    return {
        "accuracy": acc["accuracy"],
        "f1": f1["f1"],
        "auroc": auc["roc_auc"]
    }

def compute_metrics(eval_pred):
    """evaluation function for plms"""
    logits, labels = eval_pred
    probs = np.exp(logits) / np.exp(logits).sum(-1, keepdims=True)  # softmax
    predictions = np.argmax(probs, axis=-1)

    acc = accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="binary")
    auc = roc_auc_metric.compute(prediction_scores=probs[:, 1], references=labels)

    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()

    return {
        "accuracy": acc["accuracy"],
        "f1": f1["f1"],
        "auroc": auc["roc_auc"],
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn)
    }

def append_meta_file(meta: dict, model_dir: str):
    meta_path = os.path.join(model_dir, "meta_overview.jsonl")
    with open(meta_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")

def get_model_number(model_dir: str) -> int:
    model_names = [os.path.splitext(d)[0] for d in os.listdir(model_dir) if d.startswith("model_")]
    
    numbers = []
    for name in model_names:
        
        num = int(name.split("_")[1])
        numbers.append(num)
        
    next_number = max(numbers) + 1 if numbers else 1
    return next_number

def tokenise_data(ds, tokenizer, context_bool: bool):
    sep_token = tokenizer.sep_token or "[SEP]"
    def format_input(example):
        # concat input
        claim = example["claim"].strip()
        if context_bool:
            section = example['section']
            context = example['context']
            
            # only use non-empty parts
            parts = [p.strip() for p in [section, context, claim] if p]
            out = f" {sep_token} ".join(parts)
            print(out)
            return out
        # claim only, no context
        else:
            return claim

    def tokenize(example):
        return tokenizer(
            format_input(example),
            padding="max_length",
            truncation=True,
            max_length=512
        )

    return ds.map(tokenize, batched=False)


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