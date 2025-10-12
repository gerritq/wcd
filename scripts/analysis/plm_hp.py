import json
import random
import numpy as np
import torch
import argparse
import os
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    set_seed,
    TrainingArguments,
    Trainer,
)
from utils import (
                    MODEL_MAPPING, 
                    compute_metrics, 
                    append_meta_file, 
                    get_model_number,
                    tokenise_data
)
import time

parser = argparse.ArgumentParser()
parser.add_argument("--lang", type=str, required=True)
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--trials", type=int, required=True)
parser.add_argument("--context", type=int, required=True)
args = parser.parse_args()
args.context = bool(args.context)

BASE_DIR = os.getenv("BASE_WCD")
DATA_DIR = os.path.join(BASE_DIR, "data/sets")
MODEL_DIR = os.path.join(BASE_DIR, "data/models/plm")

MODEL_ID = MODEL_MAPPING[args.model]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

set_seed(42)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# def tokenise_data(data, tokenizer):
#     sep_token = tokenizer.sep_token or "[SEP]"
#     def format_input(example):
#         # concat input
#         claim = example["claim"].strip()
#         if args.context:
#             section = example['section'].strip()
#             context = example['context'].strip()
            
#             # only use non-empty parts
#             parts = [p for p in [section, context, claim] if p]
#             return f" {sep_token} ".join(parts)
#         # claim only, no context
#         else:
#             return claim

#     def tokenize(example):
#         return tokenizer(
#             format_input(example),
#             padding="max_length",
#             truncation=True,
#         )

#     return data.map(tokenize, batched=False)

def main():
    start = time.time()

    # load data
    ds = load_from_disk(os.path.join(DATA_DIR, args.lang))
    train_tok = tokenise_data(ds["train"], tokenizer, args.context)
    dev_tok  = tokenise_data(ds["dev"], tokenizer, args.context)
    # test_tok  = tokenise_data(ds["test"], tokenizer)

    
    print(f"\nRUNNING {args.lang}. Len training data {len(ds['train'])}")

    def model_init():
        return AutoModelForSequenceClassification.from_pretrained(
            MODEL_ID,
            num_labels=2
        )

    training_args = TrainingArguments(
        output_dir=None,
        eva_strategy="epoch",
        logging_strategy="epoch",
        metric_for_best_model="accuracy",
        greater_is_better=True,
        report_to="none"
        )

    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=dev_tok,
        compute_metrics=compute_metrics,
        )

    # define hp space as in https://huggingface.co/docs/transformers/en/hpo_train
    def hp_space(trial):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True),
            "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [8, 16, 32]),
            "num_train_epochs": trial.suggest_int("num_train_epochs", 2, 8),
            }

    def compute_objective(metrics):
        return metrics["eval_accuracy"]

    print(f"\tBegin HP search for {args.trials} trials...")
    best_run = trainer.hyperparameter_search(
        n_trials=args.trials,
        direction="maximize",
        hp_space=hp_space,
        compute_objective=compute_objective,
        backend="optuna",
    )
    print(f"\tHP search done in {(time.time() - start) / 60:.2f} mins")
    print("\nBest run:", best_run)

    collect_trials = [r._asdict() for r in best_run.trials]  

    print("\nCollect trials:", collect_trials)

    # Rerun the best model
    # Seems like easiest is to just rerun the best model: https://discuss.huggingface.co/t/how-to-save-the-best-trials-model-using-trainer-hyperparameter-search/8783
    # Took inspiration from here: https://discuss.huggingface.co/t/how-to-save-the-best-trials-model-using-trainer-hyperparameter-search/8783/5
    trainer.apply_hyperparameters(best_run.hyperparameters, final_model=True)
    trainer.train()
    best_model_dev_metrics = trainer.evaluate(eval_dataset=dev_tok)
    # best_model_test_metrics = trainer.evaluate(eval_dataset=test_tok)

    # Save best mmodel
    model_number = get_model_number(MODEL_DIR)
    out_dir = os.path.join(MODEL_DIR, f"model_{model_number}")
    trainer.save_model(out_dir)

    end = time.time()

    train_losses, eval_losses = [], []
    for log in trainer.state.log_history:
        if "loss" in log:
            train_losses.append({"epoch": log.get("epoch"), "loss": log["loss"]})
        if "eval_loss" in log:
            eval_losses.append({"epoch": log.get("epoch"), "eval_loss": log["eval_loss"]})

    meta = {
        "model_number": model_number,
        "data": args.lang,
        "context": args.context,
        "model": MODEL_ID,
        "dev_metrics": best_model_dev_metrics,
        # "test_metrics": best_model_test_metrics,
        "train_n": len(ds['train']),
        "dev_n": len(ds['dev']),
        # "test_n": len(ds['test']),
        "train_losses": train_losses,
        "eval_losses": eval_losses,
        "time_mins": (end - start) / 60.0,
        "device": DEVICE,
        "cuda_max_memory_allocation_mb": (torch.cuda.max_memory_allocated() / 1024**2) if torch.cuda.is_available() else None,
        "best_run": {
            "objective": best_run.objective,
            "hyperparameters": best_run.hyperparameters,
        },
        "all_hp_trials": collect_trials 
    }

    # save meta
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()