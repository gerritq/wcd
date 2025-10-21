import json
import random
import numpy as np
import torch
import argparse
import os
import shutil
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
from datetime import datetime

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
TEMP_DIR = os.path.join(BASE_DIR, "scripts/analysis/.hp_trials")

# clean all previous runs
if os.path.exists(TEMP_DIR):
    shutil.rmtree(TEMP_DIR)

os.makedirs(TEMP_DIR, exist_ok=True)

MODEL_ID = MODEL_MAPPING[args.model]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

set_seed(42)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

def main():
    start = time.time()

    # load data
    ds = load_from_disk(os.path.join(DATA_DIR, args.lang))
    train_tok = tokenise_data(ds["train"].select(range(10)), tokenizer, args.context)
    dev_tok  = tokenise_data(ds["dev"], tokenizer, args.context)
    # test_tok  = tokenise_data(ds["test"], tokenizer)

    
    print(f"\nRUNNING {args.lang}. Len training data {len(ds['train'])}")

    def model_init():
        return AutoModelForSequenceClassification.from_pretrained(
            MODEL_ID,
            num_labels=2
        )

    training_args = TrainingArguments(
        output_dir=TEMP_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        # metric_for_best_model="accuracy",
        greater_is_better=True,
        report_to="none",
        save_total_limit=1
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
            "learning_rate": trial.suggest_float("learning_rate", 1e-6, 5e-5, log=False),
            "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [8, 16, 32]),
            "num_train_epochs": trial.suggest_int("num_train_epochs", 2, 5),
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



    # Rerun the best model
    # Seems like easiest is to just rerun the best model: https://discuss.huggingface.co/t/how-to-save-the-best-trials-model-using-trainer-hyperparameter-search/8783
    # Took inspiration from here: https://discuss.huggingface.co/t/how-to-save-the-best-trials-model-using-trainer-hyperparameter-search/8783/5
    # trainer.apply_hyperparameters(best_run.hyperparameters, final_model=True)
    
    # trainer.train()
    # best_model_dev_metrics = trainer.evaluate(eval_dataset=dev_tok)
    # best_model_test_metrics = trainer.evaluate(eval_dataset=test_tok)

    end = time.time()

    # Save best model
    run_id = best_run.run_id
    best_model_dir = os.path.join(TEMP_DIR, f"run-{run_id}")
    checkpoints = [d for d in os.listdir(best_model_dir) if d.startswith("checkpoint-")]
    if not checkpoints or (checkpoints and len(checkpoints) > 1):
        raise ValueError(f"No or too many checkpoints found in {best_model_dir}")
    best_model_path = os.path.join(best_model_dir, checkpoints[0])

    final_model = AutoModelForSequenceClassification.from_pretrained(best_model_path)
    
    model_number = get_model_number(MODEL_DIR)
    out_dir = os.path.join(MODEL_DIR, f"model_{model_number}")
    final_model.save_pretrained(out_dir)


    # train_losses, eval_losses = [], []
    # for log in trainer.state.log_history:
    #     if "loss" in log:
    #         train_losses.append({"epoch": log.get("epoch"), "loss": log["loss"]})
    #     if "eval_loss" in log:
    #         eval_losses.append({"epoch": log.get("epoch"), "eval_loss": log["eval_loss"]})

    meta = {
        "model_number": model_number,
        "hp_search": True,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data": args.lang,
        "context": args.context,
        "model": MODEL_ID,
        # "dev_metrics": best_model_dev_metrics,
        # "test_metrics": best_model_test_metrics,
        "train_n": len(ds['train']),
        "dev_n": len(ds['dev']),
        # "test_n": len(ds['test']),
        # "train_losses": train_losses,
        # "eval_losses": eval_losses,
        "time_mins": (end - start) / 60.0,
        "device": DEVICE,
        "cuda_max_memory_allocation_mb": (torch.cuda.max_memory_allocated() / 1024**2) if torch.cuda.is_available() else None,
        "best_run": {
            "objective": best_run.objective,
            "hyperparameters": best_run.hyperparameters,
        },
    }

    # claan also at the end to avoid unnecessary storage
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)

    os.makedirs(TEMP_DIR, exist_ok=True)


    # save and append meta
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    append_meta_file(meta, MODEL_DIR)

if __name__ == "__main__":
    main() 