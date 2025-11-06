import json
import random
import numpy as np
import torch
import argparse
import os
import shutil
import time
from datetime import datetime
import tempfile
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
                    get_data,
                    collect_and_save_losses
)



BASE_DIR = os.getenv("BASE_WCD")
DATA_DIR = os.path.join(BASE_DIR, "data/sets/main")
MODEL_DIR = os.path.join(BASE_DIR, "data/models/plm/hp")
TEMP_DIR = os.path.join(BASE_DIR, "data/models/plm/hp/.hp_random")
os.makedirs(TEMP_DIR, exist_ok=True)
TEMP_DIR = tempfile.mkdtemp(dir=TEMP_DIR)

set_seed(42)


def get_configs(args):
    training_args = TrainingArguments(
        output_dir=TEMP_DIR,
        report_to="none",
        save_strategy="epoch",
        greater_is_better=False,
        metric_for_best_model="eval_loss", 
        save_total_limit=1,
        logging_strategy="steps",
        logging_steps=20,
        eval_strategy="epoch",
        load_best_model_at_end=True,
        )

    def model_init():
        return AutoModelForSequenceClassification.from_pretrained(
            args.model,
            num_labels=2
        )

    def hp_space(trial):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-6, 5e-5, log=True),
            "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [8, 16]),
            "num_train_epochs": trial.suggest_int("num_train_epochs", 1, 2),
            }
    return training_args, model_init, hp_space


def find_and_save_model(best_run, model_dir):
    run_dir = os.path.join(TEMP_DIR, f"run-{best_run.run_id}")
    subdirs = [os.path.join(run_dir, d) for d in os.listdir(run_dir) if os.path.isdir(os.path.join(run_dir, d))]
    if len(subdirs) != 1:
        raise ValueError(f"Expected exactly one folder in {run_dir}, found {len(subdirs)}.")
    model = AutoModelForSequenceClassification.from_pretrained(subdirs[0])
    model.save_pretrained(model_dir)
    
    state_file = os.path.join(subdirs[0], "trainer_state.json")
    with open(state_file, "r") as f:
        state = json.load(f)
    return state

def run(args):
    
    print(f"\n========== RUNNING {args.lang} TRIALS {args.trials}==========\n")

    start = time.time()
    
    # load data
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    train_tok, dev_tok, test_tok = get_data(args, tokenizer)
    if args.smoke_test:
        train_tok, dev_tok, test_tok = train_tok.select(range(64)), dev_tok.select(range(64)), test_tok.select(range(64))
    training_args, model_init, hp_space = get_configs(args)

    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=dev_tok,
        compute_metrics=compute_metrics,

        )

    print(f"\tBegin HP search for {args.trials} trials...")

    best_run = trainer.hyperparameter_search(
        hp_space=hp_space,
        n_trials=args.trials,
        direction="minimize",
        compute_objective=lambda m: m["eval_loss"],
        backend="optuna",
    )
    
    print(f"\tHP search done in {(time.time() - start) / 60:.2f} mins")
    print("\nBest run:", best_run)
    end = time.time()
    
    # Save the best model
    # we do what suggested here: https://discuss.huggingface.co/t/how-to-save-the-best-trials-model-using-trainer-hyperparameter-search/8783

    # Get the best model
    model_number = get_model_number(MODEL_DIR)
    model_dir = os.path.join(MODEL_DIR, f"model_{model_number}")
    meta = find_and_save_model(best_run, model_dir)

    meta['model_number'] = model_number
    meta['date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    meta['trials'] = args.trials
    meta['time_mins'] = (end - start) / 60.0
    meta['cuda_max_memory_allocation_mb'] = (torch.cuda.max_memory_allocated() / 1024**2) if torch.cuda.is_available() else None
    
    # eval 
    final_model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    training_args = TrainingArguments(
            output_dir=None,
            report_to="none"
        )
    trainer = Trainer(model=final_model, 
                      args=training_args,
                      compute_metrics=compute_metrics)
    
    # test set
    dev_results = trainer.evaluate(dev_tok)
    test_results = trainer.evaluate(test_tok)

    meta['dev_metrics']  = dev_results
    meta['test_metrics']  = test_results
    
    # save and append meta
    append_meta_file(meta, MODEL_DIR)
    with open(os.path.join(model_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--trials", type=int, required=True)
    parser.add_argument("--smoke_test", type=int, required=True)
    args = parser.parse_args()
    args.model = MODEL_MAPPING[args.model]
    args.smoke_test = bool(args.smoke_test)
    if args.smoke_test:
        args.trials=1

    run(args)

if __name__ == "__main__":
    main() 