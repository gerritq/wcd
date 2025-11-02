# https://docs.ray.io/en/latest/tune/examples/pbt_transformers.html
# Trial chedulers (early terminate bad trials, and optimise others)
    # https://docs.ray.io/en/latest/tune/api/schedulers.html
# Search algorithms
    # https://docs.ray.io/en/latest/tune/api/suggestion.html#tune-search-alg
# Search space definitions
    # https://docs.ray.io/en/latest/tune/api/search_space.html
# HP Search techniques: https://www.deepchecks.com/hyperparameter-optimization-llms-best-practices-advanced-techniques/
# Trainer HP search method\
    # https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/trainer#transformers.Trainer.hyperparameter_search



"""
Questions:
1. Should we stop earlier? Based on a step level rather or always wait for a full epoch?
- We have seen really bad runs so steps may make more sense. How to do this?

2. Maybe implement the inference at the end and safe a meta file? For direct access to acc.



"""
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
    AutoModelForCausalLM,
    set_seed,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model
from utils import (
                    MODEL_MAPPING, 
                    append_meta_file, 
                    get_model_number,
                    tokenise_data,
                    plot_loss_curves
)
from qlora2 import  (preprocess_function,
                     preprocess_function_generation,
                     get_dataset,
                     get_testset,
                     collect_and_save_losses,
                     get_tokenizer)

from ray import tune
from ray.tune.schedulers import ASHAScheduler

set_seed(42)

BASE_DIR = os.getenv("BASE_WCD")
DATA_DIR = os.path.join(BASE_DIR, "data/sets/main")
MODEL_DIR = os.path.join(BASE_DIR, "data/models/slm/hp")
# BASE_TEMP_ROOT = os.path.join(BASE_DIR, "scripts/analysis/.slm_hp_trials")
TEMP_DIR = os.path.join(BASE_DIR, "scripts/classification/.slm_hp_trials")
os.makedirs(TEMP_DIR, exist_ok=True)
# os.makedirs(BASE_TEMP_ROOT, exist_ok=True)
# TEMP_DIR = tempfile.mkdtemp(prefix="run_", dir=BASE_TEMP_ROOT)

def get_config(args, tokenizer):
    bnb_config = BitsAndBytesConfig(
                                load_in_4bit=True, 
                                bnb_4bit_use_double_quant=True, 
                                bnb_4bit_quant_type="nf4", 
                                bnb_4bit_compute_dtype=torch.bfloat16
                                )
    lora_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=.05,
            r=16,
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
    )

    training_args = TrainingArguments(
        output_dir=TEMP_DIR,
        report_to="none",
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_strategy="steps",
        logging_steps=20,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03, 
        load_best_model_at_end=True,
        per_device_eval_batch_size=16,
        evaluation_strategy="epoch",
        # save_total_limit=1,
        # overwrite_output_dir=True, 
        )

    tune_config = {
        "learning_rate": tune.loguniform(5e-6, 5e-4),
        # "num_train_epochs": tune.randint(1, 6),
        "per_device_train_batch_size": tune.choice([4, 8]),
        "max_grad_norm": tune.uniform(0.1, 2.0), # defaults to 1.0
        "weight_decay": tune.loguniform(1e-6, 1e-1),
    }

    asha_scheduler = ASHAScheduler(
        time_attr='training_iteration',
        metric='eval_loss',
        mode='min',
        max_t=6, 
        grace_period=1, # one eval = eval_steps
        reduction_factor=3,
        brackets=1,
    )

    return bnb_config, lora_config, training_args, tune_config, asha_scheduler


def make_model_init(model: str, bnb_config, lora_config):
    model = AutoPeftModelForCausalLM.from_pretrained(
        model,
        num_labels=2,
        quantization_config=bnb_config
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model

def run_hp_search(args):

    num_gpus = torch.cuda.device_count()
    print("Number of GPUs:", num_gpus)

    # Tokeniser and data
    tokenizer = get_tokenizer(args)
    train, dev = get_dataset(args, tokenizer)
    bnb_config, lora_config, training_args, tune_config, asha_scheduler = get_config(args, tokenizer)
    print(train[0]['text'])

    # Trainer
    trainer = Trainer(
        model_init=make_model_init(args.model, bnb_config, lora_config),
        args=training_args,
        train_dataset=train,
        eval_dataset=dev,
        processing_class=tokenizer,
        )

    print("\tHP search started ...")

    best_run = trainer.hyperparameter_search(
        hp_space=lambda _: tune_config, # our tune config
        backend="ray",
        n_trials=args.trials,
        resources_per_trial={"cpu": 8, "gpu": 1},
        scheduler=asha_scheduler,
        checkpoint_score_attr="training_iteration",
        direction="minimize",
        compute_objective=lambda m: m["eval_loss"],
        log_to_file=True,
        local_dir=os.path.join(TEMP_DIR, "ray_results/"),
        keep_checkpoints_num=1, # keep only one checkpoint
    )

    trainer.save_model(MODEL_DIR)
    return best_run

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--trials", type=int, required=True)
    args = parser.parse_args()

    start = time.time()

    print("="*10, f"Running MODEL {args.model} - LANG {args.lang}","="*10)

    best_run = run_hp_search(args)

    print(f"\tHP search done in {(time.time() - start) / 60:.2f} mins")
    print("\nBest run:", best_run)