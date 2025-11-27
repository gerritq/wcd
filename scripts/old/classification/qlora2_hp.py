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
import json
import random
import numpy as np
import torch
import argparse
import os
import shutil
import time
import re
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
                    collect_and_save_losses,
                    get_train_dev,
                    get_test,
                    get_tokenizer,
                    evaluation_non_tok
)
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CheckpointConfig
from transformers import DataCollatorForLanguageModeling
from peft import AutoPeftModelForCausalLM

BASE_DIR = os.getenv("BASE_WCD")
DATA_DIR = os.path.join(BASE_DIR, "data/sets/main")
MODEL_DIR = os.path.join(BASE_DIR, "data/models/slm/hp")
# BASE_TEMP_ROOT = os.path.join(BASE_DIR, "scripts/analysis/.slm_hp_trials")
# idea to create a unique tmp dir for each run
RAY_DIR = os.path.join(BASE_DIR, "scripts/classification/.ray_results")
os.makedirs(RAY_DIR, exist_ok=True)
RAY_DIR = tempfile.mkdtemp(dir=RAY_DIR)

ray.init(num_cpus=24,
         num_gpus=1,
         runtime_env={"working_dir": BASE_DIR, # to find the toml
                        "excludes": [".git", "__pycache__", "data", 
                                     "outputs", "models" ,"trainer_output",
                                     ".slm_hp_trials"
                                     ],
        })
set_seed(42)

def get_train_dev_tok(args, tokenizer):
    """custom wo padding which we handle here with the collator, works better"""
    train, dev = get_train_dev(args, tokenizer)
    if args.smoke_test:
        train, dev = train.select(range(128)), dev.select(range(128))
    print(train[0]['text'])
    max_len = tokenizer.model_max_length

    def tokenize_fn(example):
        encoded = tokenizer(
            example["text"],
            truncation=True,
            max_length=tokenizer.model_max_length,
            padding=False,
            return_attention_mask=True,
        )
        return encoded

    cols_to_remove = list(set(train.column_names))
    train_tok = train.map(tokenize_fn, batched=False, remove_columns=cols_to_remove)
    dev_tok   = dev.map(tokenize_fn,   batched=False, remove_columns=cols_to_remove)

    return train_tok, dev_tok

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
        output_dir=None,
        report_to="none",
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_strategy="steps",
        logging_steps=20,
        lr_scheduler_type="linear",
        eval_strategy="epoch",
        save_strategy="epoch",
        warmup_ratio=0.03,
        gradient_accumulation_steps=4, # we keep this alwatys the same
        gradient_checkpointing=False, # set with peft when loading the model
        bf16=torch.cuda.is_bf16_supported(), 
        per_device_eval_batch_size=16,
        per_device_train_batch_size=4, #  
        num_train_epochs=5,
        # save_total_limit=1,
        # overwrite_output_dir=True, 
        )

    tune_config = {
        "learning_rate": tune.loguniform(5e-6, 5e-4),
        # "num_train_epochs": tune.randint(1, 6), # we have max runs set to 5
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

def make_model_init(model_name, bnb_config, lora_config):
    def fn():
        base = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            trust_remote_code=True,
        )
        model = get_peft_model(base, lora_config)
        model.print_trainable_parameters()
        model.gradient_checkpointing_enable()  # PEFT + checkpointing
        # This does the trick to enable check pointing: https://discuss.huggingface.co/t/i-used-to-have-no-problem-with-peft-fine-tuning-after-hundreds-of-trainings-but-now-i-have-encountered-the-error-runtimeerror-element-0-of-tensors-does-not-require-grad-and-does-not-have-a-grad-fn/168829/3
        model.enable_input_require_grads()  
        model.config.use_cache = False         # disable cache
        return model
    return fn

def find_and_save_best_model(best_run, model_dir):
    keep_id = best_run.run_id
    best_folder = None
    # find the folder of the best run
    for root, dirs, _ in os.walk(RAY_DIR):
            for d in dirs:
                if keep_id in d:
                    best_folder = os.path.join(root, d)
                    break
            if best_folder:
                break
    if not best_folder:
        raise FileNotFoundError(f"No folder found for run ID {keep_id}")

    # find the latest checkpoint, we only save the latest so there is only one
    checkpoint_dir = None
    pattern = re.compile(r"checkpoint-\d+")
    for root, dirs, _ in os.walk(best_folder):
        for d in dirs:
            if pattern.fullmatch(d):
                checkpoint_dir = os.path.join(root, d)
                break
        if checkpoint_dir:
            break
    if not checkpoint_dir:
        raise FileNotFoundError(f"No checkpoint found in {best_folder}")

    model = AutoPeftModelForCausalLM.from_pretrained(checkpoint_dir)
    model.save_pretrained(model_dir)

    state_file = os.path.join(checkpoint_dir, "trainer_state.json")
    with open(state_file, "r") as f:
        state = json.load(f)
    return state

def run_hp_search(args):

    num_gpus = torch.cuda.device_count()
    print("Number of GPUs:", num_gpus)

    # Tokeniser and datax
    tokenizer = get_tokenizer(args)
    train_tok, dev_tok = get_train_dev_tok(args, tokenizer)
    bnb_config, lora_config, training_args, tune_config, asha_scheduler = get_config(args, tokenizer)
    print("padding length", tokenizer.model_max_length)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Trainer
    trainer = Trainer(
        model_init=make_model_init(args.model, bnb_config, lora_config),
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=dev_tok,
        data_collator=data_collator
        )

    print("\tHP search started ...")

    # checkpoints:
    # https://docs.ray.io/en/latest/tune/api/doc/ray.tune.CheckpointConfig.html
    best_run = trainer.hyperparameter_search(
        hp_space=lambda _: tune_config, # our tune config
        backend="ray",
        n_trials=args.trials,
        resources_per_trial={"cpu": 8, "gpu": 1},
        scheduler=asha_scheduler,
        direction="minimize",
        compute_objective=lambda m: m["eval_loss"],
        log_to_file=True,
        storage_path=RAY_DIR,
        checkpoint_config=CheckpointConfig(num_to_keep=1, 
                                            checkpoint_score_attribute="eval_loss", 
                                            checkpoint_score_order="min")
    )

    # trainer.save_model(MODEL_DIR)
    print(best_run)
    print(best_run.hyperparameters)
    
    model_number = get_model_number(MODEL_DIR)
    model_dir = os.path.join(MODEL_DIR, f"model_{model_number}")
    meta = find_and_save_best_model(best_run, model_dir)
    meta['model_number'] = f"model_{model_number}"
    
    tokenizer = get_tokenizer(args, inference=True)
    test = get_test(args, tokenizer)
    if args.smoke_test:
        test = test.select(range(64))
    metrics = evaluation_non_tok(test, model_dir, tokenizer, bnb_config)
    meta['test_metrics'] = metrics

    print(metrics)

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

    start = time.time()

    print("="*10, f"Running MODEL {args.model} - LANG {args.lang}","="*10)

    best_run = run_hp_search(args)


    
    print(f"\tHP search done in {(time.time() - start) / 60:.2f} mins")
    print("\nBest run:", best_run)

if __name__ == "__main__":
    main()