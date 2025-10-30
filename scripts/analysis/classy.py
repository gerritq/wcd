# https://huggingface.co/docs/transformers/model_doc/llama#transformers.LlamaForSequenceClassification
# https://blog.redsift.com/ai/text-classification-in-the-age-of-llms/
# this is how the classification head is implemented
# based on the last hidden state with a simple linear layer, no dropout
    # https://github.com/huggingface/transformers/blob/b47b35637f5c0c0a6f4b7563072a36c083fb4159/src/transformers/modeling_layers.py#L98
# this is instead how the classification head is implemented with bert 
    # https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/bert/modeling_bert.py#L1444
# https://github.com/huggingface/transformers/issues/1001
    # similar to this post, we have just passed a new classifier (in our case score) model to the model and made sure it is on the same device 
import os
import sys
import json
import argparse
import time
import re
from datetime import datetime
from tqdm import tqdm
from utils import (
                    MODEL_MAPPING, 
                    append_meta_file, 
                    get_model_number,
                    plot_loss_curves
)
from prompts import SYSTEM_PROMPTS_SLM

import torch
import torch.nn as nn
import evaluate
from datasets import load_from_disk, concatenate_datasets
from peft import LoraConfig, get_peft_model
from sklearn.metrics import confusion_matrix
from transformers import (AutoTokenizer, 
                          AutoModelForSequenceClassification, 
                          set_seed, 
                          BitsAndBytesConfig, 
                          Trainer, 
                          TrainingArguments)

acc_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

set_seed(42)

BASE_DIR = os.getenv("BASE_WCD") 
DATA_DIR = os.path.join(BASE_DIR, "data/sets/main")
MODEL_DIR = os.path.join(BASE_DIR, "data/models/slm/test")


class CustomClassificationHead(nn.Module):
    """this is our custom classification head
    this should work in full precision and not be touched by lora"""
    def __init__(self, hidden_size, num_labels):
        super().__init__()

        if hidden_size != 1024:
            intermediate1 = hidden_size // 2
            intermediate2 = intermediate1 // 2
            self.net = nn.Sequential(
                nn.Linear(hidden_size, intermediate1),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(intermediate1, intermediate2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(intermediate2, num_labels, bias=False)
            )
        else:
            intermediate_size = hidden_size // 2
            self.net = nn.Sequential(
                nn.Linear(hidden_size, intermediate_size),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(intermediate_size, num_labels, bias=False)
            )

    def forward(self, x):
        return self.net(x)

def get_dataset(args, tokenizer):
    """default implementation to get the data"""
    def process(example):
        tok = tokenizer(example["claim"], truncation=True, padding="max_length", max_length=1024)
        tok["labels"] = example["label"]
        return tok

    ds = load_from_disk(os.path.join(DATA_DIR, args.lang))
    
    keep = ['claim', 'label']
    train = ds['train'].remove_columns([c for c in ds["train"].column_names if c not in keep])
    dev = ds['dev'].remove_columns([c for c in ds["dev"].column_names if c not in keep])
    test = ds['test'].remove_columns([c for c in ds["test"].column_names if c not in keep])

    # this is for testing
    # train = train.select(range(64))
    # dev = dev.select(range(64))
    # test = test.select(range(64))

    tok_train = train.map(process, remove_columns=train.column_names)
    tok_dev   = dev.map(process,   remove_columns=dev.column_names)
    tok_test   = test.map(process,   remove_columns=test.column_names)

    return tok_train, tok_dev, tok_test

def get_dataset_small(args, tokenizer):
    """reduces the training data size but keeps it balanced"""

    def take_n_per_label(ds, n=500):
        ds = ds.shuffle(seed=42)
        pos = ds.filter(lambda x: x['label'] == 1)
        neg = ds.filter(lambda x: x['label'] == 0)
        pos = pos.select(range(n))
        neg = neg.select(range(n))
        return concatenate_datasets([pos, neg]).shuffle(seed=42)

    def process(example):
        tok = tokenizer(example["claim"], truncation=True, padding="max_length", max_length=1024)
        tok["labels"] = example["label"]
        return tok

    ds = load_from_disk(os.path.join(DATA_DIR, args.lang))
    
    keep = ['claim', 'label']
    train = ds['train'].remove_columns([c for c in ds["train"].column_names if c not in keep])
    dev = ds['dev'].remove_columns([c for c in ds["dev"].column_names if c not in keep])
    test = ds['test'].remove_columns([c for c in ds["test"].column_names if c not in keep])

    train = take_n_per_label(train, n=500)

    print("\tReduced training data size", len(train))

    tok_train = train.map(process, remove_columns=train.column_names)
    tok_dev   = dev.map(process,   remove_columns=dev.column_names)
    tok_test   = test.map(process,   remove_columns=test.column_names)

    return tok_train, tok_dev, tok_test


def collect_and_save_losses(history, model_dir):
    train_losses, eval_losses = [], []
    for log in history:
        if "loss" in log:
            train_losses.append({"epoch": log.get("epoch"), "loss": log["loss"]})
        if "eval_loss" in log:
            eval_losses.append({"epoch": log.get("epoch"), "eval_loss": log["eval_loss"]})

    if train_losses and eval_losses:
        plot_loss_curves(train_losses, eval_losses, model_dir)

def get_config(args, tokenizer):
    bnb_config = BitsAndBytesConfig(
                                load_in_4bit=True, 
                                bnb_4bit_use_double_quant=True, 
                                bnb_4bit_quant_type="nf4", 
                                bnb_4bit_compute_dtype=torch.bfloat16
                                )

    # LoRA config based on QLoRA paper & Sebastian Raschka experiment
    # https://www.philschmid.de/fine-tune-google-gemma
    # We actuallu use those: https://docs.unsloth.ai/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide
    lora_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=.05,
            r=16, # rec by unsloth to set this to r or r*2
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type = 'SEQ_CLS',
            modules_to_save=["score"] # this is the classification head!
    )
    # https://www.philschmid.de/fine-tune-google-gemma
    training_args = TrainingArguments(
        output_dir=None,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_acc, # effective batch size = batch_size * gradient_accumulation_steps
        gradient_checkpointing=True,            # computes gradient on the fly, does not keep them in memory for a forward pass
        bf16=True,
        learning_rate=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=0.03,                      # warmup ratio based on QLoRA paper
        lr_scheduler_type="linear", # qlora paper uses linear
        weight_decay=0.01, # rec by unsloth
        report_to='none',
        logging_strategy="steps",
        logging_steps=20,
        # model_init_kwargs={"quantization_config": bnb_config},
        eval_strategy="steps",
        eval_steps=60,
        per_device_eval_batch_size=16,
        )
    return training_args, bnb_config, lora_config


def get_tokenizer(args, inference=False):
    if inference:
        tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side='left')
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    # if tokenizer has no padding token, then reuse the end of sequence token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.model_max_length > 100_000:
        tokenizer.model_max_length = 512*10 
    if not tokenizer.chat_template:
        raise Exception("Tokeniser has not cha template.")

    return tokenizer

def get_model(model_name, quantization_config, lora_config, tokenizer):
    
    # model = CustomGenericForSequenceClassification.from_pretrained(
    #    model_name,
    #     quantization_config=quantization_config,
    #     num_labels=2
    # )    
    

    model = AutoModelForSequenceClassification.from_pretrained(
       model_name,
        quantization_config=quantization_config,
        num_labels=2
    )

    hidden_size = model.config.hidden_size
    num_labels = model.config.num_labels
    
    model.score = CustomClassificationHead(hidden_size, num_labels)

    model = get_peft_model(model, lora_config)
    if not model.config.pad_token_id:
        model.config.pad_token_id = tokenizer.pad_token_id

    model.to(device="cuda", dtype=torch.bfloat16)
    return model

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)

    acc = acc_metric.compute(predictions=preds, references=labels)["accuracy"]
    f1 = f1_metric.compute(predictions=preds, references=labels, average="binary")["f1"]

    return {"accuracy": acc, "f1": f1}

def set_to_list(obj):
    """stupid function to convert sets to list for json saving"""
    for key, value in obj.items():
        if isinstance(value, set):
            obj[key] = list(value)
    return obj

def main():
    start = time.time()

    # ARGPARSE
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--grad_acc", type=int, required=True)
    parser.add_argument("--max_grad_norm", type=float, required=True)
    parser.add_argument("--plw", type=int, default=0)
    args = parser.parse_args()
    args.plw = bool(args.plw)
    args.model = MODEL_MAPPING[args.model]

    print("="*10, f"Running MODEL","="*10)
    print(args, "\n\n")

    tokenizer = get_tokenizer(args)
    training_args, bnb_config, lora_config = get_config(args, tokenizer)
    model = get_model(args.model, bnb_config, lora_config, tokenizer)
    train_tok, dev_tok, test_tok = get_dataset(args, tokenizer)
    # train_tok, dev_tok, test_tok = get_dataset_small(args, tokenizer)
        
    print(model)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=dev_tok,
        compute_metrics=compute_metrics
        )
    
    print(f"\tMax memory allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
    
    train_result = trainer.train()
    
    dev_metrics = trainer.evaluate(dev_tok)
    test_metrics = trainer.evaluate(test_tok)

    print("\tTest metrics", test_metrics)
    
    model_number = get_model_number(MODEL_DIR)
    model_dir = os.path.join(MODEL_DIR, f"model_{model_number}")
    os.makedirs(model_dir, exist_ok=True)  

    meta = {
        "model_number": model_number,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data": args.lang,
        "model": args.model,
        "train_n": len(train_tok),
        "dev_n": len(dev_tok),
        "test_n": len(test_tok),
        "training_args": training_args.to_dict(),
        "lora_args": set_to_list(lora_config.to_dict()),
        "time_min": (time.time() - start) / 60.0,
        "cuda_max_memory_allocation": torch.cuda.max_memory_allocated() / 1024**2,
        "dev_metrics": dev_metrics,
        "test_metrics": test_metrics
    }

    collect_and_save_losses(trainer.state.log_history, model_dir)
    append_meta_file(meta, MODEL_DIR)
    with open(os.path.join(model_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()