# inspired by: https://www.analyticsvidhya.com/blog/2024/06/finetuning-llama-3-for-sequence-classification/

import os
import sys
import json
import argparse
import random
import torch
import time
from utils import (
                    MODEL_MAPPING, 
                    append_meta_file, 
                    get_model_number
)
from datasets import load_from_disk
from typing import List, Dict

from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    AutoTokenizer, 
    AutoModelForCausalLM, 
    set_seed, 
    BitsAndBytesConfig,
    DataCollatorWithPadding
)
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score

# HF recommends nf4
# 
# bfloat speeds up training
# load in 4bit reducses storage size
# HF recommends nf4
# Use Qlora effectively as in https://ai.google.dev/gemma/docs/core/huggingface_text_finetune_qlora
bnb_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_use_double_quant=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_compute_dtype=torch.bfloat16
)

# ARGPARSE
parser = argparse.ArgumentParser()
parser.add_argument("--lang", type=str, required=True)
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--epochs", type=int, default=5)
args = parser.parse_args()

# DIRs
BASE_DIR = os.getenv("BASE_WCD")
DATA_DIR = os.path.join(BASE_DIR, "data/sets")
MODEL_DIR = os.path.join(BASE_DIR, "data/models/plm")
SHOTS_DIR = os.path.join(BASE_DIR, "data/sents/shots")

# VARs
MODEL_ID = MODEL_MAPPING[args.model]

set_seed(42)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# TOKENISER
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
tokenizer.truncation_side = "right"

MAX_LENGTH = tokenizer.model_max_length
print("Max context length:", MAX_LENGTH)

def build_classification_data(dataset: Dataset) -> Dataset:
    def _proc(example):
        tok = tokenizer(example['claim'], truncation=True, max_length=1024)
        tok["labels"] = int(example['label'])
        return tok
    
    ds_tok = dataset.map(_proc)
    return ds_tok

def main():
    start = time.time()

    # load data
    ds = load_from_disk(os.path.join(DATA_DIR, args.lang))
    # train_msgs_ds = build_messages(ds["train"].select(range(10)))
    # train_msgs_ds = build_messages(ds["train"])

    train_tok = build_classification_data(ds["train"])
    dev_tok = build_classification_data(ds["dev"])
    
    base_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        quantization_config=bnb_config
    )

    # if hasattr(base_model, "enable_input_require_grads"):
    #     base_model.enable_input_require_grads()
    # else:
    #     def make_inputs_require_grad(module, input, output):
    #         output.requires_grad_(True)

    #     base_model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    # base_model.config.use_cache = False

    # LoRa
    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_CLS",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
    )
    
    model = get_peft_model(base_model, lora_cfg)
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    trainable_params, all_param = model.get_nb_trainable_parameters()
    trainable_params_str = (f"trainable params: {trainable_params:,d} || "
                        f"all params: {all_param:,d} || "
                        f"trainable%: {100 * trainable_params / all_param:.4f}"
                        )

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    training_args = TrainingArguments(
        output_dir=None,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=1e-4, # higher learning rate as only adapters are trained
        save_strategy="no",
        save_total_limit=None,
        logging_strategy="epoch",
        report_to="none",
        fp16=False,
        bf16=use_bf16, 
        gradient_checkpointing=True,
        eval_strategy="epoch"
        # assistant_only_loss=True # does not work, hence our own implmentation
    )

    print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
        
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(-1)
        acc = (preds == labels).mean().item() if hasattr((preds==labels), "item") else ((preds==labels).mean())
        # macro F1 over all classes
        f1 = f1_score(labels, preds, average="macro")
        return {"accuracy": acc, "f1_macro": f1}


    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=dev_tok,
        compute_metrics=compute_metrics,
        data_collator=data_collator
    )

    trainer.train()

    
    dev_metrics = trainer.evaluate(eval_dataset=dev_tok)

    # collect losses
    train_losses = []
    eval_losses = []
    
    for log in trainer.state.log_history:
        if "loss" in log:
            train_losses.append({"epoch": log.get("epoch"), "loss": log["loss"]})
        if "eval_loss" in log:
            eval_losses.append({"epoch": log.get("epoch"), "eval_loss": log["eval_loss"]})

    end = time.time()

    # Save meta and model
    model_number = get_model_number(MODEL_DIR)

    meta = {
        "model_number": model_number,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data": args.lang,
        "model": MODEL_ID,
        "train_n": len(ds['train']),
        "dev_n": len(ds['dev']),
        "dev_metrics": dev_metrics,
        "epochs": args.epochs,
        "batch_size": training_args.per_device_train_batch_size,
        "learning_rate": training_args.learning_rate,
        "train_losses": train_losses,
        "eval_losses": eval_losses,
        "lora": {"r": lora_cfg.r, "alpha": lora_cfg.lora_alpha, "dropout": lora_cfg.lora_dropout},
        "trainable_parameters": trainable_params_str,
        "time_min": (end - start) / 60.0,
        "cuda_max_memory_allocation": torch.cuda.max_memory_allocated() / 1024**2,
    }

    out_dir = os.path.join(MODEL_DIR, f"model_{model_number}")
    model.save_pretrained(out_dir)

    append_meta_file(meta, MODEL_DIR)

    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()