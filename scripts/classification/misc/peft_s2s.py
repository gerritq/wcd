import os
import sys
import json
import argparse
import random
import torch
import time
from typing import List, Dict
from datetime import datetime

from utils import (
    MODEL_MAPPING,
    append_meta_file,
    get_model_number,
)
from prompts import SYSTEM_PROMPTS_SLM
from datasets import load_from_disk, Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    set_seed,
    BitsAndBytesConfig,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)

# 4-bit NF4 quantization
bnb_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_use_double_quant=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_compute_dtype=torch.bfloat16
)

# ARGPARSE
parser = argparse.ArgumentParser()
parser.add_argument("--lang", type=str, required=True)
parser.add_argument("--model", type=str, required=True)      # key into MODEL_MAPPING (use a seq2seq ckpt, e.g. google/mt5-base)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--plw", type=int, default=1)            # kept for parity; not used in seq2seq labeling
parser.add_argument("--system", type=int, required=True)     # whether to use system prompts per language
parser.add_argument("--notes", type=str)
args = parser.parse_args()
args.plw = bool(args.plw)
args.system = bool(args.system)

# DIRs
BASE_DIR = os.getenv("BASE_WCD")
DATA_DIR = os.path.join(BASE_DIR, "data/sets")
MODEL_DIR = os.path.join(BASE_DIR, "data/models/slm")

# VARs
MODEL_ID = MODEL_MAPPING[args.model]
SEED = 42

set_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# TOKENIZER
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
tokenizer.truncation_side = "left"

MAX_SOURCE_LEN = 1024
MAX_TARGET_LEN = 32

def format_example(example: dict) -> Dict[str, str]:

    lang = example["lang"]
    prompt = SYSTEM_PROMPTS_SLM[lang]
    
    source = prompt["system"] + "\n" + prompt["user"].format(claim=example["claim"])
    target = prompt["assistant"].format(label=example["label"])  # e.g., "<label>1</label>"
    return {"source": source, "target": target}

def tokenise_seq2seq(dataset: Dataset) -> Dataset:
    def _tok(example):
        source = f"Decide whether this claim needs a citation or not: {example['claim']}" 
        target = SYSTEM_PROMPTS_SLM[example['lang']]["assistant"].format(label=example["label"])

        model_inputs = tokenizer(
            source,
            max_length=MAX_SOURCE_LEN,
            truncation=True,
            padding=False,
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                target,
                max_length=MAX_TARGET_LEN,
                truncation=True,
                padding=False,
            )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return dataset.map(_tok)

def main():
    start = time.time()

    # Load dataset
    ds = load_from_disk(os.path.join(DATA_DIR, args.lang))
    train_raw = ds["train"]
    dev_raw   = ds["dev"]

    # Tokenize
    train_tok = tokenise_seq2seq(train_raw)
    dev_tok   = tokenise_seq2seq(dev_raw)

    # Load seq2seq model in 4-bit
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        quantization_config=bnb_config,
    )
    base_model.config.use_cache = False

    # LoRA for seq2seq
    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_2_SEQ_LM",
        target_modules=["k","q","v","o"],
    )
    model = get_peft_model(base_model, lora_cfg)

    # Trainer setup
    training_args = Seq2SeqTrainingArguments(
        output_dir=None,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=1e-4,               # adapters-only, can be higher
        save_strategy="no",
        logging_strategy="epoch",
        report_to="none",
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        predict_with_generate=True,       # handy if you later eval decode outputs
    )
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
  
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_tok,
        eval_dataset=dev_tok,
    )

    trainer.train()

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
    out_dir = os.path.join(MODEL_DIR, f"model_{model_number}")
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    # Memory info (guarded for CPU)
    cuda_mem_mb = (torch.cuda.max_memory_allocated() / 1024**2) if torch.cuda.is_available() else None

    meta = {
        "model_number": model_number,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data": args.lang,
        "model": MODEL_ID,
        "seq2seq": True,
        "plw": args.plw,
        "system": args.system,
        "train_n": len(ds['train']),
        "dev_n": len(ds['dev']),
        "epochs": args.epochs,
        "batch_size": training_args.per_device_train_batch_size,
        "learning_rate": training_args.learning_rate,
        "train_losses": train_losses,
        "eval_losses": eval_losses,
        "lora": {"r": lora_cfg.r, "alpha": lora_cfg.lora_alpha, "dropout": lora_cfg.lora_dropout},
        "time_min": (end - start) / 60.0,
        "cuda_max_memory_allocation_mb": cuda_mem_mb,
        "notes": args.notes,
    }

    append_meta_file(meta, MODEL_DIR)
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()