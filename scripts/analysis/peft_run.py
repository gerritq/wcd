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
from prompts import SYSTEM_PROMPTS_SLM
from datasets import load_from_disk
from typing import List, Dict

from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed, BitsAndBytesConfig
from datetime import datetime

# HF recommends nf4
# 
# bfloat speeds up training
# load in 4bit reducses storage size
# HF recommends nf4
nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_compute_dtype=torch.bfloat16
)


# ARGPARSE
parser = argparse.ArgumentParser()
parser.add_argument("--lang", type=str, required=True)
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--plw", type=int, default=1)
parser.add_argument("--system", type=int, required=True)
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

set_seed(42)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# TOKENISER
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
tokenizer.truncation_side = "left"

MAX_LENGTH = tokenizer.model_max_length
print("Max context length:", MAX_LENGTH)

def build_messages(dataset: Dataset, system: bool) -> Dataset:

    def preprocess_function(example):
        if args.system:
            PROMPT = SYSTEM_PROMPTS_SLM[example['lang']]
        else:
            # to do
            pass
        
        if system:
            x = {
            "messages": [
                {"role": "system", "content": PROMPT['system']},
                {"role": "user", "content": PROMPT['user'].format(claim=example['claim'])},
                {"role": "assistant", "content": PROMPT['assistant'].format(label=example['label'])}
            ]
        }

        else:
            # to do
            pass
        return x
    
    data = list(dataset) 
    random.shuffle(data)

    dataset = dataset.map(preprocess_function, remove_columns=["claim", "label"])
    return dataset


def get_generation_tag(tokenizer):
    messages = ([{"role": "user","content":"test"}])
    no_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    with_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    if with_prompt.startswith(no_prompt):
        assistant_tag = with_prompt[len(no_prompt):]
    else:
        raise ValueError("Cannot identify the generation token.")
    
    assistant_tag_ids = tokenizer(assistant_tag, add_special_tokens=False)["input_ids"]

    return assistant_tag, assistant_tag_ids

def custom_tokenize(example: dict, tokenizer, assistant_tag_ids: List[int], pwt=False):
    """Allows to apply pwt."""

    def find_sublist_reverse(sub, lst):
        for i in range(len(lst) - len(sub), -1, -1):  # start from the back
            if lst[i:i+len(sub)] == sub:
                return i
        return -1

    text = tokenizer.apply_chat_template(
                                            example["messages"],
                                            tokenize=False,
                                            add_generation_prompt=False,
                                            enable_thinking=False
                                        )
    # print(text)
    # text tokens
    text_tok = tokenizer(text, truncation=True, max_length=MAX_LENGTH)
    
    if pwt:
        text_input_ids = text_tok['input_ids']

        # assistant tag tokens
        assistant_tag_index = find_sublist_reverse(assistant_tag_ids, text_input_ids)
        assert assistant_tag_index != -1, f"Could not find the assistant tag."
        assert (assistant_tag_index / len(text_input_ids)) > .8, f"Assistant tag not in the final 20% of the text."

        # generate labels
        # no need to shift lables, done by the model internally:
        # https://discuss.huggingface.co/t/where-does-the-transformers-do-the-target-text-shifting-in-causal-lm/32408
        labels = [-100]*len(text_input_ids)
        start = assistant_tag_index + len(assistant_tag_ids)
        labels[start:] = text_input_ids[start:]

        assert len(labels) == len(text_input_ids), "Labels length is incorrect."

        text_tok['labels'] =  labels
    else:
        text_tok["labels"] = text_tok["input_ids"].copy()
    return text_tok

def main():
    start = time.time()

    # load data
    ds = load_from_disk(os.path.join(DATA_DIR, args.lang))
    # train_msgs_ds = build_messages(ds["train"].select(range(10)))
    # train_msgs_ds = build_messages(ds["train"])
    train_msgs_ds = build_messages(ds["train"], args.system)
    dev_msgs_ds = build_messages(ds["dev"], args.system)
    
    print("Example message for training\n\t", train_msgs_ds[0],"\n\n")

    assistant_tag, assistant_tag_ids = get_generation_tag(tokenizer)
        
    train_tok = train_msgs_ds.map(custom_tokenize,
                                fn_kwargs={"tokenizer": tokenizer,
                                           "assistant_tag_ids": assistant_tag_ids,
                                           "pwt": args.plw}
                                )
    dev_tok = dev_msgs_ds.map(custom_tokenize,
                                fn_kwargs={"tokenizer": tokenizer,
                                           "assistant_tag_ids": assistant_tag_ids,
                                           "pwt": args.plw}
                                )

    # Load model
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        # device_map="auto", # let sft handle this
        trust_remote_code=True,
        quantization_config=nf4_config
    )

    if hasattr(base_model, "enable_input_require_grads"):
        base_model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        base_model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    base_model.config.use_cache = False

    # LoRa
    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
    )
    
    model = get_peft_model(base_model, lora_cfg)

    trainable_params, all_param = model.get_nb_trainable_parameters()
    trainable_params_str = (f"trainable params: {trainable_params:,d} || "
                        f"all params: {all_param:,d} || "
                        f"trainable%: {100 * trainable_params / all_param:.4f}"
                        )

    training_args = SFTConfig(
        output_dir=None,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=1e-4, # higher learning rate as only adapters are trained
        save_strategy="no",
        save_total_limit=1,
        logging_strategy="epoch",
        report_to="none",
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        do_eval=False,
        packing=False,
        max_length=MAX_LENGTH
        # assistant_only_loss=True # does not work, hence our own implmentation
    )

    print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
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

    meta = {
        "model_number": model_number,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data": args.lang,
        "model": MODEL_ID,
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
        "trainable_parameters": trainable_params_str,
        "time_min": (end - start) / 60.0,
        "cuda_max_memory_allocation": torch.cuda.max_memory_allocated() / 1024**2,
        "notes": args.notes
    }

    out_dir = os.path.join(MODEL_DIR, f"model_{model_number}")
    model.save_pretrained(out_dir)

    append_meta_file(meta, MODEL_DIR)

    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()