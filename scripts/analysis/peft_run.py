import os
import sys
import json
import argparse
import random
import torch
import time
from utils import MODEL_MAPPING
from prompts import PROMPTS
from datasets import load_from_disk
from typing import List, Dict

from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

"""
1. what is add_generation_prompt doing?

2. How to do assistant only loss?
- https://huggingface.co/docs/trl/en/sft_trainer
- https://www.reddit.com/r/LocalLLaMA/comments/1f1ygd7/masking_loss_for_input_tokens_when_finetuning/
- https://github.com/tloen/alpaca-lora/blob/main/finetune.py
- https://gist.github.com/Blaizzy/40de0f6b4340490e3920db9e182e6455
- for data collators: https://huggingface.co/docs/trl/v0.9.6/en/sft_trainer
- Data collator example used to build the below: https://gist.github.com/Blaizzy/40de0f6b4340490e3920db9e182e6455

3. Try doing the -100 by our own
"""

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
args.plw = bool(args.plw) # make plw bool
args.system = bool(args.system) # make plw bool

print(f"Running plw {args.plw}")

# DIRs
BASE_DIR = os.getenv("BASE_WCD")
DATA_DIR = os.path.join(BASE_DIR, "data/sets")
MODEL_DIR = os.path.join(BASE_DIR, "data/models/slm")

# VARs
MODEL_ID = MODEL_MAPPING[args.model]
MAX_LENGTH = 256*4
PROMPT_KEY = args.lang
if args.system:
    PROMPT_KEY = PROMPT_KEY + "_system"
PROMPT = PROMPTS[PROMPT_KEY]

set_seed(42)
random.seed(42)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# TOKENISER
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
tokenizer.truncation_side = "left"

def append_meta_file(meta: dict, model_dir: str):
    meta_path = os.path.join(model_dir, "meta_overview.jsonl")
    with open(meta_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")

def get_model_number(model_dir: str) -> int:
    model_names = [d for d in os.listdir(model_dir) if d.startswith("model_") and os.path.isdir(os.path.join(model_dir, d))]
    
    numbers = []
    for name in model_names:
        
        num = int(name.split("_")[1])
        numbers.append(num)
        
    next_number = max(numbers) + 1 if numbers else 1
    return next_number

def build_messages(dataset: Dataset, PROMPT: str, system: bool) -> Dataset:
    
    def preprocess_function(example):
        
        if system:
            x = {
            "messages": [
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": f"Claim: {example['claim']}"},
                {"role": "assistant", "content": json.dumps({"label": int(example["label"])})}
            ]
        }

        else:
            x = {
                "messages": [
                    {"role": "system", "content": "You are a seasoned Wikipedia fact-checker."},
                    {"role": "user", "content": PROMPT.replace("{{claim}}", example["claim"])},
                    {"role": "assistant", "content": json.dumps({"label": int(example["label"])})}
                ]
            }

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
    train_msgs_ds = build_messages(ds["train"], PROMPT, args.system)
    
    print("Example msg" ,train_msgs_ds[0],"\n\n")

    assistant_tag, assistant_tag_ids = get_generation_tag(tokenizer)
        
    train_tok = train_msgs_ds.map(custom_tokenize,
                                fn_kwargs={"tokenizer": tokenizer,
                                           "assistant_tag_ids": assistant_tag_ids,
                                           "pwt": args.plw}
                                )

    # Load model
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        # device_map="auto", # let sft handle this
        torch_dtype="auto",
        trust_remote_code=True
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
    
    print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")


    training_args = SFTConfig(
        output_dir=None,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        save_strategy="no",
        save_total_limit=1,
        logging_strategy="epoch",
        report_to="none",
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        do_eval=False,
        # chat_template_path="HuggingFaceTB/SmolLM3-3B",
        # assistant_only_loss=True # important so that loss is only computed on the assistant tokens
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tok
    )

    trainer.train()

    print(trainer.state.log_history)

    train_losses = []
    for log in trainer.state.log_history:
        if "loss" in log:
            train_losses.append({"epoch": log.get("epoch"), "loss": log["loss"]})

    end = time.time()

    # Save meta and model
    model_number = get_model_number(MODEL_DIR)

    meta = {
        "model_number": model_number,
        "data": args.lang,
        "model": MODEL_ID,
        "plw": args.plw,
        "system": args.system,
        "train_n": len(ds['train']),
        "test_n": len(ds['test']),
        "epochs": args.epochs,
        "batch_size": training_args.per_device_train_batch_size,
        "learning_rate": training_args.learning_rate,
        "train_losses": train_losses,
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