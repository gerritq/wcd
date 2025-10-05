import os
import json
import argparse
import random
import torch
import time
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed
)
from peft import LoraConfig, get_peft_model
from prompts import PROMPT
from datasets import load_from_disk

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, required=True)
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--lora_r", type=int, default=16)
parser.add_argument("--lora_alpha", type=int, default=32)
parser.add_argument("--lora_dropout", type=float, default=0.05)
args = parser.parse_args()

model_id = {
    "llama_3_70b": "meta-llama/Meta-Llama-3-70B-Instruct",
    "llama_3_8b": "meta-llama/Meta-Llama-3-8B-Instruct",
    "qwen8b": "Qwen/Qwen3-8B",
    "qwen3b": "Qwen/Qwen3-30B-A3B",
    "qwen06b": "Qwen/Qwen3-0.6B",
    "qwen32b": "Qwen/Qwen3-32B"
}.get(args.model)

BASE_DIR = "/scratch/prj/inf_nlg_ai_detection/wcd"
MAX_LENGTH = 256
SEED = 42



set_seed(SEED)
random.seed(SEED)

def build_messages(dataset: Dataset) -> Dataset:
    data = list(dataset)  # convert to list of dicts
    random.shuffle(data)

    samples = []
    for x in data:
        claim = x["claim"] 
        label = int(x["label"])
        user_msg = PROMPT.replace("{{claim}}", claim)
        messages = [
            {"role": "system", "content": "You are a seasoned Wikipedia fact-checker."},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": json.dumps({"label": label})}
        ]
        samples.append({"messages": messages, "label": label})

    return Dataset.from_list(samples)

def load_data():
    if args.data in ["cn_fa","cn_fa_ss","cn_fa_ss_nl"]:
        with open(os.path.join(BASE_DIR, f"data/proc/{args.data}.jsonl"), "r", encoding="utf-8") as f:
            return [json.loads(l) for l in f]
    data = []
    with open(os.path.join(BASE_DIR, f"data/proc/{args.data}.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            obj["label"] = obj.pop("label_2")
            data.append(obj)
    return data


def preprocess(dataset, tokenizer):
    def tokenize_fn(example):
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False
        )
        return tokenizer(text, truncation=True, max_length=MAX_LENGTH)
    return dataset.map(tokenize_fn, remove_columns=["messages"])

def main():
    start = time.time()
    save_dir = os.path.join(BASE_DIR, f"data/ft/{args.data.split('_')[0]}_lora_{args.model}")
    os.makedirs(save_dir, exist_ok=True)
    
    ds = load_from_disk(os.path.join(BASE_DIR, f"data/sets/{args.data}"))
    train_ds, test_ds = build_messages(ds["train"]), build_messages(ds["test"])
    

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    train_ds = preprocess(train_ds, tokenizer)
    test_ds = preprocess(test_ds, tokenizer)

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
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


    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
    )
    model = get_peft_model(base_model, lora_cfg)

    print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=f"./tmp",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="no",
        save_total_limit=1,
        logging_dir="./logs",
        logging_steps=50,
        report_to="none",
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=data_collator
    )

    trainer.train()

    adapter_dir = os.path.join(BASE_DIR, f"data/ft/lora_{args.model}_{args.data}")
    os.makedirs(adapter_dir, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    end = time.time()
    meta = {
        "data": args.data,
        "model": model_id,
        "n": len(ds),
        "train_n": len(train_ds),
        "val_n": len(test_ds),
        "epochs": args.epochs,
        "lora": {"r": args.lora_r, "alpha": args.lora_alpha, "dropout": args.lora_dropout},
        "time_min": (end - start) / 60.0,
        "cuda_max_memory_allocation": torch.cuda.max_memory_allocated() / 1024**2
    }
    with open(os.path.join(adapter_dir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

if __name__ == "__main__":
    main()