# https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/Mistral/Supervised_fine_tuning_(SFT)_of_an_LLM_using_Hugging_Face_tooling.ipynb#scrollTo=VvIfUqjK6ntu
# https://www.philschmid.de/fine-tune-google-gemma
# https://github.com/unslothai/unsloth/issues/1264
# https://docs.unsloth.ai/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide
# https://cookbook.openai.com/articles/gpt-oss/fine-tune-transfomers
# https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_(4B)-Instruct.ipynb
# https://docs.unsloth.ai/get-started/unsloth-notebooks
import os
import json
import argparse
import time
from datetime import datetime
from tqdm import tqdm
from utils import (
                    MODEL_MAPPING, 
                    append_meta_file, 
                    get_model_number,
                    plot_loss_curves
)
# from prompts import SYSTEM_PROMPTS_SLM

import torch
from datasets import load_from_disk
from peft import LoraConfig, AutoPeftModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
# from transformers import TrainingArguments
import evaluate
from sklearn.metrics import confusion_matrix
import re

acc_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

set_seed(42)

BASE_DIR = os.getenv("BASE_WCD")
DATA_DIR = os.path.join(BASE_DIR, "data/sets/old")
MODEL_DIR = os.path.join(BASE_DIR, "data/models/slm/sft")

SYSTEM_PROMPTS_SLM = {
    "en": {
        "system": (
            "You are an expert Wikipedia editor. Your task is to classify whether a claim requires a citation or not. "
            "Answer with \"YES\" if h"
        ),
        "user": "Claim: {claim}",
        "user_context": "Section: {section} Context: {context} Claim: {claim}",
        "assistant": "<label>{label}</label>"
    },
    "nl": {
        "system": (
            "You are a multilingual Wikipedia citation classifier. Read a Dutch claim and decide if it needs a citation. Output ONLY <label>1</label> if a citation is needed, otherwise <label>0</label>."
        ),
        "user": "Claim (Dutch): {claim}",
        # "user": "Section: {section} Context: {context} Claim: {claim}",
        "assistant": "<label>{label}</label>"
    }
}


def preprocess_function(example, tokenizer):
    claim = example['claim']
    label = example['label']
    lang = example['lang'][:2] # in case we test more data eg en_8k

    system = SYSTEM_PROMPTS_SLM[lang]['system']
    user = SYSTEM_PROMPTS_SLM[lang]['user'].format(**example)
    assistant = SYSTEM_PROMPTS_SLM[lang]['assistant'].format(**example)
    
    messages = {
    "messages": [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant}
    ]
    }

    example["text"] = tokenizer.apply_chat_template(messages['messages'], tokenize=False)

    return example

def preprocess_function_generation(example, tokenizer):
    claim = example['claim']
    lang = example['lang'][:2]

    system = SYSTEM_PROMPTS_SLM[lang]['system']
    user = SYSTEM_PROMPTS_SLM[lang]['user'].format(**example)
    
    messages = {
    "messages": [
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ]
    }

    example["text"] = tokenizer.apply_chat_template(messages['messages'], 
                                                    tokenize=False, 
                                                    add_generation_prompt=True)

    return example

def get_dataset(args, tokenizer):
    ds = load_from_disk(os.path.join(DATA_DIR, args.lang))
    train = ds['train']
    dev = ds['dev']

    remove_columns = [x for x in train.column_names if x != "text"]
    train = train.map(preprocess_function, 
                      remove_columns=remove_columns,
                      fn_kwargs={"tokenizer": tokenizer},)
    dev = dev.map(preprocess_function, 
                  remove_columns=remove_columns,
                  fn_kwargs={"tokenizer": tokenizer},)

    
    return train, dev

def get_testset(args, tokenizer):
    ds = load_from_disk(os.path.join(DATA_DIR, args.lang))
    test = ds['test']
    test = test.map(preprocess_function_generation, 
                      fn_kwargs={"tokenizer": tokenizer})

    text = list(test['text'])
    labels = list(test['label'])
    
    return text, labels

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
            lora_alpha=32,
            lora_dropout=.1,
            r=32,
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
    )
    # https://www.philschmid.de/fine-tune-google-gemma
    training_args = SFTConfig(
        output_dir=None,
        num_train_epochs=args.epochs,                     # number of training epochs
        per_device_train_batch_size=args.batch_size,          # batch size per device during training
        gradient_accumulation_steps=2, #2 before          # number of steps before performing a backward/update pass
        gradient_checkpointing=True,            # use gradient checkpointing to save memory
        bf16=True,                              # use bfloat16 precision
        learning_rate=args.learning_rate,                     # learning rate, based on QLoRA paper
        # max_grad_norm=0.3,                      # max gradient norm based on QLoRA paper
        warmup_ratio=0.03,                      # warmup ratio based on QLoRA paper
        lr_scheduler_type="linear", # cosine
        dataset_text_field="text",
        max_length=tokenizer.model_max_length,
        report_to='none',
        logging_strategy="steps",
        logging_steps=20,
        model_init_kwargs={"quantization_config": bnb_config},
        eval_strategy="steps",
        eval_steps=40,
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

def inference(args, model_dir, tokenizer, bnb_config, batch_size=8):
    """
    Need to set model eval and torch no grad: https://discuss.pytorch.org/t/model-eval-vs-with-torch-no-grad/19615
    """
    model = AutoPeftModelForCausalLM.from_pretrained(model_dir, 
                                                     device_map="auto", 
                                                     torch_dtype="auto", 
                                                     trust_remote_code=True,
                                                     quantization_config=bnb_config)
    model.eval()

    test, labels = get_testset(args, tokenizer)

    # batch inference
    predictions = []
    for i in tqdm(range(0, len(test), batch_size), desc="Running batch inference ..."):
        batch = test[i:i+batch_size]
        input_ids = tokenizer(batch,
                              padding=True,
                              truncation=True,
                              return_tensors="pt").to(model.device)

        with torch.no_grad():
            out = model.generate(
                **input_ids,
                max_new_tokens=54, 
                )

        for j in range(len(batch)):
            output_ids = out[j][len(input_ids["input_ids"][j]):].tolist()
            try:
                idx = len(output_ids) - output_ids[::-1].index(151668)  # </think> for Qwen
            except ValueError:
                idx = 0
            response = tokenizer.decode(output_ids[idx:], skip_special_tokens=True).strip()

            # identify labels            
            label = None
            match = re.search(r"<label>\s*([01])\s*</label>", response, re.DOTALL | re.IGNORECASE)
            if match:
                label = int(match.group(1))
            predictions.append(label)
    return predictions


def compute_metrics(preds, labels):
    
    acc = acc_metric.compute(predictions=preds, references=labels)["accuracy"]
    f1  = f1_metric.compute(predictions=preds, references=labels, average="binary")["f1"]

    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()

    return {
        "accuracy": acc,
        "f1": f1,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn)
    }

def evaluation(predictions, labels):
    
    valid = [(p, y) for p, y in zip(predictions, labels) if p is not None]
    if valid:
        p_clean, y_clean = zip(*valid)
        metrics = compute_metrics(list(p_clean), list(y_clean))
    else:
        raise ValueError("No valid predictions.")
            
    return metrics, len(labels), len(valid)

def main():
    start = time.time()

    # ARGPARSE
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--plw", type=int, default=0)
    args = parser.parse_args()
    args.plw = bool(args.plw)
    args.model = MODEL_MAPPING[args.model]

    print("="*10, f"Running MODEL {args.model} - LANG {args.lang}","="*10)

    tokenizer = get_tokenizer(args)
    train, dev = get_dataset(args, tokenizer)
    print(train[0]['text'])

    # configs
    training_args, bnb_config, lora_config = get_config(args, tokenizer)

    trainer = SFTTrainer(
        model=args.model,
        args=training_args,
        peft_config=lora_config,
        train_dataset=train,
        eval_dataset=dev,
        processing_class=tokenizer,
        )
    
    print(f"\tMax memory allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
    
    train_result = trainer.train()

    # Save model
    model_number = get_model_number(MODEL_DIR)
    model_dir = os.path.join(MODEL_DIR, f"model_{model_number}")
    trainer.save_model(model_dir)
    
    # collect_and_save_losses
    collect_and_save_losses(trainer.state.log_history, model_dir)

    # EVAL
    tokenizer = get_tokenizer(args, inference=True)
    test, labels = get_testset(args, tokenizer)
    predictions = inference(args, model_dir, tokenizer, bnb_config)
    metrics = evaluation(predictions, labels)

    meta = {
        "model_number": model_number,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data": args.lang,
        "model": args.model,
        "train_n": len(train),
        "dev_n": len(dev),
        "test_n": len(test),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "lora": {"r": lora_config.r, "alpha": lora_config.lora_alpha, "dropout": lora_config.lora_dropout},
        "time_min": (time.time() - start) / 60.0,
        "cuda_max_memory_allocation": torch.cuda.max_memory_allocated() / 1024**2,
        "n_test": metrics[1],
        "n_valid": metrics[2],
        "metrics": metrics[0],
        "note": "test prompt"

    }

    print(meta)

    append_meta_file(meta, MODEL_DIR)
    with open(os.path.join(model_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()