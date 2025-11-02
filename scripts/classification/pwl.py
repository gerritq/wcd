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
import numpy as np
from utils import (
                    MODEL_MAPPING, 
                    append_meta_file, 
                    get_model_number,
                    plot_loss_curves
)
from prompts import SYSTEM_PROMPTS_SLM

from qlora2 import  (preprocess_function_generation,
                     collect_and_save_losses,
                     get_tokenizer)

import torch
from datasets import load_from_disk
from peft import LoraConfig, AutoPeftModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
# from transformers import TrainingArguments
import evaluate
from sklearn.metrics import confusion_matrix
import re

"""
1. Check whether the decoding is correct. Print some outputs
"""

acc_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

set_seed(42)

BASE_DIR = os.getenv("BASE_WCD")
DATA_DIR = os.path.join(BASE_DIR, "data/sets/main")
MODEL_DIR = os.path.join(BASE_DIR, "data/models/slm/pwl")

def get_data(args):
    ds = load_from_disk(os.path.join(DATA_DIR, args.lang))
    return ds['train'], ds['dev'], ds['test']

def check_labels(tokenizer, tokenised_item):
    token_ids = np.array(tokenised_item['input_ids'])
    labels = np.array(tokenised_item['labels'])
    mask = labels != -100

    print(tokenizer.decode(token_ids[mask]))

def tokenise_data(data, 
                  tokenizer):
    """custom tokenise to implement pwl"""

    def get_assistant_tag(tokenizer):
        """function to get the assistant tag
        
        return the assistant tag and ids
        """
        messages = ([{"role": "user", "content":"test"}])
        no_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        with_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        if with_prompt.startswith(no_prompt):
            assistant_tag = with_prompt[len(no_prompt):]
        else:
            raise ValueError("Cannot identify the generation token.")
        
        assistant_tag_ids = tokenizer(assistant_tag, add_special_tokens=False)["input_ids"]

        return assistant_tag, assistant_tag_ids

    def preprocess_function(example, tokenizer):
        """preprocess to obtain the prompt"""
        claim = example['claim']
        label = example['label']
        lang = example['lang'][:2] # in case we test more data eg en_8k

        system = SYSTEM_PROMPTS_SLM[lang]['system']
        user = SYSTEM_PROMPTS_SLM[lang]['user'].format(claim=claim)
        assistant = SYSTEM_PROMPTS_SLM[lang]['assistant'].format(label=label)
        
        messages = {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant}
        ]
        }

        example['text'] = tokenizer.apply_chat_template(messages,
                                                        tokenize=False,
                                                        add_generation_prompt=False,
                                                        enable_thinking=False
                                                        )
        return example
        
    def custom_tokenizer(example, tokenizer, assistant_tag_ids):
        def find_sublist_reverse(sub, lst):
            for i in range(len(lst) - len(sub), -1, -1):  # start from the back
                if lst[i:i+len(sub)] == sub:
                    return i
            return -1

        text_tok = tokenizer(example['text'], truncation=True, max_length=tokenizer.model_max_length)
        text_input_ids = text_tok['input_ids']
        
        # assistant tag tokens
        assistant_tag_index = find_sublist_reverse(assistant_tag_ids, text_input_ids)
        assert assistant_tag_index != -1, f"Could not find the assistant tag."
        assert (assistant_tag_index / len(text_input_ids)) > .8, f"Assistant tag not in the final 20% of the text."

        # generate completion mask
        # no need to shift lables, done by the model internally:
        # https://discuss.huggingface.co/t/where-does-the-transformers-do-the-target-text-shifting-in-causal-lm/32408
        labels = [-100]*len(text_input_ids)
        start = assistant_tag_index + len(assistant_tag_ids)
        labels[start:] = text_input_ids[start:]

        assert len(labels) == len(text_input_ids), "Labels length is incorrect."

        text_tok['labels'] =  labels
        return text_tok


    # find assistant tag and ids
    assistant_tag, assistant_tag_ids = get_assistant_tag(tokenizer)

    # get messages
    ds_messages = data.map(preprocess_function, fn_kwargs={"tokenizer": tokenizer},)

    ds_tok = ds_messages.map(custom_tokenizer,
                              fn_kwargs={"tokenizer": tokenizer,
                              "assistant_tag_ids": assistant_tag_ids}
                            )

    return ds_tok


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
            r=16,
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
    )
    # https://www.philschmid.de/fine-tune-google-gemma
    training_args = SFTConfig(
        output_dir=None,
        num_train_epochs=args.epochs,                     # number of training epochs
        per_device_train_batch_size=args.batch_size,          # batch size per device during training
        gradient_accumulation_steps=args.grad_acc, #2 before          # number of steps before performing a backward/update pass
        gradient_checkpointing=True,            # use gradient checkpointing to save memory
        bf16=True,                              # use bfloat16 precision
        learning_rate=args.learning_rate,                     # learning rate, based on QLoRA paper
        max_grad_norm=args.max_grad_norm,                      # max gradient norm based on QLoRA paper
        warmup_ratio=args.warmup_ratio,                      # warmup ratio based on QLoRA paper
        weight_decay=args.weight_decay,
        lr_scheduler_type="linear", # cosine
        max_length=tokenizer.model_max_length,
        report_to='none',
        logging_strategy="steps",
        logging_steps=20,
        model_init_kwargs={"quantization_config": bnb_config},
        eval_strategy="steps",
        eval_steps=60,
        per_device_eval_batch_size=16,
        )
    return training_args, bnb_config, lora_config
 

def inference(test, model_dir, tokenizer, bnb_config, batch_size=8):
    """
    Need to set model eval and torch no grad: https://discuss.pytorch.org/t/model-eval-vs-with-torch-no-grad/19615
    """
    model = AutoPeftModelForCausalLM.from_pretrained(model_dir, 
                                                     device_map="auto", 
                                                     torch_dtype="auto", 
                                                     trust_remote_code=True,
                                                     quantization_config=bnb_config)
    model.eval()

    # batch inference
    predictions, labels = [], []
    for i in tqdm(range(0, len(test), batch_size), desc="Running batch inference ..."):
        batch = test[i:i+batch_size]
        labels.extend(batch['labels'])
        
        with torch.no_grad():
            out = model.generate(
                torch.tensor(batch["input_ids"]).to(model.device),
                max_new_tokens=24, 
                )

        for j in range(len(batch["input_ids"])):
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
    return predictions, labels


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

def evaluation(test, model_dir, tokenizer, bnb_config):

    
    predictions, labels = inference(test, model_dir, tokenizer, bnb_config)

    valid = [(p, y) for p, y in zip(predictions, labels) if p is not None]
    if valid:
        p_clean, y_clean = zip(*valid)
        metrics = compute_metrics(list(p_clean), list(y_clean))
    else:
        raise ValueError("No valid predictions.")

    metrics['valid'] = len(valid)
            
    return metrics

def run(args):
    # Get tokenizer and data
    tokenizer = get_tokenizer(args)
    train, dev, test = get_data(args)
    train_tok, dev_tok, test_tok = tokenise_data(train, tokenizer), tokenise_data(dev, tokenizer), tokenise_data(test, tokenizer)
    print(train[0]['text'])

    # configs
    training_args, bnb_config, lora_config = get_config(args, tokenizer)

    # trainer
    trainer = SFTTrainer(
        model=args.model,
        args=training_args,
        peft_config=lora_config,
        train_dataset=train_tok,
        eval_dataset=dev_tok,
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
    metrics = evaluation(test_tok, model_dir, tokenizer, bnb_config)

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
    }

    print(meta)

    append_meta_file(meta, MODEL_DIR)
    with open(os.path.join(model_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

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
    parser.add_argument("--weight_decay", type=float, required=True)
    parser.add_argument("--warmup_ratio", type=float, required=True)
    args = parser.parse_args()
    args.model = MODEL_MAPPING[args.model]

    print("="*10, f"Running MODEL {args.model} - LANG {args.lang}","="*10)

    run(args)

if __name__ == "__main__":
    main()