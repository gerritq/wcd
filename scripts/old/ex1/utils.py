import os
import random
import evaluate
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import random
import torch
import re
from tqdm import tqdm
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
from datasets import load_from_disk

from prompts import SYSTEM_PROMPTS_SLM
# https://huggingface.co/collections/meta-llama/llama-33-67531d5c405ec5d08a852000
# https://huggingface.co/collections/Qwen/qwen3-67dd247413f0e2e4f653967f

random.seed(42)

accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

BASE_DIR = os.getenv("BASE_WCD")
DATA_DIR = os.path.join(BASE_DIR, "data/sets/main")

##########################################################################################
# Meta functions
##########################################################################################

MODEL_MAPPING =  {
    "mBert": "google-bert/bert-base-multilingual-uncased",
    "xlm-r-b": "FacebookAI/xlm-roberta-base",
    "xlm-r-l": "FacebookAI/xlm-roberta-large",
    "mDeberta-b": "microsoft/mdeberta-v3-base",
    "mDeberta-l": "microsoft/deberta-v3-large",
    "llama3_1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama3_3b": "meta-llama/Llama-3.2-3B-Instruct",
    "llama3_8b": "meta-llama/Llama-3.1-8B-Instruct",
    "llama3_70b": "meta-llama/Llama-3.3-70B-Instruct",
    "qwen3_06b": "Qwen/Qwen3-0.6B",
    "qwen3_4b": "Qwen/Qwen3-4B-Instruct-2507",
    "qwen3_8b": "Qwen/Qwen3-8B",
    "qwen3_30b": "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "qwen3_32b": "Qwen/Qwen3-32B",
    "gemma3_12b": "google/gemma-3-12b-it",
    "gpt_oss": "openai/gpt-oss-20b",
    "mistral_8b": "mistralai/Ministral-8B-Instruct-2410",
    "ds_llama": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "aya": "CohereLabs/aya-101",
    "gpt-4o-mini": "openai/gpt-4o-mini",
    "gemini-2.5-flash-lite": "google/gemini-2.5-flash-lite"
    }

def append_meta_file(meta: dict, model_dir: str):
    """appends to the meta file in the model dir; creates the file if it does not exist"""
    meta_path = os.path.join(model_dir, "meta_overview.jsonl")
    with open(meta_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")

def get_model_number(model_dir: str) -> int:
    """identifies the latest model number on the model doir"""
    model_names = [os.path.splitext(d)[0] for d in os.listdir(model_dir) if d.startswith("model_")]
    
    numbers = []
    for name in model_names:
        
        num = int(name.split("_")[1])
        numbers.append(num)
        
    next_number = max(numbers) + 1 if numbers else 1
    return next_number

def collect_and_save_losses(history, model_dir):
    """inputs the trainer history and the model dir
    collects train and eval losses and saves a plot in the model dir"""

    def plot_loss_curves(train_losses, eval_losses, out_dir):

        out_path = os.path.join(out_dir, "loss_plot.pdf")

        train_epochs = [x["epoch"] for x in train_losses]
        train_vals = [x["loss"] for x in train_losses]
        eval_epochs = [x["epoch"] for x in eval_losses]
        eval_vals = [x["eval_loss"] for x in eval_losses]

        plt.figure(figsize=(6, 4))
        plt.plot(train_epochs, train_vals, marker="o", label="Train Loss", color="steelblue")
        plt.plot(eval_epochs, eval_vals, marker="s", label="Eval Loss", color="tomato")

        plt.title("Training and Evaluation Loss over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        
    train_losses, eval_losses = [], []
    for log in history:
        if "loss" in log:
            train_losses.append({"epoch": log.get("epoch"), "loss": log["loss"]})
        if "eval_loss" in log:
            eval_losses.append({"epoch": log.get("epoch"), "eval_loss": log["eval_loss"]})

    if train_losses and eval_losses:
        plot_loss_curves(train_losses, eval_losses, model_dir)



##########################################################################################
# Code for SLMs
##########################################################################################

def preprocess_function(example, tokenizer):
    """preprocess function for training
    gets the prompt and turns it into a chat template
    """
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

    example["text"] = tokenizer.apply_chat_template(messages['messages'], tokenize=False)

    return example

def preprocess_function_generation(example, tokenizer):
    """ preprocess fucntion for generation !; adds the generation tag, just for eval
    """
    claim = example['claim']
    lang = example['lang'][:2]

    system = SYSTEM_PROMPTS_SLM[lang]['system']
    user = SYSTEM_PROMPTS_SLM[lang]['user'].format(claim=claim)
    
    messages = {
    "messages": [
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ]
    }

    example["text"] = tokenizer.apply_chat_template(messages['messages'], 
                                                    tokenize=False, 
                                                    add_generation_prompt=True,
                                                    enable_thinking=False)
    return example

def get_train_dev(args, tokenizer):
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

def get_test(args, tokenizer):
    test = load_from_disk(os.path.join(DATA_DIR, args.lang))['test']
    test = test.map(preprocess_function_generation, 
                      fn_kwargs={"tokenizer": tokenizer})

    return test


def get_tokenizer(args, inference=False):
    if inference:
        tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side='left')
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    # if tokenizer has no padding token, then reuse the end of sequence token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.model_max_length > 100_000:
        tokenizer.model_max_length = 512*2 
    if not tokenizer.chat_template:
        raise Exception("Tokeniser has not cha template.")

    return tokenizer


def compute_metrics(preds, labels):
    
    acc = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]
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

def evaluation_tok(test, model_dir, tokenizer, bnb_config):
    """takes a non tok dataset as input
    as expecetd for the pwl file
    """
    
    def inference(test, model_dir, tokenizer, bnb_config, batch_size=8):
        """
        inference when test is tokensied!
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
    
    # run here
    predictions, labels = inference(test, model_dir, tokenizer, bnb_config)

    valid = [(p, y) for p, y in zip(predictions, labels) if p is not None]
    if valid:
        p_clean, y_clean = zip(*valid)
        metrics = compute_metrics(list(p_clean), list(y_clean))
    else:
        print("No valid predictions.")
        metrics = {
            "accuracy": 0,
            "f1": 0,
            "true_positives": 0,
            "false_positives": 0,
            "true_negatives": 0,
            "false_negatives": 0
        }   

    metrics['valid'] = len(valid)
            
    return metrics


def evaluation_non_tok(test, model_dir, tokenizer, bnb_config):
    """takes a non tok dataset as input"""
    
    def inference(test, model_dir, tokenizer, bnb_config, batch_size=8):
        model = AutoPeftModelForCausalLM.from_pretrained(model_dir, 
                                                        device_map="auto", 
                                                        torch_dtype="auto", 
                                                        trust_remote_code=True,
                                                        quantization_config=bnb_config)
        model.eval()

        # batch inference
        texts = test['text']
        labels = test['label']
        predictions = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Running batch inference ..."):
            batch = texts[i:i+batch_size]

            input_ids = tokenizer(batch,
                                padding=True,
                                truncation=True,
                                return_tensors="pt").to(model.device)

            with torch.no_grad():
                out = model.generate(
                    **input_ids,
                    max_new_tokens=128, 
                    )

            for j in range(len(batch)):
                output_ids = out[j][len(input_ids["input_ids"][j]):].tolist()
                try:
                    idx = len(output_ids) - output_ids[::-1].index(151668)  # </think> for Qwen
                    print("indec", idx)
                except ValueError:
                    idx = 0
                response = tokenizer.decode(output_ids[idx:], skip_special_tokens=True).strip()
                # print(response)
                # identify labels            
                label = None
                match = re.search(r'["\']label["\']\s*:\s*([01])', response)
                if match:
                    label = int(match.group(1))
                predictions.append(label)
        return predictions, labels
    
    # run here
    predictions, labels = inference(test, model_dir, tokenizer, bnb_config)

    valid = [(p, y) for p, y in zip(predictions, labels) if p is not None]
    if valid:
        p_clean, y_clean = zip(*valid)
        metrics = compute_metrics(list(p_clean), list(y_clean))
    else:
        print("No valid predictions.")
        metrics = {
            "accuracy": 0,
            "f1": 0,
            "true_positives": 0,
            "false_positives": 0,
            "true_negatives": 0,
            "false_negatives": 0
        }   

    metrics['valid'] = len(valid)
            
    return metrics

def get_max_sequence_length(dataset, tokenizer):
    
    def tokenize_and_get_length(example):
        tokenized = tokenizer(
            example['text'], 
            truncation=False, 
            padding=False,
        )
        example['length'] = len(tokenized['input_ids'])
        return example

    dataset_with_lengths = dataset.map(tokenize_and_get_length, batched=False, )    
    max_length = max(dataset_with_lengths['length'])
    
    return max_length


##########################################################################################
# PWL Functions
##########################################################################################

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

def pwl_tokenizer(example, tokenizer, assistant_tag_ids, max_seq_length):
    """pwl tokeniser which only sets labels for the assistant tokens
    note: also special tokens like padding are set to -100
    note 2: can test this with check_labels
    """
    def find_sublist_reverse(sub, lst):
        for i in range(len(lst) - len(sub), -1, -1):  # start from the back
            if lst[i:i+len(sub)] == sub:
                return i
        return -1

    # tokenise text
    text_tok = tokenizer(example['text'], 
                            truncation=True, 
                            max_length=max_seq_length, 
                            padding='max_length')
    text_input_ids = text_tok['input_ids']
    am  = text_tok["attention_mask"]

    # assistant tag tokens
    assistant_tag_index = find_sublist_reverse(assistant_tag_ids, text_input_ids)
    assert assistant_tag_index != -1, f"Could not find the assistant tag."
    # assert (assistant_tag_index / len(text_input_ids)) > .8, f"Assistant tag not in the final 20% of the text."

    # generate completion mask
    # no need to shift lables, done by the model internally:
    # https://discuss.huggingface.co/t/where-does-the-transformers-do-the-target-text-shifting-in-causal-lm/32408
    labels = [-100]*len(text_input_ids)
    start = assistant_tag_index + len(assistant_tag_ids)
    labels[start:] = text_input_ids[start:]
    for i, m in enumerate(am):
        if m == 0:
            labels[i] = -100
    assert len(labels) == len(text_input_ids), "Labels length is incorrect."

    text_tok['labels'] =  labels
    return text_tok

def check_labels(tokenizer, tokenised_item):
    token_ids = np.array(tokenised_item['input_ids'])
    labels = np.array(tokenised_item['labels'])
    mask = labels != -100

    print("Supervised tokens:")
    print(tokenizer.decode(token_ids[mask]))
