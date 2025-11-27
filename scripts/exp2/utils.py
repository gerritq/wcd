import os
import json
import re
from tqdm import tqdm
from transformers import (AutoTokenizer, 
                          BitsAndBytesConfig,
                          PreTrainedTokenizerBase)
from torch.utils.data import Dataset
from typing import Dict, List, Callable, Sequence, Tuple, Optional, Any
from transformers import BitsAndBytesConfig
import torch
from peft import (LoraConfig, 
                  AutoPeftModelForCausalLM, 
                  AutoPeftModelForSequenceClassification, 
                  TaskType)
from datasets import load_from_disk
import evaluate
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

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
    "llama3_8b_base": "meta-llama/Llama-3.1-8B",
    "qwen3_06b": "Qwen/Qwen3-0.6B",
    "qwen3_4b": "Qwen/Qwen3-4B-Instruct-2507",
    "qwen3_8b": "Qwen/Qwen3-8B",
    "qwen3_8b_base": "Qwen/Qwen3-8B-Base",
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

# CONFIGS
BNB_CONFIG = BitsAndBytesConfig(
                            load_in_4bit=True, 
                            bnb_4bit_use_double_quant=True, 
                            bnb_4bit_quant_type="nf4", 
                            bnb_4bit_compute_dtype=torch.bfloat16
                            )

LORA_CONFIG_LM = LoraConfig(
        lora_alpha=32,
        lora_dropout=.05,
        r=16,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type=TaskType.CAUSAL_LM,
)

LORA_CONFIG_CLS = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

    
##########################################################################################
# Meta functions
##########################################################################################

def append_meta_file(meta: dict, model_dir: str) -> None:
    """
    Append to the meta file in the model_dir
    """
    meta_path = os.path.join(model_dir, "meta_overview.jsonl")
    with open(meta_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")

import os

def get_model_number(model_dir: str) -> int:
    """
    Identifies the latest meta_<n>.json file in model_dir.
    Returns the next available number.
    """
    meta_files = [f for f in os.listdir(model_dir) if f.startswith("meta_") and f.endswith(".json")]

    numbers = []
    for fname in meta_files:
        num = int(fname.split("_")[1].split(".")[0])
        numbers.append(num)

    return max(numbers) + 1 if numbers else 1

def get_run_number(model_dir: str) -> int:
    """
    Identifies the latest run_\d* directory in model_dir.
    Returns the next available number.
    """
    run_dirs = [d for d in os.listdir(model_dir)
                if os.path.isdir(os.path.join(model_dir, d)) and d.startswith("run_")]

    numbers = []
    for dname in run_dirs:
        try:
            num = int(dname.split("_")[1])
            numbers.append(num)
        except (IndexError, ValueError):
            pass

    return max(numbers) + 1 if numbers else 1

def collect_and_save_losses(history, model_dir):
    """
    Takes the traine.state.history and model_dir.
    Outputs a train/dev loss plot in to model_dir.
    """

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

def collect_checkpoints(temp_dir: str, epochs: int, save_last_epoch: bool) -> list[dict]:
    """
    Helper function to collect all checkpoints int the tmp dir.
    """
    checkpoints = []
    items = [d for d in os.listdir(temp_dir) if d.startswith("checkpoint")]
    items = sorted(items, key=lambda x: int(x.split("-")[-1]))

    assert len(items) == epochs, f"Found {len(items)} but only {epochs} epochs."

    for d in items:
        # get the log history form the trainer state file
        path = os.path.join(temp_dir, d, "trainer_state.json")
        with open(path, "r") as f:
            state = json.load(f)

        log_history = state.get("log_history", [])

        # get last eval loss which is the dev loss
        last_eval_loss = None
        for entry in reversed(log_history):
            if "eval_loss" in entry:
                last_eval_loss = entry["eval_loss"]
                break

        checkpoints.append({
            "path": os.path.join(temp_dir, d),
            "epoch": state.get("epoch"),
            "log_history": log_history,
            "dev_loss": last_eval_loss,
        })
    if save_last_epoch:
        return [checkpoints[-1]]
    else:
        return checkpoints
##########################################################################################
# Tokenizer
##########################################################################################


def get_tokenizer(model_type: str, model_name: str, inference=False):
    """
    Loads and prepares the tokenizer.
    Return it for training or inference (bool keyword)
    """
    if inference:
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    # if tokenizer has no padding token, then reuse the end of sequence token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.model_max_length > 100_000:
        tokenizer.model_max_length = 512*2 
    if model_type != "classifier":
        if not tokenizer.chat_template:
            raise Exception("tokenizer has not cha template.")
    return tokenizer

##########################################################################################
# Data prep
##########################################################################################

def get_all_data_sets(path: str, lang: str) -> List[Dataset]:
    """
    Takes a path and language.
    Returns all three datasets.
    """
    ds = load_from_disk(os.path.join(path, lang))
    return ds['train'], ds['dev'], ds['test']

def preprocess_function_training(example: Dict, 
                                prompt_template: str, 
                                tokenizer: PreTrainedTokenizerBase) -> Dict:
    """
    Preprocess function for training.
    Takes an example, applies the chat template, and returns it.
    """

    system = prompt_template['system']
    user = prompt_template['user'].format(**example)
    assistant = prompt_template['assistant'].format(**example)
    
    messages = {
    "messages": [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant}
    ]
    }

    example["text"] = tokenizer.apply_chat_template(messages['messages'], tokenize=False)

    return example

def preprocess_function_generation(example: Dict, 
                                   prompt_template: Dict, 
                                   tokenizer: PreTrainedTokenizerBase) -> Dict:
    """
    Preprocess function for evaluation -- set generation to true and no assitant message.
    """
    system = prompt_template['system']
    user = prompt_template['user'].format(**example)
    
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

def ds_apply_chat_templates(ds: Dataset, 
                            tokenizer: PreTrainedTokenizerBase, 
                            prompt_template:str,
                            preprocess_function: Callable) -> Dataset:
    """
    Takes a ds.
    Applies the chat template for train/dev or test and returns the data.
    """
    remove_columns = [x for x in ds.column_names if x!="label"]

    ds_chat = ds.map(preprocess_function, 
                     remove_columns=remove_columns,
                     fn_kwargs={"tokenizer": tokenizer,
                                "prompt_template": prompt_template},)

    return ds_chat

def get_max_sequence_length(ds: Dataset, tokenizer: PreTrainedTokenizerBase) -> int:
    """
    Get the max seq length of a ds. Bit inefficient.
    """
    
    def tokenize_and_get_length(example: dict):
        tokenized = tokenizer(
            example['text'], 
            truncation=False, 
            padding=False,
        )
        example['length'] = len(tokenized['input_ids'])
        return example

    dataset_with_lengths = ds.map(tokenize_and_get_length, batched=False, )    
    max_length = max(dataset_with_lengths['length'])
    print(f"Max sequence length: {max_length}")
    
    return max_length

def tokenize_ds(ds: Dataset, tokenizer: PreTrainedTokenizerBase) -> Dataset:
    """
    Takes a ds, applies basic tokenisation with tokenize_fn.
    Return tokenized data.
    """
    
    def tokenize_fn(example: Dict, 
                    tokenizer: PreTrainedTokenizerBase, 
                    max_length: int):
        """
        Basic tok function.
        Returns input_ids, am, and labels.
        """
        enc = tokenizer(
            example["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_attention_mask=True,
        )
        enc["labels"] = enc["input_ids"].copy()
        enc["cd_labels"] = example["label"]
        return enc    
    
    max_length = get_max_sequence_length(ds, tokenizer)
    
    # Tokenize data
    ds_tok = ds.map(tokenize_fn,
                    fn_kwargs={
                               "tokenizer": tokenizer,
                               "max_length": max_length,
                                },
                          batched=False)
    
    return ds_tok

def get_assistant_tag(tokenizer: PreTrainedTokenizerBase):
    """
    Takes a tokenizer.
    Identifies the beginning of the assitant tag to mask prior tokens for assitant loss.
    Returns the assitant tag and ids.
    """

    messages = ([{"role": "user", "content":"test"}])
    no_prompt = tokenizer.apply_chat_template(messages, 
                                              tokenize=False, 
                                              add_generation_prompt=False)
    with_prompt = tokenizer.apply_chat_template(messages, 
                                                tokenize=False, 
                                                add_generation_prompt=True)

    if with_prompt.startswith(no_prompt):
        assistant_tag = with_prompt[len(no_prompt):]
    else:
        raise ValueError("Cannot identify the assitant generation token.")
    
    assistant_tag_ids = tokenizer(assistant_tag, add_special_tokens=False)["input_ids"]

    print("="*20)
    print(f"Assistant tag: <START>{assistant_tag}<END>")
    print("="*20)

    return assistant_tag, assistant_tag_ids


def assistant_loss_tokenizer(example: dict, 
                             tokenizer: PreTrainedTokenizerBase, 
                             assistant_tag_ids: List[int],
                             max_seq_length: int):
    """
    Function to ignore all labels __but__ the assistant tokens for the loss.
    This implement prompt weigt loss or assistant token loss (Atl). 
    Check check_labels() function to see how masking is applied.

    Idea: loss is only based on the assistant tokens. We begin the ATL with the first token 
    __after__ the assistant token starting token. This mirrors the generation setting where
    the model starts producting an answer after the generation token (wich is the initial 
    assistant tag).

    Note: no need to shift labels this is being done by the model.
    See: https://discuss.huggingface.co/t/where-does-the-transformers-do-the-target-text-shifting-in-causal-lm/32408
    """

    def find_sublist_reverse(sub, lst):
        """
        Search a sublist in a list from the back
        """
        for i in range(len(lst) - len(sub), -1, -1):
            if lst[i:i+len(sub)] == sub:
                return i
        return -1

    # Tokenize text
    text_tok = tokenizer(example['text'], 
                         truncation=True, 
                         max_length=max_seq_length, 
                         padding='max_length')
    text_input_ids = text_tok['input_ids']
    am  = text_tok["attention_mask"]

    # Identify where the assistant tokens begin
    assistant_tag_index = find_sublist_reverse(assistant_tag_ids, text_input_ids)
    assert assistant_tag_index != -1, f"Could not find the assistant tag."

    # Generate completion mask
    labels = [-100]*len(text_input_ids)
    start = assistant_tag_index + len(assistant_tag_ids) # We want to get the loss from the a
    # Set input_ids where the assitant tag starts - till the end
    labels[start:] = text_input_ids[start:]
    # Correction: set all padding tokens back to -100
    for i, m in enumerate(am):
        if m == 0:
            labels[i] = -100
    assert len(labels) == len(text_input_ids), "Labels length is incorrect."

    text_tok['labels'] =  labels
    return text_tok

def tokenize_ds_ATL(ds: Dataset, 
                    tokenizer: PreTrainedTokenizerBase):    
    """"""
        
    # find assistant tag and ids, get max length
    assistant_tag, assistant_tag_ids = get_assistant_tag(tokenizer)
    max_length = get_max_sequence_length(ds, tokenizer)
    
    # Tokenize data
    ds_tok = ds.map(assistant_loss_tokenizer,
                    fn_kwargs={
                               "tokenizer": tokenizer,
                               "max_seq_length": max_length,
                               "assistant_tag_ids": assistant_tag_ids
                                },
                          batched=False)
    
    return ds_tok



def check_assitant_token_lables(tokenizer: PreTrainedTokenizerBase, 
                                tokenised_item: dict):
    """Takes a tokenizer and a tokenized item.
    Prints the lables considered for the loss.
    """
    token_ids = np.array(tokenised_item['input_ids'])
    labels = np.array(tokenised_item['labels'])
    mask = labels != -100

    print("Assistant loss example:")
    print(tokenizer.decode(token_ids[mask]))

##########################################################################################
# Evaluation
##########################################################################################

def compute_metrics(preds: Sequence[Any], labels:Sequence[Any]) -> Dict[str, float]:
    """
    Function to compute metrics acc, F1, and true/false pos/neg.
    """
    
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

def compute_metrics_classification(eval_pred: Sequence[Any]) -> Dict[str, float]:
    """
    Fucntion for compute metrics for the classification case to be used in Trainer 
    (i.e. eval_pred as parameter).
    """
    logits, labels = eval_pred
    preds = logits.argmax(-1)

    acc = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]
    f1 = f1_metric.compute(predictions=preds, references=labels, average="binary")["f1"]

    return {"accuracy": acc, "f1": f1}


def evaluation(model_type: str,
               ds: Dataset, 
               model_path: str,
               tokenizer_eval: PreTrainedTokenizerBase, 
               smoke_test: bool = False,
               explanation: bool = False,
               bnb_config: Optional[BitsAndBytesConfig] = None) -> Dict[str, float]:
    """
    Run evaluation based on the tokenized dev or test set.
    Takes a ds and model_path.
    Loads the model and performs inference. 
    Evaluate Acc and F1 on the predictions.
    Returns a dict of metrics.
    """

    def evaluation_classification(ds: Dataset, 
                                  model_path: str,
                                  bnb_config: Optional[BitsAndBytesConfig] = None) -> dict[str, float]:
        """
        Takes a tokenized dataset and a model_path.
        Runs evaluation of the dataset with the model in the path using trainer.eval().
        Returns a dict of metrics.
        """

        model = AutoPeftModelForSequenceClassification.from_pretrained(
            model_path,
            device_map={"": "cuda"},
            torch_dtype="auto", 
            trust_remote_code=True,
            quantization_config=bnb_config
        )

        args = TrainingArguments(
            output_dir=None,
            report_to="none",
            per_device_eval_batch_size=16,
        )

        trainer = Trainer(
            model=model,
            args=args,
            compute_metrics=compute_metrics_classification
        )

        metrics = trainer.evaluate(ds)
        return metrics
    
    def inference(ds: Dataset, 
                  model_path: str, 
                  tokenizer_eval: PreTrainedTokenizerBase,
                  explanation: bool,
                  bnb_config: Optional[BitsAndBytesConfig], 
                  batch_size=16) -> Tuple[List, List, List]:
        """
        Loads the model in model_path and runs inference
        """
        RE_LABEL = re.compile(r'"label"\s*:\s*([01])')
        RE_EXPLANATION = re.compile(r'"rationale"\s*:\s*"(.*?)"', re.DOTALL)

        # Load the model in eval mode
        model = AutoPeftModelForCausalLM.from_pretrained(model_path, 
                                                         device_map={"": "cuda"},
                                                         torch_dtype="auto", 
                                                         trust_remote_code=True,
                                                         quantization_config=bnb_config)
        model.eval()

        model.generation_config.pad_token_id = tokenizer_eval.pad_token_id

        # run batch inference
        predictions, labels, explanations = [], [], []
        for i in tqdm(range(0, len(ds), batch_size), desc="Running batch inference ..."):
            
            batch = ds[i:i+batch_size]
            labels.extend(batch['cd_labels'])
            
            input_ids=torch.tensor(batch["input_ids"]).to(model.device)
            attention_mask=torch.tensor(batch["attention_mask"]).to(model.device)

            current_batch_size = input_ids.shape[0]
            input_seq_length = input_ids.shape[1]
            
            with torch.no_grad():
                out = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=32 if not explanation else 512, # higher for explanations
                    temperature=.1,
                    )

            for j in range(current_batch_size): # batch size
                # decode outputs
                output_ids = out[j][input_seq_length:].tolist()
                try:
                    idx = len(output_ids) - output_ids[::-1].index(151668)  # </think> for Qwen
                except ValueError:
                    idx = 0
                response = tokenizer_eval.decode(output_ids[idx:], 
                                                 skip_special_tokens=True).strip()
                if smoke_test:
                    print(f"Batch {i} Item {j}")
                    print("Response:", response)

                # identify labels            
                label = None
                match = RE_LABEL.search(response)
                if match:
                    label = int(match.group(1))
                predictions.append(label)

                # identify explanations            
                explanation = None
                match = RE_EXPLANATION.search(response)
                if match:
                    explanation = match.group(1)
                explanations.append(explanation)

        return predictions, labels, explanations
    
    
    if model_type != "classifier":
        # run here
        predictions, labels, explanations = inference(ds, 
                                                    model_path, 
                                                    tokenizer_eval,
                                                    explanation,
                                                    bnb_config=bnb_config)

        # Binary predictions
        valid = [(p, y) for p, y in zip(predictions, labels) if p is not None]
        if valid:
            p_clean, y_clean = zip(*valid)
            metrics = compute_metrics(list(p_clean), list(y_clean))
        else:
            print("\tNo valid predictions.")
            metrics = {
                "accuracy": 0.0,
                "f1": 0.0,
                "true_positives": 0.0,
                "false_positives": 0.0,
                "true_negatives": 0.0,
                "false_negatives": 0.0
            }   

        metrics['valid'] = len(valid)
        metrics['total'] = len(labels)

        # Explanations
        # if explanations:
        #     with open(os.path.join(model_path, "explanations.jsonl"), "w", encoding="utf-8") as f:
        #         for line in explanations:
        #             f.write(json.dumps(line, ensure_ascii=False) + "\n")

        return metrics
    else:
        return evaluation_classification(ds, model_path, bnb_config)