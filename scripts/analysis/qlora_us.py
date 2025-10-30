import os
import json
import unsloth
import torch
import time
import argparse
from datetime import datetime
from tqdm import tqdm
import re
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel
from utils import (
                    MODEL_MAPPING, 
                    append_meta_file, 
                    get_model_number,
                    plot_loss_curves
)
from prompts import SYSTEM_PROMPTS_SLM

from datasets import load_from_disk
BASE_DIR = os.getenv("BASE_WCD")
DATA_DIR = os.path.join(BASE_DIR, "data/sets/main")
MODEL_DIR = os.path.join(BASE_DIR, "data/models/slm/test")

def preprocess_function(example, tokenizer):
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

def get_model():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Qwen3-4B-unsloth-bnb-4bit",
        max_seq_length = 2048*2,   # Context length - can be longer, but uses more memory
        load_in_4bit = True,     # 4bit uses much less memory
        load_in_8bit = False,    # A bit more accurate, uses 2x memory
        full_finetuning = False, # We have full finetuning now!
        # token = "hf_...",      # use one if using gated models
    )


    model = FastLanguageModel.get_peft_model(
        model,
        r = 32,           # Choose any number > 0! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 32,  # Best to choose alpha = rank or rank*2
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,   # We support rank stabilized LoRA
        loftq_config = None,  # And LoftQ
    )

    return model, tokenizer

def inference(args, model_dir, tokenizer, batch_size=8):
    """
    Need to set model eval and torch no grad: https://discuss.pytorch.org/t/model-eval-vs-with-torch-no-grad/19615
    """

    model, _ = FastLanguageModel.from_pretrained(
        model_name = model_dir,
        max_seq_length = 2048,
        load_in_4bit = True,
    )
    model.eval()

    # model = AutoModelForCausalLM.from_pretrained(model_dir, 
    #                                              device_map="auto",
    #                                              quantization_config=bnb_config)
    # model.eval()
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

def collect_and_save_losses(history, model_dir):
    train_losses, eval_losses = [], []
    for log in history:
        if "loss" in log:
            train_losses.append({"epoch": log.get("epoch"), "loss": log["loss"]})
        if "eval_loss" in log:
            eval_losses.append({"epoch": log.get("epoch"), "eval_loss": log["eval_loss"]})

    if train_losses and eval_losses:
        plot_loss_curves(train_losses, eval_losses, model_dir)

def main():
    start = time.time()

    # ARGPARSE
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, required=True)
    args = parser.parse_args()

    print("="*10, f"Running LANG {args.lang}","="*10)

    model, tokenizer = get_model()
    train, dev = get_dataset(args, tokenizer)

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train,
        eval_dataset = dev, # Can set up evaluation!
        args = SFTConfig(
            dataset_text_field = "text",
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4, # Use GA to mimic batch size!
            warmup_steps = 5,
            num_train_epochs = 1, # Set this for 1 full training run.
            # max_steps = 30,
            learning_rate = 2e-4, # Reduce to 2e-5 for long training runs
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            report_to = "none", # Use TrackIO/WandB etc
            logging_strategy="steps",
            logging_steps=20,
            eval_strategy="steps",
            eval_steps=40,
            per_device_eval_batch_size=16,
        ),
    )

    # @title Show current memory stats
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    trainer_stats = trainer.train()

    # Save model
    model_number = get_model_number(MODEL_DIR)
    model_dir = os.path.join(MODEL_DIR, f"model_{model_number}")
    trainer.save_model(model_dir)
    
    # collect_and_save_losses
    collect_and_save_losses(trainer.state.log_history, model_dir)

    # EVAL
    # tokenizer = get_tokenizer(args, inference=True)
    test, labels = get_testset(args, tokenizer)
    predictions = inference(args, model_dir, tokenizer)
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

    }

    with open(os.path.join(model_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()