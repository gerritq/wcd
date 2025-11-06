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
                    collect_and_save_losses,
                    evaluation_non_tok,
                    get_tokenizer,
                    get_train_dev,
                    get_test
)

import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, set_seed, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig

set_seed(42)

BASE_DIR = os.getenv("BASE_WCD")
DATA_DIR = os.path.join(BASE_DIR, "data/sets/main")
MODEL_DIR = os.path.join(BASE_DIR, "data/models/slm/sft")

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
        dataset_text_field="text",
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
 
def run(args):
    start = time.time()
    tokenizer = get_tokenizer(args)
    train, dev = get_train_dev(args, tokenizer)
    test = get_test(args, tokenizer)
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
    metrics = evaluation_non_tok(test, model_dir, tokenizer, bnb_config)

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
        "test_metrics": metrics,
    }

    print(meta)

    append_meta_file(meta, MODEL_DIR)
    with open(os.path.join(model_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

def main():

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
    parser.add_argument("--plw", type=int, default=0)
    args = parser.parse_args()
    args.plw = bool(args.plw)
    args.model = MODEL_MAPPING[args.model]

    print("\n\n",  "="*10, f"Running MODEL {args.model} - LANG {args.lang}","="*10)

    run(args)

if __name__ == "__main__":
    main()