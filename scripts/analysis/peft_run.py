import os
import sys
import json
import argparse
import random
import torch
import time
from utils import MODEL_MAPPING
from prompts import PROMPT
from datasets import load_from_disk

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

3. Try doing the -100 by our own
"""


parser = argparse.ArgumentParser()
parser.add_argument("--lang", type=str, required=True)
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--epochs", type=int, default=5)
args = parser.parse_args()


BASE_DIR = os.getenv("BASE_WCD")
DATA_DIR = os.path.join(BASE_DIR, "data/sets")
MODEL_DIR = os.path.join(BASE_DIR, "data/models")

MODEL_ID = MODEL_MAPPING[args.model]
MAX_LENGTH = 256*4
set_seed(42)
random.seed(42)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")


# tokeniser and ensure chat template
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

def build_messages(dataset: Dataset) -> Dataset:
    
    def preprocess_function(example):
        return {
            "messages": [
                {"role": "system", "content": "You are a seasoned Wikipedia fact-checker."},
                {"role": "user", "content": PROMPT.replace("{{claim}}", example["claim"])},
                {"role": "assistant", "content": json.dumps({"label": int(example["label"])})}
            ]
        }
    
    data = list(dataset) 
    random.shuffle(data)

    samples = []

    dataset = dataset.map(preprocess_function, remove_columns=["claim", "label"])
    return dataset


def main():
    start = time.time()

    # load data
    ds = load_from_disk(os.path.join(DATA_DIR, args.lang))
    train_ds = build_messages(ds["train"])
    # test_ds = build_messages(ds["test"]) 
    
    print("Example msg" ,train_ds[0],"\n\n")
    # train_tok = preprocess(train_ds, tokenizer)
    # test_tok = preprocess(test_ds, tokenizer)

    # Load model
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
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
        train_dataset=train_ds
    )

    trainer.train()

    print(trainer.state.log_history)

    train_losses = []
    for log in trainer.state.log_history:
        if "loss" in log:
            train_losses.append({"epoch": log.get("epoch"), "loss": log["loss"]})

    end = time.time()

    meta = {
        "data": args.lang,
        "model": MODEL_ID,
        "train_n": len(ds['train']),
        "test_n": len(ds['test']),
        "epochs": args.epochs,
        "batch_size": training_args.per_device_train_batch_size,
        "learning_rate": training_args.learning_rate,
        "train_losses": train_losses,
        "lora": {"r": lora_cfg.lora_r, "alpha": lora_cfg.lora_alpha, "dropout": lora_cfg.lora_dropout},
        "trainable_parameters": trainable_params_str,
        "time_min": (end - start) / 60.0,
        "cuda_max_memory_allocation": torch.cuda.max_memory_allocated() / 1024**2
    }

    out_dir = os.path.join(MODEL_DIR, f"{args.lang}_{args.model}")
    model.save_pretrained(out_dir)
    # tokenizer.save_pretrained(out_dir)

    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()