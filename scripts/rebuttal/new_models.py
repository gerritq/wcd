import argparse
from ast import arg
from email import parser
import json
import os
import random
import re

import torch
from datasets import DatasetDict, load_from_disk
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

MODEL_DICT = {
    "google/mt5-base": "mt5_base",
    'google/umt5-small': "umt5_small",
    "google/mt5-xl": "mt5_xl", 
    "google/mt5-large": "mt5_large", 
    "facebook/mbart-large-50": "mbart_large_50", 
}

CONTEXT_TEMPLATE = (
    "Section: {section}\n"
    "Previous Sentence: {previous_sentence}\n"
    "Claim: {claim}\n"
    "Subsequent Sentence: {subsequent_sentence}"
)

LANGUAGE_NAME = {
    "en": "English",
    "nl": "Dutch",
    "no": "Norwegian",
    "it": "Italian",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "uk": "Ukrainian",
    "bg": "Bulgarian",
    "id": "Indonesian",
    "vi": "Vietnamese",
    "tr": "Turkish",
    "ar": "Arabic",
    "mk": "Macedonian",
    "hy": "Armenian",
    "sq": "Albanian",
    "az": "Azerbaijani",
    "sr": "Serbian",
    "de": "German",
    "uz": "Uzbek",
}


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_language_splits(base_dir: str, lang: str) -> DatasetDict:
    data_path = os.path.join(base_dir, "data", "sets", "main", lang)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset path does not exist: {data_path}")

    ds = load_from_disk(data_path)
    required = {"train", "dev", "test"}
    if not required.issubset(set(ds.keys())):
        raise ValueError(
            f"Dataset at {data_path} must contain train/dev/test; got {list(ds.keys())}"
        )
    print(f"Loaded data from: {data_path}")
    return ds


# def to_text(example: dict, default_lang: str) -> dict:
#     lang_code = str(example.get("lang", default_lang))[:2]
#     lang_name = LANGUAGE_NAME.get(lang_code, lang_code)
#     context = CONTEXT_TEMPLATE.format(
#         section=str(example.get("section", "")),
#         previous_sentence=str(example.get("previous_sentence", "")),
#         claim=str(example.get("claim", "")),
#         subsequent_sentence=str(example.get("subsequent_sentence", "")),
#     )
#     label = str(int(example["label"]))
#     return {"input_text": f"classify: {context}", "target_text": label}

def to_text(example: dict, default_lang: str) -> dict:
    prev = str(example.get("previous_sentence", ""))
    claim = str(example.get("claim", ""))
    next_sent = str(example.get("subsequent_sentence", ""))
    
    # Simple concatenation with spaces
    context = f"{prev} {claim} {next_sent}".strip()
    
    label = str(int(example["label"]))
    return {"input_text": f"classify: {context}", "target_text": label}

def tokenize_dataset(
    args,
    ds: DatasetDict,
    tokenizer,
    lang: str,
    max_source_length: int,
    max_target_length: int,
) -> DatasetDict:
    ds = ds.map(to_text, fn_kwargs={"default_lang": lang})

    if args.smoke_test:
        print("=" * 20)
        print("EXAMPLE INPUTS (SMOKE TEST)")
        for i in range(3):
            print(f"RAW INPUT {i}:")
            print(ds["train"][i]["input_text"])
            print(f"RAW TARGET {i}: {ds['train'][i]['target_text']}")
            print("-" * 20)
        print("=" * 20)

    def _tokenize(batch: dict) -> dict:
        model_inputs = tokenizer(
            batch["input_text"],
            max_length=max_source_length,
            truncation=True,
        )
        labels = tokenizer(
            text_target=batch["target_text"],
            max_length=max_target_length,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized = ds.map(
        _tokenize,
        batched=True,
        remove_columns=ds["train"].column_names,
    )
    return tokenized


def build_model(model_name: str, use_qlora: bool) -> torch.nn.Module:
    quantization_config = None
    if use_qlora:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto" if use_qlora else None,
    )

    if use_qlora:
        model = prepare_model_for_kbit_training(model)

    if "mbart" in model_name.lower():
        target_modules = ["q_proj", "k_proj", "v_proj", "out_proj"]
    else:
        # mT5, T5, UMT5
        target_modules = ["q", "k", "v", "o"]

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=target_modules,
    )
    model = get_peft_model(model, peft_config)
    model.config.use_cache = False
    model.print_trainable_parameters()
    return model


def parse_label(text: str) -> int | None:
    # Remove T5 sentinel tokens first
    text = re.sub(r"<extra_id_\d+>", "", text)
    
    # Now search for 0 or 1
    m = re.search(r"[01]", text)
    if m:
        return int(m.group(0))
    return None


def compute_binary_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_pos": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_pos": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_pos": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


# def evaluate_with_generation(
#     args,
#     trainer: Seq2SeqTrainer,
#     tokenized_test,
#     raw_test,
#     tokenizer,
#     max_new_tokens: int,
# ) -> tuple[dict[str, float]]:
    
#     out = trainer.predict(tokenized_test, max_new_tokens=max_new_tokens)
#     pred_ids = out.predictions
#     if isinstance(pred_ids, tuple):
#         pred_ids = pred_ids[0]

#     pred_texts = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
#     print("test")
#     print(pred_texts[:5])
#     y_pred = [parse_label(t) for t in pred_texts]
#     y_true = [int(x["label"]) for x in raw_test]
#     metrics = compute_binary_metrics(y_true, y_pred)

#     if args.smoke_test:
#         print("=" * 20)
#         print("PREDICTIONS (SMOKE TEST)")
#         for x in pred_texts[:3]:
#             print(x)
#         print("=" * 20)
#     return metrics

def evaluate_with_generation(
    args,
    trainer: Seq2SeqTrainer,
    tokenized_test,
    raw_test,
    tokenizer,
    max_new_tokens: int,
) -> tuple[dict[str, float]]:
    
    out = trainer.predict(tokenized_test, max_new_tokens=max_new_tokens)
    pred_ids = out.predictions
    if isinstance(pred_ids, tuple):
        pred_ids = pred_ids[0]

    pred_texts = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    print("test")
    print(pred_texts[:5])
    
    y_pred = [parse_label(t) for t in pred_texts]
    y_true = [int(x["label"]) for x in raw_test]
    
    # Check if all predictions are valid
    if None in y_pred:
        print(f"Invalid predictions found: {y_pred.count(None)}/{len(y_pred)}")
        return {}
    
    metrics = compute_binary_metrics(y_true, y_pred)

    if args.smoke_test:
        print("=" * 20)
        print("PREDICTIONS (SMOKE TEST)")
        for x in pred_texts[:3]:
            print(x)
        print("=" * 20)
    return metrics


def maybe_smoke(ds: DatasetDict) -> DatasetDict:
    n_train = min(512, len(ds["train"]))
    n_dev = min(32, len(ds["dev"]))
    n_test = min(32, len(ds["test"]))
    return DatasetDict(
        {
            "train": ds["train"].select(range(n_train)),
            "dev": ds["dev"].select(range(n_dev)),
            "test": ds["test"].select(range(n_test)),
        }
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="google/mt5-base")
    parser.add_argument("--smoke_test", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_idx", type=int, required=True)

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)

    parser.add_argument("--max_source_length", type=int, default=512)
    parser.add_argument("--max_target_length", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--eval_steps", type=int, default=100)

    parser.add_argument("--use_qlora", required=True, type=int)
    parser.add_argument("--save_total_limit", type=int, default=2)
    args = parser.parse_args()

    args.use_qlora = bool(args.use_qlora)
    args.smoke_test = bool(args.smoke_test)

    set_seed(args.seed)
    random.seed(args.seed)

    base_dir = os.getenv("BASE_WCD", os.getcwd())
    args.output_dir = os.path.join(base_dir, "scripts" , "rebuttal", "results")
    os.makedirs(args.output_dir, exist_ok=True)
    device = get_device()

    print("=" * 60)
    print("EXPERIMENT CONFIGURATION")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Requested QLoRA: {args.use_qlora}")
    print("=" * 60)

    ds = load_language_splits(base_dir=base_dir, lang=args.lang)
    if args.smoke_test:
        print("=" * 60)
        print("RUNNING SMOKE TEST")
        print("=" * 60)

        ds = maybe_smoke(ds)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    tokenized = tokenize_dataset(
        args=args,
        ds=ds,
        tokenizer=tokenizer,
        lang=args.lang,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
    )

    use_qlora = bool(args.use_qlora and torch.cuda.is_available())
    model = build_model(model_name=args.model_name, use_qlora=use_qlora)

    os.makedirs(args.output_dir, exist_ok=True)
    training_args = Seq2SeqTrainingArguments(
        output_dir=None,
        eval_steps=args.eval_steps,
        logging_strategy="steps",
        logging_steps=100,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        predict_with_generate=True,
        generation_max_length=args.max_new_tokens,
        save_total_limit=args.save_total_limit,
        bf16=False,
        fp16=False,
        report_to="none",
        remove_unused_columns=True,
    )

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["dev"],
        processing_class=tokenizer,
        data_collator=collator,
    )

    trainer.train()

    metrics = evaluate_with_generation(
        args=args,
        trainer=trainer,
        tokenized_test=tokenized["test"],
        raw_test=ds["test"],
        tokenizer=tokenizer,
        max_new_tokens=args.max_new_tokens,
    )

    print("=" * 20)
    print("TEST METRICS")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    print("=" * 20)

    results = {
        "run_idx": args.run_idx,
        "lang": args.lang,
        "model": args.model_name,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "metrics": metrics,
    }

    metrics_path = os.path.join(args.output_dir, f"{args.lang}_{MODEL_DICT[args.model_name]}_{args.run_idx}.jsonl")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
