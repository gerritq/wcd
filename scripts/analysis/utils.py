import random
import evaluate
import json

# https://huggingface.co/collections/meta-llama/llama-33-67531d5c405ec5d08a852000
# https://huggingface.co/collections/Qwen/qwen3-67dd247413f0e2e4f653967f

accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

MODEL_MAPPING =  {
    "xlm-r": "FacebookAI/xlm-roberta-base",
    "mBert": "google-bert/bert-base-multilingual-uncased",
    "llama3_1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama3_3b": "meta-llama/Llama-3.2-3B-Instruct",
    "llama3_8b": "meta-llama/Llama-3.1-8B-Instruct",
    "llama3_70b": "meta-llama/Llama-3.3-70B-Instruct",
    "qwen_06b": "Qwen/Qwen3-0.6B",
    "qwen3_4b": "Qwen/Qwen3-4B-Instruct-2507",
    "qwen3_8b": "Qwen/Qwen3-8B",
    "qwen3_30b": "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "qwen3_32b": "Qwen/Qwen3-32B"
    }

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]
    f1  = f1_metric.compute(predictions=preds, references=labels, average="binary")["f1"]
    return {"accuracy": acc, "f1": f1}

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

def tokenise_data(ds, tokenizer, context_bool: bool):
    sep_token = tokenizer.sep_token or "[SEP]"
    def format_input(example):
        # concat input
        claim = example["claim"].strip()
        if context_bool:
            section = example['section'].strip()
            context = example['context'].strip()
            
            # only use non-empty parts
            parts = [p for p in [section, context, claim] if p]
            return f" {sep_token} ".join(parts)
        # claim only, no context
        else:
            return claim

    def tokenize(example):
        return tokenizer(
            format_input(example),
            padding="max_length",
            truncation=True,
        )

    return ds.map(tokenize, batched=False)