import random

# https://huggingface.co/collections/meta-llama/llama-33-67531d5c405ec5d08a852000
# https://huggingface.co/collections/Qwen/qwen3-67dd247413f0e2e4f653967f

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

def copute_metrics():
    # acc, f1, auroc, prec-roc
    pass

def prepare_monolingual_data():
    pass