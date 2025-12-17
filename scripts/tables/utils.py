
from pathlib import Path
import json

LANGS = {"high": ["en", "pt", "de", "ru", "it", "vi", "tr", "nl"],
         "medium": ["uk", "ro", "id", "bg", "uz"],
         "low": ["no", "az", "mk", "hy", "sq"],
         }

LANG_ORDER = LANGS["high"] + LANGS["medium"] + LANGS["low"]

MODEL_DISPLAY_NAMES = {"meta-llama/Llama-3.1-8B": "Llama3-8B",
                      "meta-llama/Llama-3.1-8B-Instruct": "Llama3-8B", # same for cls and slm
                       "Qwen/Qwen3-8B-Base": "Qwen3-8B",
                       "Qwen/Qwen3-8B": "Qwen3-8B",
                       "CohereLabs/aya-expanse-8b": "Aya-8b",
                       "microsoft/mdeberta-v3-base": "mDeberta-base",
                       "microsoft/deberta-v3-large": "mDeberta-large",
                       "google-bert/bert-base-multilingual-uncased": "mBert",
                       "FacebookAI/xlm-roberta-base": "XLM-R-base",
                       "FacebookAI/xlm-roberta-large": "XLM-R-large",
                       "openai/gpt-4o-mini": "GPT-4o-mini"
                        }
                        
def load_metrics(path):
    """"Load a single meta_file"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def find_best_metric_from_hyperparameter_search(all_meta_file_paths: Path, metric: str) -> dict:
    """Takes a meta file path and returns the best test metric based on the best dev metric."""
        
    best_dev_metric = float('-inf')
    best_test_metric = float('-inf')

    all_meta_file_paths = [str(f) for f in all_meta_file_paths]
    for meta_file_path in all_meta_file_paths:
        # load the meta file
        meta = load_metrics(meta_file_path)

        # loop epochs
        for dev_metrics, test_metrics in zip(meta.get("dev_metrics", []), meta.get("test_metrics", [])):
            dev_value = dev_metrics.get("metrics", {}).get(metric)
            test_value = test_metrics.get("metrics", {}).get(metric)

            if dev_value is not None and dev_value > best_dev_metric:
                best_dev_metric = dev_value
                best_test_metric = test_value
    
    return best_test_metric