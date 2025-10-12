import json
import torch
import argparse
import os
import time
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    set_seed,
    Trainer,
)
from utils import (
                    MODEL_MAPPING, 
                    compute_metrics, 
                    append_meta_file, 
                    get_model_number,
                    tokenise_data
)

parser = argparse.ArgumentParser()
parser.add_argument("--model_number", type=str, required=True)
parser.add_argument("--languages", nargs="+", required=True)
args = parser.parse_args()

BASE_DIR = os.getenv("BASE_WCD")
DATA_DIR = os.path.join(BASE_DIR, "data/sets")
MODEL_DIR = os.path.join(BASE_DIR, "data/models/plm")
METRICS_DIR = os.path.join(BASE_DIR, "data/metrics/plm")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

set_seed(42)

def main():
    start = time.time()

    model_dir = os.path.join(MODEL_DIR, args.model_number) 
    # Load meta
    with open(os.path.join(model_dir, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    
    # Load tokeniser
    tokenizer = AutoTokenizer.from_pretrained(meta['model'])

    for lang in args.languages:
        # load test data
        test = load_from_disk(os.path.join(DATA_DIR, lang))['test']
        test_tok = tokenise_data(test, tokenizer, meta['context'])

        # pred and eval
        trainer = Trainer(model=model, compute_metrics=compute_metrics)
        results = trainer.evaluate(test_tok)
    
        meta.update({f"eval_results_{lang}": results})

        # save meta
        out_path = os.path.join(METRICS_DIR, f"{meta['data']}_2_{lang}_{meta['model_number']}.json")
        with open(out_path, "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()