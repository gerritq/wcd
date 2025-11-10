import os
import re
import json
import glob
from collections import defaultdict
import sys

type_ = sys.argv[1]
lang = sys.argv[2]

BASE_DIR = os.getenv("BASE_WCD") 
DATA_DIR = os.path.join(BASE_DIR, "data/sets/main")
MODEL_DIR = os.path.join(BASE_DIR, f"data/models/{type_}/{lang}")

def load_slms_classy():
    paths = glob.glob(os.path.join(MODEL_DIR, "model_*"))
    model_dir = [p for p in paths if re.search(r"model_\d+$", os.path.basename(p))]
    print(model_dir)
    
    rows = defaultdict(list)
    for path in model_dir:
        try:
            with open(os.path.join(path, "meta.json"), "r", encoding="utf-8") as f:
                meta = json.load(f)
        except:
            print(f"No meta for {path}")
            continue
        lang = meta['lang']
        model_number = meta['model_number'] 
        model_name = meta['model_name']
        try:
            test_accuracy = meta['test_metrics']['accuracy']
        except:
            test_accuracy = meta['test_metrics']['eval_accuracy']
        
        hps = {}
        try:
            hps['epochs'] = meta['epochs']
            hps['learning_rate'] = meta['learning_rate']
            hps['batch_size'] = meta['batch_size']
            hps['max_grad_norm'] = meta['max_grad_norm']
        except:
            pass

        rows[lang].append([model_number, model_name, test_accuracy, hps])
    
    
    # sort rows
    for lang in rows:
        rows[lang] = sorted(rows[lang], key=lambda x: x[2], reverse=True)
    return rows


def main():

    rows_models = load_slms_classy()

    for key, value in rows_models.items():
        print("\n", key)
        for model in value[:15]:
            print("\t", model)



if __name__ == "__main__":
    main()