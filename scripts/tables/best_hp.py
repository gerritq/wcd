import os
import re
import json
import glob
from collections import defaultdict
import sys

type_ = sys.argv[1]
lang = sys.argv[2]
context = int(sys.argv[3])
assert context in [0, 1], "Context argparse must be binary"
context = bool(context)

BASE_DIR = os.getenv("BASE_WCD") 
DATA_DIR = os.path.join(BASE_DIR, "data/sets/main")
MODEL_DIR = os.path.join(BASE_DIR, f"data/models/{lang}")

def load_best_hps():
    all_metas = [
                os.path.join(MODEL_DIR, f)
                for f in os.listdir(MODEL_DIR)
                if os.path.isfile(os.path.join(MODEL_DIR, f)) and f.endswith(".json")
            ]  
    rows = defaultdict(list)
    for path in all_metas:
        try:
            with open(path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except:
            print(f"No meta for {path}")
            continue
        if meta['model_type'] != type_ or meta['context'] != context:
            continue
        lang = meta['lang']
        model_number = meta['model_number'] 
        model_name = meta['model_name']
        model_context = meta['context']
        
        # test acc
        dev_accuracy = meta['dev_metrics']['accuracy']
        test_accuracy = meta['test_metrics']['accuracy']
        

        
        hps = {}
        try:
            hps['epochs'] = meta['epoch']
            hps['learning_rate'] = meta['learning_rate']
            hps['batch_size'] = meta['batch_size']
            hps['max_grad_norm'] = meta['max_grad_norm']
        except:
            pass
        

        rows[lang].append({"model_number": model_number,
                           "model_name": model_name,
                           "model_context": model_context,
                           "test_accuracy": round(test_accuracy, 4),
                           "dev_accuracy": round(dev_accuracy, 4),
                        #    "dev_loss": round(dev_loss, 4) if dev_loss else dev_loss,
                           "hp": hps})
    print(rows)
    for lang in rows:
        rows[lang] = sorted(rows[lang], key=lambda x: x["test_accuracy"], reverse=True)

    return rows


def main():

    rows_models = load_best_hps()

    for key, value in rows_models.items():
        print("\n", key)
        for model in value[:20]:
            hp = model.pop("hp")
            print("\t", model)
            print("\t", hp)

if __name__ == "__main__":
    main()