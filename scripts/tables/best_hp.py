import os
import re
import json
import glob
from collections import defaultdict

BASE_DIR = os.getenv("BASE_WCD") 
DATA_DIR = os.path.join(BASE_DIR, "data/sets/main")
SLM_CLASSY_DIR = os.path.join(BASE_DIR, "data/models/slm/test")

def load_slms_classy():
    paths = glob.glob(os.path.join(SLM_CLASSY_DIR, "model_*"))
    classy_dir = [p for p in paths if re.search(r"model_\d+$", os.path.basename(p))]
    
    rows = defaultdict(list)
    for path in classy_dir:
        with open(os.path.join(path, "meta.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
        lang = meta['data'][:2]
        model_number = meta['model_number'] 
        model_name = meta['model'].split("/")[-1] + f" (ch)"
        test_accuracy = meta['test_metrics']['eval_accuracy']
        
        try:
            batch = meta['training_args']['per_device_train_batch_size']
            grad_acc = meta['training_args']['gradient_accumulation_steps']
            lr = meta['training_args']['learning_rate']
            wd = meta['training_args']['weight_decay']
            epochs = meta['training_args']['num_train_epochs']
            warmup_r = meta['training_args']['warmup_ratio']
            max_grad_norm = meta['training_args']['max_grad_norm']
            hps = [epochs, batch, grad_acc, lr, wd, warmup_r, max_grad_norm]
        except:
            hps = []

        rows[lang].append([model_number, model_name, test_accuracy, hps])
    
    
    # sort rows
    for lang in rows:
        rows[lang] = sorted(rows[lang], key=lambda x: x[2], reverse=True)
    return rows


def main():

    rows_classy = load_slms_classy()

    for key, value in rows_classy.items():
        print("\n", key, f"[epochs, batch, grad_acc, lr, wd, warmup_r, max_grad_norm]")
        for model in value[:5]:
            print("\t", model)



if __name__ == "__main__":
    main()