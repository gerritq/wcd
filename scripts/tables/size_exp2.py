import os
import re
import json
import glob
from collections import defaultdict
import sys
import matplotlib.pyplot as plt
import argparse

BASE_DIR = os.getenv("BASE_WCD")
EXP2_DIR = os.path.join(BASE_DIR, "data/exp2")

def get_metas(lang_dir: str):
    all_metas = [
        os.path.join(lang_dir, f)
        for f in os.listdir(lang_dir)
        if os.path.isfile(os.path.join(lang_dir, f)) and f.endswith(".json")
    ]

    models = {}

    for meta_path in all_metas:
        with open(meta_path, "r") as f:
            meta = json.load(f)

        if meta['atl'] == 1 and meta['model_type'] == "slm":
            name = "atl"
        elif meta['atl'] == 0 and meta['model_type'] == "slm":
            name = "van"
        if meta['model_type'] == "classifier":
            name = "clf"

        training_size = meta['training_size']    
        

        best_dev = -1.0
        best_dev_epoch = None
        for x in meta['dev_metrics']:
            dev_acc = x['metrics']['accuracy']
            if dev_acc > best_dev:
                best_dev = dev_acc
                best_dev_epoch = x['epoch']

        test_acc = None
        for x in meta['test_metrics']:
            if x['epoch'] == best_dev_epoch:
                test_acc = x['metrics']['accuracy']
                break

        if name not in models:
            models[name] = []
        models[name].append((training_size, best_dev, test_acc))

    return models

def plot_training_vs_accuracy(runs, out_path: str):
    """Plot training size vs dev/test accuracy for all runs."""
    plt.figure()

    # dev accuracy
    for model, r in runs.items():
        xs = [ts for ts, _, _ in r]
        ys = [test for _, _, test in r]
        pairs = sorted(zip(xs, ys), key=lambda x: x[0])
        xs, ys = zip(*pairs)
        plt.plot(xs, ys, marker="o", label=f"{model}")

    plt.xlabel("Training size")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def main():
    LANGS  = ["en", "nl", "it" ,"bg"]
    for lang in LANGS:
        # check if dir exists
        lang_dir = os.path.join(EXP2_DIR, f"{lang}")
        if not os.path.exists(lang_dir):
            print(f"Directory {lang_dir} does not exist. Skipping.")
            continue
            
        
        models = get_metas(lang_dir)
        print(f"Language: {lang}, Models found: {list(models.keys())}")

        plot_training_vs_accuracy(models, f"plots/ts_{lang}.png")


if __name__ == "__main__":
    main()

