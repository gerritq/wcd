import os
import re
import json
import glob
from collections import defaultdict
import sys
import matplotlib.pyplot as plt
import argparse

BASE_DIR = os.getenv("BASE_WCD")
EXP4_DIR = os.path.join(BASE_DIR, "data/exp4")

def get_metas(lang_dir: str):
    all_metas = [
        os.path.join(lang_dir, f)
        for f in os.listdir(lang_dir)
        if os.path.isfile(os.path.join(lang_dir, f)) and f.endswith(".json")
    ]

    evals = []

    for meta_path in all_metas:
        print(meta_path)
        if not os.path.basename(meta_path).startswith("meta"):
            continue
        with open(meta_path, "r") as f:
            meta = json.load(f)

        if meta['atl'] == 1 and meta['model_type'] == "slm":
            name = "atl"
        elif meta['atl'] == 0 and meta['model_type'] == "slm":
            name = "van"
        if meta['model_type'] == "classifier":
            name = "clf"
                
        # if met path is meta_1
        if 'meta_1' in os.path.basename(meta_path):
            training_size = 0
        else:
            training_size = meta['training_size']

        title =f"M={name} - ALT={meta['atl']} - TRAIN={','.join(meta['training_langs'])} - TEST={meta['test_lang']}"

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

        evals.append((training_size, best_dev, test_acc))
    return evals, title

def plot_training_vs_accuracy(evals, title, out_path: str):
    """Plot training size vs dev/test accuracy for all runs."""
    plt.figure()

    # dev accuracy
    xs = [ts for ts, _, _ in evals]
    ys = [test for _, _, test in evals]
    pairs = sorted(zip(xs, ys), key=lambda x: x[0])
    xs, ys = zip(*pairs)
    plt.plot(xs, ys, marker="o", label="Test Accuracy")
    plt.title(title)
    plt.xlabel("Training size")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def main():

    # find all lang folder in exp4
    all_langs = [
        f for f in os.listdir(EXP4_DIR)
        if os.path.isdir(os.path.join(EXP4_DIR, f))
        if f != "smoke_test"
    ]
    for lang in all_langs:
        # check if dir exists
        lang_dir = os.path.join(EXP4_DIR, f"{lang}")
        print(lang_dir)

        all_runs = [
        f for f in os.listdir(lang_dir)
        if os.path.isdir(os.path.join(lang_dir, f))
        if f != "smoke_test"
        ]
        for run in all_runs:
            run_dir = os.path.join(lang_dir, run)
            if not os.path.exists(run_dir):
                print(f"Directory {run_dir} does not exist. Skipping.")
                continue
                
            
            evals, title = get_metas(run_dir)
            print(f"Language: {lang}, Runs found: {list(evals)}")

            plot_training_vs_accuracy(evals, title, f"plots/cl_{lang}_{run}.png")


if __name__ == "__main__":
    main()

