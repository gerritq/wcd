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

def get_metas(run_dir: str):
    all_metas = [
        os.path.join(run_dir, f)
        for f in os.listdir(run_dir)
        if os.path.isfile(os.path.join(run_dir, f)) and f.endswith(".json")
    ]

    all_metas = sorted(
        all_metas,
        key=lambda p: int(os.path.basename(p).split("_")[-1].split(".")[0])
    )

    results = []

    for meta_path in all_metas:
        with open(meta_path, "r") as f:
            data = json.load(f)

        best_dev = -1.0
        selected_test = None
        training_size = None

        for key, epoch_data in data.items():
            
            if not key.startswith("epoch_"):
                continue

            explanation = epoch_data['explanation']

            dev_acc = epoch_data.get("dev_metrics", {}).get("accuracy")
            if dev_acc is None:
                print("Skip")
                continue

            if training_size is None:
                training_size = epoch_data.get("training_size")

            if dev_acc > best_dev:
                best_dev = dev_acc
                selected_test = epoch_data.get("test_metrics", {}).get("accuracy")

            

        results.append((training_size, best_dev, selected_test))

    return [results, explanation]

def plot_training_vs_accuracy(runs, out_path: str):
    """Plot training size vs dev/test accuracy for all runs."""
    plt.figure()

    # dev accuracy
    for run_name, entries in runs.items():
        # unpack and sort by training size
        results = entries[0] 
        xs = [ts for ts, _, _ in results]
        ys = [test for _, _, test in results]
        pairs = sorted(zip(xs, ys), key=lambda x: x[0])
        xs, ys = zip(*pairs)
        plt.plot(xs, ys, marker="o", label=f"R: {entries[1]}")

    plt.xlabel("Training size")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def main():
    lang = 'nl_en'
    # keep_runs = [4, 5, 7]
    keep_runs = [12, 13, 14]
    runs = {}
    all_folders = [
                os.path.join(EXP2_DIR, f) for f in os.listdir(EXP2_DIR)
                if os.path.isdir(os.path.join(EXP2_DIR, f)) and
                f.endswith(lang)
                ]


    for language_dir in all_folders:
        all_runs = sorted(
            [os.path.join(language_dir, f) for f in os.listdir(language_dir) 
                          if os.path.isdir(os.path.join(language_dir, f))],
            key=lambda p: int(os.path.basename(p).split("_")[-1])
        )
        for run_dir in all_runs:
            run_name = os.path.basename(run_dir).split("_")[-1]
            if int(run_name) in keep_runs:
                runs[run_name] = get_metas(run_dir)
    
    # runs = {k: v for k, v in runs.items() if int(k.split("_")[-1]) in keep_runs}
    plot_training_vs_accuracy(runs, f"plots/runs{''.join([str(x) for x in keep_runs])}.png")
    print(runs)

if __name__ == "__main__":
    main()

