import os
from pdb import run
import re
import json
import glob
from collections import defaultdict
import sys
import matplotlib.pyplot as plt
import argparse

BASE_DIR = os.getenv("BASE_WCD")
EXP2_DIR = os.path.join(BASE_DIR, "data/exp2")

LANG_METRICS_MAP = {"bg": {"mono": 0.80, "size": 0.65},}


def load_meta(meta_path: str):
    with open(meta_path, "r") as f:
        meta = json.load(f)
    return meta

def get_metas(run_dict: dict) -> list[tuple[int, float, float]]:

    evals = []

    all_metas = glob.glob(os.path.join(run_dict["run_dir"], "meta_*.json"))

    for meta_path in all_metas:
        print(meta_path)

        meta = load_meta(meta_path)

        # if met path is meta_1
        if 'meta_1' in os.path.basename(meta_path):
            training_size = 0
        else:
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

        evals.append((training_size, best_dev, test_acc))
    
    run_dict['evals'] = evals
    return run_dict

def plot_2_stage_ft(all_evals: list, configs: dict):
    """Plot training size vs dev/test accuracy for all runs."""
    
    title = f"Test Language: {configs['test_lang']}"
    out_path = os.path.join("plots", f"exp2_{configs['test_lang']}.png")
    plt.figure()
    
    for run_dict in all_evals:
        training_langs = run_dict['training_langs']
        evals = run_dict['evals']
        xs = [ts for ts, _, _ in evals]
        ys = [test for _, _, test in evals]
        xs, ys = zip(*sorted(zip(xs, ys)))
        plt.plot(xs, ys, marker="o", label="1st-stage Training: " + ", ".join(training_langs))
    
    plt.axhline(
        y=LANG_METRICS_MAP[configs['test_lang']]['mono'],
        color='black',
        linestyle='--',
        label='Monolingual Baseline',
    )

    plt.axhline(
        y=LANG_METRICS_MAP[configs['test_lang']]['size'],
        color='grey',
        linestyle='--',
        label=f"800-Sample {configs['test_lang']} Baseline",
    )
    
    plt.xticks([0, 200, 400, 600, 800])
    plt.title(title)
    plt.xlabel("Target-Language Fine-Tuning Samples")
    plt.ylabel("Accuracy")
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
        frameon=True, 
    )
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(out_path, dpi=300)
    plt.close()

def select_runs(configs: dict) -> list[str]:
    
    select_runs = []
    # find all lang folder
    all_langs = [
        f for f in os.listdir(EXP2_DIR)
        if os.path.isdir(os.path.join(EXP2_DIR, f))
        if f != "smoke_test"
    ]
    for lang in all_langs:
        
        if lang != configs["test_lang"]:
            continue

        # check if dir exists
        lang_dir = os.path.join(EXP2_DIR, f"{lang}")
        print(lang_dir)

        all_runs = [
            f for f in os.listdir(lang_dir)
            if os.path.isdir(os.path.join(lang_dir, f))
            if f != "smoke_test"
        ]


        for run in all_runs:
            # load meta_1
            run_dir = os.path.join(lang_dir, run)
            if not os.path.exists(run_dir):
                print(f"Directory {run_dir} does not exist. Skipping.")
                continue
            meta_1_path = os.path.join(run_dir, "meta_1.json")
            meta_1 = load_meta(meta_1_path)
            
            # Skip if not fitting configs
            if meta_1['model_type'] != configs["model_type"]:
                continue
            if meta_1['training_langs'] not in configs["training_langs"]:
                continue

            select_runs.append({"run_dir": run_dir,
                                "training_langs": meta_1['training_langs'],})
    return select_runs

def main():
    configs = {"test_lang": "bg",
               "training_langs": [["ru"], ["en", "ru"]],
               "model_type": "slm",}
    
    selected_runs = select_runs(configs=configs)
    print("="*20)
    print(f"Selected {len(selected_runs)} runs:")
    for run_dict in selected_runs:
        print(run_dict["run_dir"])
    print("="*20)
    all_evals = []
    for run_dict in selected_runs:
        evals = get_metas(run_dict=run_dict)
        print(evals)
        all_evals.append(evals)
    print(all_evals)
    plot_2_stage_ft(all_evals=all_evals, configs=configs)


if __name__ == "__main__":
    main()

