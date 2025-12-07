import os
from pdb import run

import json
import glob
from collections import defaultdict
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

# ----------------------------------------------------------------------
# Configs
# ----------------------------------------------------------------------
BASE_DIR = os.getenv("BASE_WCD")
EXP1_DIR = os.path.join(BASE_DIR, "data/exp1")
EXP2_DIR = os.path.join(BASE_DIR, "data/exp2")

LAST_EPOCH = True

def collect_monoliongual_metrics(configs: dict,) -> dict:
    lang_dir = os.path.join(EXP1_DIR, configs["test_lang"])
    # get all meta files
    all_lang_runs = [f for f in os.listdir(lang_dir)
                      if os.path.isdir(os.path.join(lang_dir, f))]
    evals = []
    for run_dir in all_lang_runs:
        meta_paths = glob.glob(os.path.join(lang_dir, run_dir, "meta_*.json"))
        
        # skipt those; as we only look for one meta per run
        if len(meta_paths) == 0 or len(meta_paths) > 1:
            continue
        
        meta = load_meta(meta_paths[0])
        
        if "training_size" not in meta:
            continue

        if meta['training_size'] in [200, 400, 600, 800] and meta['model_type'] in configs['model_type']:            
            
            if LAST_EPOCH:
                evals.append((meta['training_size'],
                              meta['dev_metrics'][-1]['metrics']['accuracy'],
                              meta['test_metrics'][-1]['metrics']['accuracy'],))
            else:
                best_epoch = find_best_epoch(meta, metric="accuracy")
                evals.append((meta['training_size'],
                              best_epoch['dev_metric'],
                              best_epoch['test_metric'],))
            
    evals = sorted(evals, key=lambda x: x[0])
    out = {"training_langs": [configs["test_lang"]], "evals": evals, "run_dir": lang_dir}
    print("="*20)
    print("MONOLINGUAL EVALS")
    print(out)
    print("="*20)
    return out

def collect_best_monolingual_baseline(configs: dict,) -> dict:
    lang_dir = os.path.join(EXP1_DIR, configs["test_lang"])
    # get all meta files
    all_lang_runs = [f for f in os.listdir(lang_dir)
                      if os.path.isdir(os.path.join(lang_dir, f))]
    
    best_acc = 0.0
    for run_dir in all_lang_runs:
        meta_paths = glob.glob(os.path.join(lang_dir, run_dir, "meta_*.json"))
        for meta_path in meta_paths:
            meta = load_meta(meta_path)

            if meta['model_type'] in configs['model_type']:
                if meta['training_size'] == 5000: 
                    meta = load_meta(meta_path)
                    best_epoch = find_best_epoch(meta, metric="accuracy")
                    test_acc = best_epoch['test_metric']
                    if test_acc > best_acc:
                        best_acc = test_acc
    configs['mono_baseline'] = best_acc
    return configs

def load_meta(meta_path: str):
    with open(meta_path, "r") as f:
        meta = json.load(f)
    return meta

def find_best_epoch(meta: dict, metric: str) -> dict:
    best_epoch = None
    best_dev_metric = float("-inf")

    dev_metrics = meta.get("dev_metrics", [])
    test_metrics = meta.get("test_metrics", [])

    for dev_entry in dev_metrics:
        epoch = dev_entry["epoch"]
        dev_metric = dev_entry["metrics"][metric]

        if dev_metric > best_dev_metric:
            test_entry = next(
                (t for t in test_metrics if t["epoch"] == epoch),
                None
            )
            if test_entry is None:
                continue

            best_dev_metric = dev_metric
            best_epoch = {
                "epoch": epoch,
                "dev_metric": dev_metric,
                "test_metric": test_entry["metrics"][metric],
            }

    return best_epoch

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

        if LAST_EPOCH:
            best_dev = meta['dev_metrics'][-1]['metrics']['accuracy']
            test_acc = meta['test_metrics'][-1]['metrics']['accuracy']
            evals.append((training_size, best_dev, test_acc))
        else:
            best_epoch = find_best_epoch(meta, metric="accuracy")
            evals.append((training_size, best_epoch['dev_metric'], best_epoch['test_metric']))
    
    run_dict['evals'] = evals
    return run_dict

def plot_2_stage_ft(all_evals: list, configs: dict):
    """Plot training size vs dev/test accuracy for all runs."""
    
    title = f"Test Language: {configs['test_lang']}"
    model = "slm" if "slm" in configs['model_type'] else "clf"
    out_path = os.path.join("plots", f"exp2_{configs['test_lang']}_{model}.png")
    plt.figure()
    
    for run_dict in all_evals:
        training_langs = run_dict['training_langs']
        evals = run_dict['evals']
        xs = [ts for ts, _, _ in evals]
        ys = [test for _, _, test in evals]
        xs, ys = zip(*sorted(zip(xs, ys)))
        plt.plot(xs, ys, marker="o", label="1st-stage Training: " + ", ".join(training_langs))
    
    plt.axhline(
        y=configs['mono_baseline'],
        color='gray',
        linestyle='--',
        label='Monolingual Baseline',
    )

    # plt.axhline(
    #     y=configs['size_baseline'],
    #     color='grey',
    #     linestyle='--',
    #     label=f"800-Sample {configs['test_lang']} Baseline",
    # )
    
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
            if meta_1['model_type'] not in configs["model_type"]:
                continue
            if meta_1['training_langs'] not in configs["training_langs"]:
                continue
            if ("atl" in configs and meta_1['atl'] != configs['atl']):
                continue


            select_runs.append({"run_dir": run_dir,
                                "training_langs": meta_1['training_langs'],})
    return select_runs

def main():
    configs = [{"test_lang": "bg",
               "training_langs": [["en"], ["en", "ru"], ["ru"]],
               "model_type": ["clf", "classifier", "cls"]},
              {"test_lang": "no",
               "training_langs": [["en"], ["en", "nl"], ["nl"]],
               "model_type": ["clf", "classifier", "cls"]},
               {"test_lang": "ro",
               "training_langs": [["en"], ["en", "it"], ['it']],
               "model_type": ["clf", "classifier", "cls"]},]

    configs = [{"test_lang": "bg",
               "training_langs": [["en"], ["en", "ru"], ["ru"]],
               "model_type": ["slm"],
               "atl": True},
              {"test_lang": "no",
               "training_langs": [["en"], ["en", "nl"], ["nl"]],
               "model_type": ["slm"],
               "atl": True},
               {"test_lang": "ro",
               "training_langs": [["en"], ["en", "it"], ['it']],
               "model_type": ["slm"],
               "atl": True},]
    
    for config in configs:
        # collec monolignual metrics to get baselines
        config = collect_best_monolingual_baseline(config)
        # collect monoligual model over training sizes
        mono_model = collect_monoliongual_metrics(config)
        print(config)
        selected_runs = select_runs(configs=config)
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
        all_evals.append(mono_model)
        plot_2_stage_ft(all_evals=all_evals, configs=config)


if __name__ == "__main__":
    main()