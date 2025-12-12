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
EXP2_DIR = os.path.join(BASE_DIR, "data/exp2")

LAST_EPOCH = True
METRIC="f1"
SHOTS = [0, 50, 100, 200, 400, 600, 800]
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


LANG_MAPPING = {"high": ["tr"],
                "low": ["mk", "sq", "no", "az", "hy"],
                "mid": ["uz", "uk", "ro", "id", "bg"],
                }

LANG_ORDER = LANG_MAPPING["low"] + LANG_MAPPING["mid"] + LANG_MAPPING["high"]

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
            best_dev = meta['dev_metrics'][-1]['metrics'][METRIC]
            test_acc = meta['test_metrics'][-1]['metrics'][METRIC]
            evals.append((training_size, best_dev, test_acc))
        else:
            best_epoch = find_best_epoch(meta, metric=METRIC)
            evals.append((training_size, best_epoch['dev_metric'], best_epoch['test_metric']))
    
    run_dict['evals'] = evals
    return run_dict

def latex_table_2_stage_ft(all_evals: list):
    """
    Produce a table summarizing the results for a single target language.
    """
    total_runs = len(SHOTS)

    # print LaTeX table
    table = "\n\n"
    colspec = "lll" + "c" * (total_runs)

    header = "\\textbf{Target} & \\textbf{Train} & \\textbf{Model} & " + " & ".join([f"\\textbf{{{shot}-Shot}}" for shot in SHOTS]) + " \\\\\n" 
    table += "\\begin{tabular}{" + colspec + "}\n"
    table += "\\toprule\n"
    table += header
    table += "\\midrule\n"

    resource_level = ""
    for run_dict in all_evals:
        
        training_langs = run_dict['training_langs']
        evals = run_dict['evals']
        xs = [ts for ts, _, _ in evals]
        ys = [test for _, _, test in evals]
        xs, ys = zip(*sorted(zip(xs, ys), key=lambda x: x[0])) 

        differences = []
        for i in range(1, len(ys)):
            diff = ys[i] - ys[0]
            differences.append(diff)

        model_type = run_dict['model_type']
        model_name = run_dict['model_name']
        model_name = f"{MODEL_DISPLAY_NAMES[model_name]} ({model_type}) "
        
        if resource_level == "" and run_dict['target_lang'] in LANG_MAPPING['low']:
            table += f"\\textit{{Low-resource}} " + "& " * (total_runs + 2) + " \\\\\n"
            resource_level = "low"
        elif resource_level == "low" and run_dict['target_lang'] in LANG_MAPPING['mid']:
            table += f"\\midrule\n"
            table += f"\\textit{{Mid-resource}} " + "& " * (total_runs + 2) + " \\\\\n"
            resource_level = "mid"
        elif resource_level == "mid" and run_dict['target_lang'] in LANG_MAPPING['high']:
            table += f"\\midrule\n"
            table += f"\\textit{{High-resource}} " + "& " * (total_runs + 2) + " \\\\\n"
            resource_level = "high"
        
        table += f"{run_dict['target_lang'].upper()} & " + ", ".join(training_langs) + f" & {model_name} "
        for shot in SHOTS:
            if shot in xs:
                idx = xs.index(shot)
                metric = ys[idx] * 100
                if shot == 0:
                    table += f" & {metric:.2f}"
                else: 
                    diff = differences[idx - 1] * 100
                    table += f" & {metric:.2f} (\\posneg{{{diff:.2f}}})"
            else:
                table += " & -"
        table += " \\\\\n"

        resource_level = "mid" if run_dict['target_lang'] in LANG_MAPPING['mid'] else "low"

    table += "\\bottomrule\n"
    table += "\\end{tabular}\n" 

    print("\n\n")
    print(table) 
    print("\n\n")

def collect_all_runs() -> list[str]:
    
    select_runs = []
    # find all lang folder
    all_langs = [
        f for f in os.listdir(EXP2_DIR)
        if os.path.isdir(os.path.join(EXP2_DIR, f))
        if f != "smoke_test"
    ]
    for lang in all_langs:

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
            
            if not os.path.exists(meta_1_path):
                print(f"Meta file {meta_1_path} does not exist. Skipping.")
                continue
            meta_1 = load_meta(meta_1_path)
            
            select_runs.append({"run_dir": run_dir,
                                "model_type": meta_1['model_type'],
                                "model_name": meta_1['model_name'],
                                "training_langs": meta_1['training_langs'],
                                "target_lang": meta_1['test_lang']})
    return select_runs

def main():


    all_runs  = collect_all_runs()
    print(f"Total runs collected: {len(all_runs)}")
    # sort by lang order
    all_runs = sorted(all_runs, key=lambda x: (LANG_ORDER.index(x['target_lang']), x['model_type']))

    all_run_dicts = [get_metas(run_dict=run_dict) for run_dict in all_runs]
    latex_table_2_stage_ft(all_evals=all_run_dicts)

if __name__ == "__main__":
    main()
