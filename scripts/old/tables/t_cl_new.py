import os
import re
import json
import glob
from collections import defaultdict
from pathlib import Path
from argparse import Namespace
from utils import load_metrics, find_best_metric_from_hyperparameter_search, LANGS, LANG_ORDER, MODEL_DISPLAY_NAMES

LANG_ORDER = LANGS['medium'] + LANGS['low']
# ------------------------------------------------------------------------------------------
# configs
# ------------------------------------------------------------------------------------------
BASE_DIR = os.getenv("BASE_WCD")
SLM_DIR = os.path.join(BASE_DIR, "data/exp2/eval")

COUNT = defaultdict(list)

CL_TRAINING_SIZES = [0, 50, 100, 250, 500]

def collect_x_shots(configs: dict,
                      rows: dict,
                      meta_tmp: dict,
                      ) -> dict[str, dict]:
    # COLLECT
    model_name = MODEL_DISPLAY_NAMES[meta_tmp["model_name"]]
    variant = ""    
    if meta_tmp["model_type"]  == 'clf':
        variant = "(classifier)"
    elif meta_tmp["model_type"] == "slm" and meta_tmp["atl"]:
        variant = "(ATL)"
    elif meta_tmp["model_type"] == "slm" and not meta_tmp["atl"]:
        variant = "(VAN)"
    else:
        variant = ""

    model_name = f"{model_name} {variant}"

    key = (model_name, meta_tmp["lang_setting"], meta_tmp["source_langs"][0], meta_tmp["lang"], meta_tmp["lower_lr"])  
    if meta_tmp['cl_setting'] == "zero":
        best_metric = meta_tmp["test_metrics_0_shot"][configs['metric']]
    if meta_tmp['cl_setting'] == "few":
        best_metric = meta_tmp["test_metrics"][-1]['metrics'][configs['metric']]

    if meta_tmp['cl_setting'] == "zero":
        rows[key][0].append(best_metric)
    if meta_tmp['cl_setting'] == "few":
        rows[key][int(meta_tmp['training_size'])].append(best_metric)

    return rows


def load_all_models(configs: dict, path: str) -> dict[str, dict]:
    root = Path(path)
    rows = defaultdict(lambda: defaultdict(list))

    # iteratore over all lang dirs
    for lang_dir in root.iterdir():
        if not lang_dir.is_dir():
            # print(f"Skipping non-lang-dir: {lang_dir}")
            continue
    
        # BASICS TO DIRECT
        meta_files = [f for f in lang_dir.iterdir() if f.is_file()]
        
        if len(meta_files) == 0:
            continue

        for meta_tmp in meta_files:
            rows = collect_x_shots(configs=configs, rows=rows, meta_tmp=load_metrics(meta_tmp))
    
    # sort keys
    rows_sorted = dict(sorted(rows.items(), key=lambda x: (x[0][1], x[0][2], x[0][3])))
    
    # sort by shots
    for k, v in rows_sorted.items():
        v_sorted = dict(sorted(v.items(), key=lambda x: (x[0])))
        rows_sorted[k] = v_sorted
        
    # PRRINT
    for k, v in rows_sorted.items():
        print(k, v)

    return rows_sorted

def latex_table(rows: dict):

        
    # print LaTeX table
    table = "\n\n"
    colspec = "l" + "c" * len(CL_TRAINING_SIZES)
    
    header = " Model-Setting-Source-Lang-LL & " + " & ".join([f'\\textbf{{{shots}}}' for shots in CL_TRAINING_SIZES]) + " \\\\"
    table += "\\begin{tabular}{" + colspec + "}\n"
    table += header + "\n"
    table += "\\hline\n"
    
    for key, shot_dict in rows.items():
        model_name, lang_setting, source_lang, lang, lower_lr = key
        row_label = f"{model_name} - {lang_setting} - {source_lang} - {lang} - {'LL' if lower_lr else 'HL'}"
        row = [row_label]
        for shots in CL_TRAINING_SIZES:
            metrics = shot_dict.get(shots, [])
            if metrics:
                avg_metric = sum(metrics) / len(metrics)
                row.append(f"{avg_metric:.2f}")
            else:
                row.append("N/A")
        table += " & ".join(row) + " \\\\\n"
    table += "\\hline\n"
    table += "\\end{tabular}\n"

    print("\n\n")
    print(table)
    print("\n\n")

def main():

    configs = {
        "metric": "f1",
    }

    # load all models
    rows = load_all_models(configs, SLM_DIR)

    # print latex table
    latex_table(rows)
if __name__ == "__main__":
    main()
