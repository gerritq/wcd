import os
import re
import json
import sys
from collections import defaultdict
from pathlib import Path
from argparse import Namespace
from sys import meta_path
from utils import load_metrics, find_best_metric_from_hyperparameter_search, LANGS, LANG_ORDER

# ------------------------------------------------------------------------------------------
# configs
# ------------------------------------------------------------------------------------------
BASE_DIR = os.getenv("BASE_WCD")
SLM_DIR = os.path.join(BASE_DIR, "data/exp1")

MODEL_DISPLAY_NAMES = {"meta-llama/Llama-3.1-8B": "Llama3-8B",
                      "meta-llama/Llama-3.1-8B-Instruct": "Llama3-8B", # same for cls and slm
                       "Qwen/Qwen3-8B-Base": "Qwen3-8B",
                       "Qwen/Qwen3-8B": "Qwen3-8B",
                       "Qwen/Qwen3-8B": "Qwen3-8B",
                       "CohereLabs/aya-expanse-8b": "Aya-8b",
                        }

run_re = re.compile(r"run_\w+")
meta_re = re.compile(r"meta_\d+")


def load_all_models(configs: dict, path: str) -> dict[str, dict]:
    root = Path(path)
    rows = defaultdict(lambda: {l: [] for l in LANG_ORDER})
    count = defaultdict(list)

    # iteratore over all lang dirs
    for lang_dir in root.iterdir():
        if not lang_dir.is_dir():
            continue
        
        # iteratre over runs
        for run_dir in lang_dir.iterdir():
            if not (run_dir.is_dir()):
                continue

            # store best run in a dict
            best_run: dict = None 

            # skip if more than one meta file
            meta_files = [f for f in run_dir.iterdir() if f.is_file()]

            if len(meta_files) == 0:
                # print(f"No meta files found in: {run_dir}")
                continue

            # CASE: seed runs
            if len(meta_files) == 1:
                meta_1 = load_metrics(meta_files[0])
                if meta_1['model_type'] in ['slm']:
                    if meta_1['seed'] in [2025, 2026]:
                        best_run = {
                                    "model_name": meta_1["model_name"],
                                    "prompt_template": meta_1["prompt_template"],
                                    "panel": "ATL" if meta_1["atl"] else "VAN",
                                    "lang": meta_1["lang"],
                                    "test_metric": meta_1["test_metrics"][-1]['metrics'][configs['metric']],
                                    }
                        rows[(best_run["panel"], best_run["model_name"])][best_run["lang"]].append(best_run['test_metric'])
                        count[(meta_1['model_name'], meta_1['lang'], meta_1['atl'])].append((meta_1['seed'], meta_files[0].parent.name))  
                continue

            # collect meta    
            meta_1 = load_metrics(meta_files[0])
            
            # FILTERS
            if (meta_1["model_type"] in ['icl', 'plm', "clf"]):
                # print(f"Skipping due to model type mismatch: {meta['model_type']}")
                continue

            if (meta_1["prompt_template"] != configs['prompt_template']):
                # print(f"Skipping due to prompt template mismatch: {meta['prompt_template']}")
                continue

            
            # CASE: hp search
            # panel generation
            if meta_1["atl"] == True:
                panel = "ATL"
            else:
                panel = "VAN"

            # best metric extraction
            best_metric = find_best_metric_from_hyperparameter_search(all_meta_file_paths=meta_files, metric=configs['metric'])
        

            best_run = {
                        "model_name": meta_1["model_name"],
                        "prompt_template": meta_1["prompt_template"],
                        "panel": panel,
                        "lang": meta_1["lang"],
                        "test_metric": best_metric,
                                    }
            count[(meta_1['model_name'], meta_1['lang'], meta_1['atl'])].append(("hp", meta_files[0].parent.name))
            rows[(panel, meta_1["model_name"])][meta_1["lang"]].append(best_run["test_metric"])
    
    
    # sort reverse rows by panel and model name
    rows = dict(sorted(rows.items(), key=lambda x: (x[0][0], x[0][1]), reverse=True))
    
    print("="*20)
    print("="*20)
    print("OVERVIEW OF COLLECTED FILES")
    print("="*20)
    print("="*20)
    # sortt by lang
    sorted_count = dict(sorted(count.items(), key=lambda x: (x[0][1], x[0][0], x[0][2])))
    for k,v in sorted_count.items():
        print(f"LANG: {k[1]} | MODEL {k[0]} {'ATL' if k[2] else 'VAN'}: {len(v)} runs -> {sorted(v, key=lambda x: str(x[0]))}")
        if len(v) > 3:
            print("WARNING: TOO MANY RUNS!")
        if set([run[0] for run in v]) != set([2025, 2026, 'hp']):
            print("WARNING: INCORRECT NUMBER OF RUNS!")
        print("")
    print("="*20)

    # create averages
    out_rows = defaultdict(dict)
    for (panel, model_name), metrics in rows.items():
        for lang, v in metrics.items():
            if isinstance(v, list) and len(v) == 3:
                avg_v = sum(v) / len(v)
                out_rows[(panel, model_name)][lang] = avg_v
            else:
                out_rows[(panel, model_name)][lang] = None
    return out_rows

def latex_table(rows):

    # unique models
    models = set()
    for (panel, model_name) in rows.keys():
        models.add(model_name)
    n_unique_models = len(models)

    lang_max = {l: float('-inf') for l in LANG_ORDER}
    lang_second_max = {l: float('-inf') for l in LANG_ORDER}

    for l in LANG_ORDER: 
        lang_max[l] = float('-inf')
        lang_second_max[l] = float('-inf')

        for (panel, model_name), metrics in rows.items():
            for lang, v in metrics.items():
                if v is None:
                    continue
                # update max and second max in one go
                if v > lang_max[lang]:
                    lang_second_max[lang] = lang_max[lang]
                    lang_max[lang] = v
                elif v > lang_second_max[lang]:
                    lang_second_max[lang] = v

    # print LaTeX table
    table = "\n\n"
    colspec = "ll" + "c" * (len(LANG_ORDER) + 3)
    
    header = " & & \\multicolumn{" + str(len(LANGS['high'])+1) + "}{c}{\\textbf{High Resource}} & \\multicolumn{" + str(len(LANGS['medium'])+1) + "}{c}{\\textbf{Medium Resource}} & \\multicolumn{" + str(len(LANGS['low'])+1) + "}{c}{\\textbf{Low Resource}} \\\\"
    header += "\\cmidrule(lr){3-" + str(len(LANGS['high'])+3) + "}  \\cmidrule(lr){" + str(len(LANGS['high'])+4) + "-" + str(len(LANGS['high'])+len(LANGS['medium'])+4) + "}  \\cmidrule(lr){" + str(len(LANGS['high'])+len(LANGS['medium'])+5) + "-" + str(len(LANGS['high'])+len(LANGS['medium'])+len(LANGS['low'])+5) + "}"
    header += "\\textbf{Loss} & \\textbf{Model} & " + " & ".join([f"\\textbf{{{l}}}" for l in LANGS["high"]]) + "& \\textbf{Avg}" + " & " + " & ".join([f"\\textbf{{{l}}}" for l in LANGS["medium"]]) + "& \\textbf{Avg}" + " & " + " & ".join([f"\\textbf{{{l}}}" for l in LANGS["low"]]) + "& \\textbf{Avg} \\\\"

    table += "\\begin{tabular}{" + colspec + "}\n"
    table += header + "\n"
    table += "\\hline\n"


    for (panel, model_name), metrics in rows.items():
        for lang, v in metrics.items():
            if v is None:
                continue
            # update max and second max in one go
            if v > lang_max[lang]:
                lang_second_max[lang] = lang_max[lang]
                lang_max[lang] = v
            elif v > lang_second_max[lang]:
                lang_second_max[lang] = v
    
    prev=None
    for (panel, model_name), metrics in rows.items():
        cells = []
        for resource_set in [LANGS['high'], LANGS['medium'], LANGS['low']]:
            for l in resource_set:
                if l not in metrics.keys():
                    v = None
                else:
                    v = metrics[l]
                if isinstance(v, (int, float)) and v == lang_max[l]:
                    cells.append(f"\\textbf{{{v:.3f}}}")
                elif isinstance(v, (int, float)) and v == lang_second_max[l]:
                    cells.append(f"\\underline{{{v:.3f}}}")
                else:
                    cells.append(f"{v:.3f}" if isinstance(v, (int, float)) else "--")
            
            # average cell  
            lang_values = [metrics[l] for l in resource_set if l in metrics.keys() and isinstance(metrics[l], (int, float))]
            if len(lang_values) > 0:
                avg = sum(lang_values) / len(lang_values)
            else:
                avg = 0.0
            cells.append(f"{avg:.3f}")
        
        if prev is None and panel == "VAN":
            table += f"\\multirow{{{n_unique_models}}}{{*}}{{FTL}} & " + f" {MODEL_DISPLAY_NAMES[model_name]} & " + " & ".join(cells) + " \\\\\n"
        elif prev and prev != panel and panel == "ATL":
            table += "\\hline\n"
            table += f"\\multirow{{{n_unique_models}}}{{*}}{{TOL}} & " + f" {MODEL_DISPLAY_NAMES[model_name]} & " + " & ".join(cells) + " \\\\\n"
        else:
            table += f" & " + f"{MODEL_DISPLAY_NAMES[model_name]} & "+ " & ".join(cells) + " \\\\\n"
            
            
        prev = panel

    table += "\\hline\n"
    table += "\\end{tabular}\n"

    print(table)

def merge_defaultdicts(d,d1):
    for k,v in d1.items():
        if (k in d):
            d[k].update(d1[k])
        else:
            d[k] = d1[k]
    return d

def main():

    configs: dict = {"context": True,
                     "metric": "f1", # accuracy or f1
                     "prompt_template": "instruct", # instruct or cls
                     "model_type": "slm",
                     }

    all_models = load_all_models(configs=configs, path=SLM_DIR)
    print(all_models)
    latex_table(all_models)

if __name__ == "__main__":
    main()
