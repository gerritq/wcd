import os
import re
import json
import glob
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
    rows = defaultdict(dict)
    count = defaultdict(int)

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

            if len(meta_files) != 6:
                # print(f"Skipping due to unexpected number of meta files ({len(meta_files)}) in: {run_dir}")
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

            # panel generation
            if meta_1["atl"] == True:
                panel = "ATL"
            else:
                panel = "VAN"

            # best metric extraction
            best_metric = find_best_metric_from_hyperparameter_search(all_meta_file_paths=meta_files, metric=configs['metric'])
        
            count[(meta_1['model_name'], meta_1['lang'], os.path.dirname(meta_files[0]))] += 1

            best_run = {
                        "model_name": meta_1["model_name"],
                        "prompt_template": meta_1["prompt_template"],
                        "panel": panel,
                        "lang": meta_1["lang"],
                        "test_metric": best_metric,
                                    }
            
            rows[(panel, meta_1["model_name"])][meta_1["lang"]] = best_run["test_metric"]
    
    
    # sort reverse rows by panel and model name
    rows = dict(sorted(rows.items(), key=lambda x: (x[0][0], x[0][1]), reverse=True))
    
    print("="*20)
    print("COLLECTED MODELS COUNT")
    for k,v in count.items():
        print(f"{k}: {v}") 
        print()
    print("="*20)
    print("FINAL ROWS")
    for k,v in rows.items():
        print(f"{k}: {v}") 
        print()
    print("="*20)



    return rows

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
