import os
import re
import json
import glob
from collections import defaultdict
from pathlib import Path
from argparse import Namespace
from sys import meta_path

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

LANGS = ["en", "nl", "no", "it", "pt", "ro", "ru", "uk", "bg", "vi", "id", "tr", "sq", "hy", "mk", "az"]

def load_metrics(path):
    """"Load a sinlge meta_file"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_all_models(configs: dict, path: str) -> dict[str, dict]:
    root = Path(path)
    rows = defaultdict(dict)
    count = defaultdict(int)

    # iteratore over all lang dirs
    for lang_dir in root.iterdir():
        if not lang_dir.is_dir():
            print(f"Skipping non-lang-dir: {lang_dir}")
            continue
        
        # iteratre over runs
        for run_dir in lang_dir.iterdir():
            if not (run_dir.is_dir()):
                print(f"Skipping non-run-dir: {run_dir}")
                continue

            # store best run in a dict
            best_run: dict = None 

            # skip if more than one meta file
            meta_files = [f for f in run_dir.iterdir() if f.is_file()]
            if len(meta_files) > 1 or len(meta_files) == 0:
                # print(f"Multiple files found: {run_dir}")
                continue

            #iteratre over all met files
            
            if not (meta_files[0].is_file()):
                # print(f"Skipping non-meta-file: {meta_files[0]}")
                continue

            meta = load_metrics(meta_files[0])
            print(meta_files[0])
            
            # FILTERS
            if (meta["model_type"] in ['icl', 'plm']):
                # print(f"Skipping due to model type mismatch: {meta['model_type']}")
                continue

            if (meta["prompt_template"] != configs['prompt_template']):
                # print(f"Skipping due to prompt template mismatch: {meta['prompt_template']}")
                continue

            # panel generation
            if meta["atl"] == True:
                panel = "ATL"
            else:
                panel = "VAN"
        
            count[(meta['model_name'], meta['lang'], os.path.dirname(meta_files[0]))] += 1

            best_run = {
                        "model_name": meta["model_name"],
                        "prompt_template": meta["prompt_template"],
                        "panel": panel,
                        "lang": meta["lang"],
                        "test_metric": meta["test_metrics"][-1]["metrics"][configs['metric']],
                                    }
            
            rows[(panel, meta["model_name"])][meta["lang"]] = best_run["test_metric"]
    
    
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

    # compute average model performance
    averages = {} 
    for (panel, model_name), metrics in rows.items():
        vals = [v for v in metrics.values() if isinstance(v, (int, float))]
        avg = sum(vals) / len(vals) if vals else 0.0
        averages[(panel, model_name)] = avg

    highest_average_model = sorted(averages.items(), key=lambda x: x[1], reverse=True)[0][0]
    second_highest_average_model = sorted(averages.items(), key=lambda x: x[1], reverse=True)[1][0]

    # print LaTeX table
    table = "\n\n"
    colspec = "ll" + "c" * (len(LANGS) + 1)
    
    header = "\\textbf{Loss} & \\textbf{Model} & " + " & ".join([f"\\textbf{{{l}}}" for l in LANGS]) + "& \\textbf{Avg}" + " \\\\"

    table += "\\begin{tabular}{" + colspec + "}\n"
    table += "\\hline\n"
    table += header + "\n"
    table += "\\hline\n"

    lang_max = {l: float("-inf") for l in LANGS}
    lang_second_max = {l: float("-inf") for l in LANGS}

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
        for l in LANGS:
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

        avg = averages[(panel, model_name)]
        if (panel, model_name) == highest_average_model:
            avg_cell = f"\\textbf{{{avg:.3f}}}"
        elif (panel, model_name) == second_highest_average_model:
            avg_cell = f"\\underline{{{avg:.3f}}}"
        else:
            avg_cell = f"{avg:.3f}"
        cells.append(avg_cell)
        
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
                     "metric": "accuracy", # accuracy or f1
                     "prompt_template": "minimal",
                     "model_type": "slm",
                     }

    all_models = load_all_models(configs=configs, path=SLM_DIR)
    print(all_models)
    latex_table(all_models)

if __name__ == "__main__":
    main()
