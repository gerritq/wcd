from t1 import (load_metrics,
                MODEL_DISPLAY_NAMES)
import os
import re
import json
import glob
from collections import defaultdict
from pathlib import Path
from argparse import Namespace

BASE_DIR = os.getenv("BASE_WCD")
SLM_DIR = os.path.join(BASE_DIR, "data/exp1")
                               
def load_all_models(configs: dict) -> dict[str, dict]:
    rows = defaultdict(dict)
    count = defaultdict(int)

    for lang in configs['langs']:
        path = os.path.join(SLM_DIR, f"{lang}_{configs['task']}") 

        if not os.path.exists(path):
            print(f"Path does not exist: {path}")
            continue

        root = Path(path)
        
        # iteratre over runs
        for run_dir in root.iterdir():
            if not (run_dir.is_dir()):
                print(f"Skipping non-run-dir: {run_dir}")
                continue

            # store best run in a dict
            best_run: dict = None 

            #iteratre over all met files
            for meta_path in run_dir.iterdir():
                if not (meta_path.is_file()):
                    print(f"Skipping non-meta-file: {meta_path}")
                    continue

                meta = load_metrics(meta_path)
                print(meta_path)
             
                # variant generation
                variant = ""
                model_name = MODEL_DISPLAY_NAMES[meta["model_name"]]
                if meta["model_type"] in ["cls", "clf", "classifier"]:
                    variant = "(clf)"
                if meta["model_type"] == "slm" and meta["atl"]  == True:
                    variant = "(atl)"
                if meta["model_type"] == "slm" and meta["atl"]  == False:
                    variant = "(van)"


                # if meta["model_type"] == "icl":
                #     if meta['shots'] == True and meta["verbose"]  == True:
                #         variant = "(x-s\&v)"
                #     if meta['shots'] == True and meta["verbose"]  == False:
                #         variant = "(x-s)"
                #     if meta['shots'] == False and meta["verbose"]  == True:
                #         variant = "(0-s\&v)"
                #     if meta['shots'] == False and meta["verbose"]  == False:
                #         variant = "(0-s)"
                


                if variant != "":
                    model_name = f"{model_name} {variant}"
                
                # icl has only one meta file
                if meta["model_type"] == "icl":
                    test_metric = meta["test_metrics"][configs['metric']]
                    best_run = {
                        "model_name": model_name,
                        "lang": meta["lang"],
                        "dev_metric": None,
                        "test_metric": test_metric,
                    }
                else:
                    # get the best metrics
                    dev_metrics = meta.get("dev_metrics", [])
                    test_metrics = meta.get("test_metrics", [])

                    count[(model_name, meta['lang'], os.path.dirname(meta_path))] += 1

                    # go over epochs
                    for dev_entry in dev_metrics:
                        epoch = dev_entry["epoch"]
                        dev_metric = dev_entry["metrics"][configs['metric']]

                        if (
                            best_run is None
                            or dev_metric > best_run["dev_metric"]
                        ):
                            test_entry = next(
                                (t for t in test_metrics if t["epoch"] == epoch),
                                None
                            )
                            if test_entry is None:
                                continue

                            best_run = {
                                "model_name": model_name,
                                "lang": meta["lang"],
                                "dev_metric": dev_metric,
                                "test_metric": test_entry["metrics"][configs['metric']],
                            }

            # after scanning all meta_* in this run
            if best_run:
                m = best_run["model_name"]
                l = best_run["lang"][:2]
                rows[m][l] = best_run["test_metric"]
    print("="*20)
    print("MODEL COUNT")
    for k,v in count.items():
        print(f"{k}: {v}") 
    print("="*20)

    def sort_key_model_name(model_name: str):
        # Expect names like "LLaMA-3 (van)" or "Qwen (atl)"
        parts = model_name.rsplit(" ", 1)
        if len(parts) == 2 and parts[1].startswith("(") and parts[1].endswith(")"):
            base = parts[0]
            variant = parts[1]
        else:
            base = model_name
            variant = ""

        variant_order = {"(van)": 0, "(atl)": 1, "(clf)": 2}
        return (base, variant_order.get(variant, 99))

    # replace your sort line with:
    sorted_rows = dict(sorted(rows.items(), key=lambda x: sort_key_model_name(x[0])))

    return sorted_rows

def latex_table(rows: dict[str, dict], config: dict) -> str:
    table = "\n\n"
    colspec = "l" + "c" * len(config['langs'])
    header = " & ".join(["\\textbf{Model}"] + [f"\\textbf{{{lang}}}" for lang in config['langs']]) + " \\\\ \\toprule\n"
    table += "\\begin{tabular}{" + colspec + "}\n"
    table += header

    # Select best per language
    
    lang_max = {l: float("-inf") for l in config['langs']}
    lang_second_max = {l: float("-inf") for l in config['langs']}

    for _, metrics in rows.items():
        for lang, v in metrics.items():
            if v is None:
                continue
            # update max and second max in one go
            if v > lang_max[lang]:
                lang_second_max[lang] = lang_max[lang]
                lang_max[lang] = v
            elif v > lang_second_max[lang]:
                lang_second_max[lang] = v


    for model_name, lang_metrics in rows.items():
        if model_name == "Best submission":
            table += f"\\multicolumn{{{len(config['langs']) + 1}}}{{l}}{{\\textit{{{config['title']}}}}} \\\\\n"
        
        cells = [f"{model_name}"]

        for lang in config['langs']:
            v = lang_metrics.get(lang)
            if v == lang_max[lang]:
                cells.append(f"\\textbf{{{v:.3f}}}")
            else:
                cells.append(f"{v:.3f}" if isinstance(v, (int, float)) else "--")

        table += " & ".join(cells) + " \\\\\n"

        if model_name == "Best submission":
                table += "\\hline\n"
                table += "\\multicolumn{" + str(len(config['langs']) + 1) + "}{l}{\\textit{Our Models}} \\\\\n   "
                    
            # row mentionind our models

    table += "\\bottomrule\n"
    table += "\\end{tabular}\n"
    print(f"LaTeX table for {config['title']}:\n{table}")
    return table

def main():
    configs = [{"task": "ct24", 
                'langs': ["en", 'ar', "nl"],
                "title": "CT–CWT–24",
                "best_f1": {"en": 0.802, 
                            "ar": 0.569, 
                            "nl": 0.732},
                },
                {"task": "nlp4if", 
                 'langs': ["en", 'ar', "bg"],
                "title": "NLP4IF",
                "best_f1": {"en": 0.835, 
                            "ar": 0.843, 
                            "bg": 0.887}
                }
             ]
    for config in configs:
        config['metric'] = 'f1'
        rows = load_all_models(config)
        print(rows)
        
        # add best f1
        final_rows = {"Best submission": {}}
        for lang in config['langs']:
            final_rows["Best submission"][lang] = config['best_f1'][lang]

        final_rows.update(rows)
        
        print(f"Rows for {config['title']}:\n{final_rows}")

        # sort rows by model name
        table = latex_table(final_rows, config)

if __name__ == "__main__":
    main()