import os
import re
import json
import glob
from collections import defaultdict
from pathlib import Path
from argparse import Namespace

# ------------------------------------------------------------------------------------------
# configs
# ------------------------------------------------------------------------------------------
BASE_DIR = os.getenv("BASE_WCD")
SLM_DIR = os.path.join(BASE_DIR, "data/exp1")

MODEL_DISPLAY_NAMES = {"meta-llama/Llama-3.1-8B": "Llama3-8B",
                      "meta-llama/Llama-3.1-8B-Instruct": "Llama3-8B", # same for cls and slm
                       "Qwen/Qwen3-8B-Base": "Qwen3-8B",
                       "microsoft/mdeberta-v3-base": "mDeberta-base",
                       "microsoft/deberta-v3-large": "mDeberta-large",
                       "google-bert/bert-base-multilingual-uncased": "mBert",
                       "FacebookAI/xlm-roberta-base": "XLM-R-base",
                       "FacebookAI/xlm-roberta-large": "XLM-R-large",
                       "openai/gpt-4o-mini": "GPT-4o-mini"
                        }

run_re = re.compile(r"run_\w+")
meta_re = re.compile(r"meta_\d+")

LANGS = ["en","nl","no","it","pt","ro","ru","uk","bg", "vi", "id", "tr"]

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

            #iteratre over all met files
            for meta_path in run_dir.iterdir():
                if not (meta_path.is_file()):
                    print(f"Skipping non-meta-file: {meta_path}")
                    continue

                meta = load_metrics(meta_path)
                print(meta_path)
                # filter by context
                if meta["context"] != configs['context']:
                    print(f"Skipping due to context mismatch: {meta_path}")
                    continue

                # filter by prompt_template
                if ("prompt_template" in meta and meta["prompt_template"] != configs['prompt_template']):
                    print(f"Skipping due to prompt template mismatch: {meta_path}")
                    continue

                # filter by training_size
                if (meta["model_type"] != "icl" and meta["training_size"] != configs['training_size']):
                    print(f"Skipping due to training size mismatch {meta['training_size']}: {meta_path}")
                    continue

                # variant generation
                variant = ""
                model_name = MODEL_DISPLAY_NAMES[meta["model_name"]]
                if meta["model_type"] in ["cls", "clf", "classifier"]: # we use inconsistent names
                    variant = "(clf)"
                if meta["model_type"] == "slm" and meta["atl"]  == True:
                    variant = "(atl)"
                if meta["model_type"] == "slm" and meta["atl"]  == False:
                    variant = "(van)"


                if meta["model_type"] == "icl":
                    if meta['shots'] == True and meta["verbose"]  == True:
                        variant = "(x-s\&v)"
                    if meta['shots'] == True and meta["verbose"]  == False:
                        variant = "(x-s)"
                    if meta['shots'] == False and meta["verbose"]  == True:
                        variant = "(0-s\&v)"
                    if meta['shots'] == False and meta["verbose"]  == False:
                        variant = "(0-s)"
                


                if variant != "":
                    model_name = f"{model_name} {variant}"

                # panel generation
                if meta["model_type"] == "icl":
                    panel = "LLMs"
                if meta["model_type"] in ["slm", "cls", "clf", "classifier"]:
                    panel = "SLMs"
                if meta["model_type"] in ["plm"]:
                    panel = "PLMs"

                model_name = (model_name, panel)
                
                # icl has only one meta file
                if meta["model_type"] == "icl":
                    test_metric = meta["test_metrics"][configs['metric']]
                    best_run = {
                        "model_name": model_name,
                        "panel": panel,
                        "lang": meta["lang"],
                        "dev_metric": None,
                        "test_metric": test_metric,
                    }
                else:
                    # get the best metrics
                    dev_metrics = meta.get("dev_metrics", [])
                    test_metrics = meta.get("test_metrics", [])

                    count[(model_name, meta['lang'], panel, os.path.dirname(meta_path))] += 1

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
                                "panel": panel,
                                "lang": meta["lang"],
                                "dev_metric": dev_metric,
                                "test_metric": test_entry["metrics"][configs['metric']],
                            }

            # after scanning all meta_* in this run
            if best_run:
                m = best_run["model_name"]
                l = best_run["lang"]
                panel = best_run["panel"]
                rows[(m, panel)][l] = best_run["test_metric"]
    print("="*20)
    print("MODEL COUNT")
    for k,v in count.items():
        print(f"{k}: {v}") 
        if k[0][1] == "PLMs" and v != 6:
            print("CHECK ABOVE")
        if k[0][1] == "SLMs" and v != 9:
            print("CHECK ABOVE")
    print("="*20)

    panel_order = ["LLMs", "PLMs", "SLMs"]
    sorted_rows = {}

    for panel in panel_order:
        panel_rows = {k[0]: v for k, v in rows.items() if k[1] == panel}

        if panel == "SLMs":
            def slm_sort_key(model_key):
                # get the name
                name = model_key[0]
                # get the variant
                variant = name.split("(")[-1].replace(")", "").strip()
                # defien custom order (painn)
                order = {"van": 0, "atl": 1, "clf": 2}
                return (order.get(variant, 999), name)

            panel_sorted = dict(sorted(panel_rows.items(),
                                    key=lambda x: slm_sort_key(x[0])))
        else:
            # sort alphabetically
            panel_sorted = dict(sorted(panel_rows.items(),
                                    key=lambda x: x[0][0]))

        sorted_rows.update(panel_sorted)

    return sorted_rows

def latex_table(rows, context):
    # print LaTeX table
    table = "\n\n"
    colspec = "l" + "c" * len(LANGS)
    
    header = "\\textbf{Model} $\\downarrow$\\ \\textbf{Language} $\\rightarrow$ & " + " & ".join([f"\\textbf{{{l}}}" for l in LANGS]) + " \\\\"

    table += "\\begin{tabular}{" + colspec + "}\n"
    table += "\\hline\n"
    table += header + "\n"
    table += "\\hline\n"

    lang_max = {l: float("-inf") for l in LANGS}
    lang_second_max = {l: float("-inf") for l in LANGS}

    for (_, panel), metrics in rows.items():
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
    for (name, panel), metrics in rows.items():
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

        if prev is None and panel == "LLMs":
                # table += "\\hline\n"
                table += f"\\rowcolor{{lightgray}}\\multicolumn{{{len(LANGS)+1}}}{{c}}{{\\textbf{{Decoder-based LLMs}}}} \\\\\n"
                table += "\\hline\n"
        if prev and prev != panel:
            if panel == "PLMs":
                table += "\\hline\n"
                table += f"\\rowcolor{{lightgray}}\\multicolumn{{{len(LANGS)+1}}}{{c}}{{\\textbf{{Encoder-based PLMs}}}} \\\\\n"
                table += "\\hline\n"
            elif panel == "SLMs":
                table += "\\hline\n"    
                table += f"\\rowcolor{{lightgray}}\\multicolumn{{{len(LANGS)+1}}}{{c}}{{\\textbf{{Decoder-based SLMs}}}} \\\\\n"   
                table += "\\hline\n"
        
        table += f"{name} & " + " & ".join(cells) + " \\\\\n"
        
        prev = panel

    table += "\\hline\n"
    table += "\\end{tabular}\n"

    # Save to file
    with open(f"table1.tex", "w", encoding="utf-8") as f:
        f.write(table)

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
                     "prompt_template": "instruct",
                     "training_size": 5000}

    all_models = load_all_models(configs=configs, path=SLM_DIR)
    latex_table(all_models, configs["context"])

if __name__ == "__main__":
    main()
