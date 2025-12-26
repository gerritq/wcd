import os
import re
import json
import glob
from collections import defaultdict
from pathlib import Path
from argparse import Namespace
from utils import load_metrics, find_best_metric_from_hyperparameter_search, LANGS, LANG_ORDER, MODEL_DISPLAY_NAMES
# ------------------------------------------------------------------------------------------
# configs
# ------------------------------------------------------------------------------------------
BASE_DIR = os.getenv("BASE_WCD")
SLM_DIR = os.path.join(BASE_DIR, "data/exp1_single_run")

run_re = re.compile(r"run_\w+")
meta_re = re.compile(r"meta_\d+")

def load_all_models(configs: dict, path: str) -> dict[str, dict]:
    root = Path(path)
    rows = defaultdict(dict)
    count = set()

    # iteratore over all lang dirs
    for lang_dir in root.iterdir():
        if not lang_dir.is_dir():
            # print(f"Skipping non-lang-dir: {lang_dir}")
            continue
        
        # iteratre over runs
        for run_dir in lang_dir.iterdir():
            if not (run_dir.is_dir()):
                # print(f"Skipping non-run-dir: {run_dir}")
                continue

            # store best run in a dict
            best_run: dict = None 


            meta_files = [f for f in run_dir.iterdir() if f.is_file()]

            if len(meta_files) == 0:
                # print(f"No meta files found in run dir: {run_dir}")
                continue
        
            meta_1 = load_metrics(meta_files[0])

            if meta_1["context"] != configs['context']:
                # print(f"Skipping due to context mismatch: {run_dir}")
                continue

            model_name = MODEL_DISPLAY_NAMES[meta_1["model_name"]]
            variant = ""

            # do not want non hp runs
            # if meta_files == 1 and meta_1["model_type"] in ["plm", "slm", "cls", "clf", "classifier"]:
            #     # print(f"Skipping non-hp run: {run_dir}")
            #     continue

            # ICL
            if len(meta_files) == 1 and meta_1["model_type"] == "icl":
                if meta_1['shots'] == True and meta_1["verbose"]  == True:
                    variant = "(x-s\&v)"
                if meta_1['shots'] == True and meta_1["verbose"]  == False:
                    variant = "(x-s)"
                if meta_1['shots'] == False and meta_1["verbose"]  == True:
                    variant = "(0-s\&v)"
                if meta_1['shots'] == False and meta_1["verbose"]  == False:
                    variant = "(0-s)"

                best_run = {
                            "panel": "LLMs",
                            "lang": meta_1["lang"],
                            "test_metric": meta_1["metrics"][configs['metric']],
                            }

            # PLMS
            if len(meta_files) == 6 and meta_1["model_type"] == "plm":
                best_run = {
                            "panel": "PLMs",
                            "lang": meta_1["lang"],
                            "test_metric": find_best_metric_from_hyperparameter_search(meta_files, configs['metric']),
                            }

            # SLMs
            if len(meta_files) == 1 and meta_1["model_type"] in ["slm", "cls", "clf", "classifier"]:
                if meta_1["model_type"] == 'slm' and (meta_1["prompt_template"] != configs['prompt_template'] or meta_1["training_size"] != configs['training_size']):
                    # print(f"Skipping SLM due to prompt template mismatch: {run_dir}")
                    continue
                if meta_1["model_type"] == 'slm':
                    variant = "(van)" if meta_1["atl"] == False else "(atl)"
                else:
                    variant = "(clf)"
                best_run = {
                            "panel": "SLMs",
                            "lang": meta_1["lang"],
                            "test_metric": find_best_metric_from_hyperparameter_search(meta_files, configs['metric']),
                            }             

            if best_run:
                model_name = f"{MODEL_DISPLAY_NAMES[meta_1['model_name']]} {variant}" if variant else MODEL_DISPLAY_NAMES[meta_1['model_name']]
                l = meta_1["lang"]
                panel = best_run["panel"]
                rows[(model_name, panel)][l] = best_run["test_metric"]
            
                # count[(model_name, meta_1['lang'], panel, len(meta_files))] += 1
                count.add((model_name, panel, len(meta_files)))

    print("="*20)
    print("MODEL COUNT")
    for item in count:
        print(f"{item}") 
    print("="*20)

    print(rows)
    panel_order = ["LLMs", "PLMs", "SLMs"]
    sorted_rows = {}

    for panel in panel_order:
        panel_rows = {k: v for k, v in rows.items() if k[1] == panel}

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

    lang_max = {l: float("-inf") for l in LANG_ORDER}
    lang_second_max = {l: float("-inf") for l in LANG_ORDER}
    print(lang_max)
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
        
    # print LaTeX table
    table = "\n\n"
    colspec = "l" + "c" * (len(LANG_ORDER) + 3)
    
    header = " & \\multicolumn{" + str(len(LANGS['high'])+1) + "}{c}{\\textbf{High Resource}} & \\multicolumn{" + str(len(LANGS['medium'])+1) + "}{c}{\\textbf{Medium Resource}} & \\multicolumn{" + str(len(LANGS['low'])+1) + "}{c}{\\textbf{Low Resource}} \\\\"
    header += "\\cmidrule(lr){2-" + str(len(LANGS['high'])+2) + "}  \\cmidrule(lr){" + str(len(LANGS['high'])+3) + "-" + str(len(LANGS['high'])+len(LANGS['medium'])+3) + "}  \\cmidrule(lr){" + str(len(LANGS['high'])+len(LANGS['medium'])+4) + "-" + str(len(LANGS['high'])+len(LANGS['medium'])+len(LANGS['low'])+4) + "}"
    header += " & " + " & ".join([f"\\textbf{{{l}}}" for l in LANGS['high']]) + " & \\textbf{Avg} & "  + " & ".join([f"\\textbf{{{l}}}" for l in LANGS['medium']]) + " & \\textbf{Avg} & " + " & ".join([f"\\textbf{{{l}}}" for l in LANGS['medium']]) + " & \\textbf{Avg} \\\\"
    
    table += "\\begin{tabular}{" + colspec + "}\n"
    table += header + "\n"
    table += "\\hline\n"
    
    prev=None
    for (model_name, panel), metrics in rows.items():
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

        if prev is None and panel == "LLMs":
                # table += "\\hline\n"
                table += f"\\rowcolor{{lightgray}}\\multicolumn{{{len(LANG_ORDER)+4}}}{{c}}{{\\textbf{{Decoder-based LLMs}}}} \\\\\n"
                table += "\\hline\n"
        if prev and prev != panel:
            if panel == "PLMs":
                table += "\\hline\n"
                table += f"\\rowcolor{{lightgray}}\\multicolumn{{{len(LANG_ORDER)+4}}}{{c}}{{\\textbf{{Encoder-based PLMs}}}} \\\\\n"
                table += "\\hline\n"
            elif panel == "SLMs":
                table += "\\hline\n"    
                table += f"\\rowcolor{{lightgray}}\\multicolumn{{{len(LANG_ORDER)+4}}}{{c}}{{\\textbf{{Decoder-based SLMs}}}} \\\\\n"   
                table += "\\hline\n"
        
        table += f"{model_name} & " + " & ".join(cells) + " \\\\\n"
    
        prev = panel

    table += "\\hline\n"
    table += "\\end{tabular}\n"

    print("\n\n")
    print(table)
    print("\n\n")

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
                     "prompt_template": "minimal",
                     "training_size": 5000}

    all_models = load_all_models(configs=configs, path=SLM_DIR)
    latex_table(all_models, configs["context"])

if __name__ == "__main__":
    main()
