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
SLM_DIR = os.path.join(BASE_DIR, "data/exp1")

COUNT = defaultdict(list)

def load_llms(configs: dict,
              rows: dict,
              meta_1: dict,
              ) -> dict[str, dict]:

    # COLLECT
    model_name = MODEL_DISPLAY_NAMES[meta_1["model_name"]]
    variant = ""
    if meta_1["verbose"]  == True:
        variant = "(verbose)"
    else:
        variant = "(instruct)"

    if meta_1['shots'] == True:
        panel = ("LLMs", "few")
    else:
        panel = ("LLMs", "zero")

    model_name = f"{model_name} {variant}"

    key = (model_name, panel)
    
    rows[key][meta_1["lang"]].append(meta_1["test_metrics"][configs['metric']])
    
    return rows

def load_slms(configs: dict, 
                  meta_files: list[Path],
                  rows: dict,
                  ) -> dict[str, dict]:
    
    meta_1 = load_metrics(meta_files[0])
    
    # panel
    if meta_1['model_type'] == "clf":
        panel = ("SLMs", "3Classifier")
    else:
        if meta_1["atl"]:
            panel = ("SLMs", "2ATL")
        else:
            panel = ("SLMs", "1VAN")
    
    model_name = MODEL_DISPLAY_NAMES[meta_1["model_name"]]
    key = (model_name, panel)

    if len(meta_files) == 1:    
        if meta_1['seed'] in [2025, 2026]:
            best_metric = meta_1["test_metrics"][-1]['metrics'][configs['metric']]
            rows[key][meta_1["lang"]].append(best_metric)
            COUNT[(model_name, meta_1['model_type'], meta_1['lang'], meta_1['atl'])].append((meta_1['seed'], meta_files[0].parent.name))  
            
        
    if len(meta_files) == 6:
        best_metric = find_best_metric_from_hyperparameter_search(all_meta_file_paths=meta_files, metric=configs['metric'])
        rows[key][meta_1["lang"]].append(best_metric) 

        COUNT[(model_name, meta_1['model_type'], meta_1['lang'], meta_1['atl'])].append(("hp", meta_files[0].parent.name))
    return rows

def load_plms(configs: dict, 
                  meta_files: list[Path],
                  rows: dict,
                  ) -> dict[str, dict]:
    
    meta_1 = load_metrics(meta_files[0])
    
    # panel
    model_name = MODEL_DISPLAY_NAMES[meta_1["model_name"]]
    panel = ("PLMs", "default")
    key = (model_name, panel)
        
    if len(meta_files) == 6:
        best_metric = find_best_metric_from_hyperparameter_search(all_meta_file_paths=meta_files, metric=configs['metric'])
        rows[key][meta_1["lang"]].append(best_metric) 

        COUNT[(model_name, meta_1['lang'], meta_1['atl'])].append((meta_1['seed'], meta_files[0].parent.name))
    return rows


def load_all_models(configs: dict, path: str) -> dict[str, dict]:
    root = Path(path)
    rows = defaultdict(lambda: {l: [] for l in LANG_ORDER})
    count = defaultdict(list)

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

            # BASICS TO DIRECT
            meta_files = [f for f in run_dir.iterdir() if f.is_file()]
            if len(meta_files) == 0:
                continue
        
            meta_1 = load_metrics(meta_files[0])

            if meta_1["model_type"] == "icl":
                # load llms
                rows = load_llms(configs=configs, rows=rows, meta_1=meta_1)
                
                count[(meta_1['model_name'], meta_1['lang'], meta_1['shots'], meta_1['verbose'])].append((meta_files[0].parent.name))  
                continue

            if meta_1["model_type"] in ["slm", "clf"]:
                rows = load_slms(configs=configs, rows=rows, meta_files=meta_files)
                hp_seed = "hp" if len(meta_files) == 6 else meta_1['seed']
                count[(meta_1['model_name'], meta_1['lang'], meta_1['model_type'], meta_1['atl'])].append((hp_seed, meta_files[0].parent.name))  
                continue

            if meta_1["model_type"] == "plm":
                rows = load_plms(configs=configs, rows=rows, meta_files=meta_files)
                count[(meta_1['model_name'], meta_1['lang'])].append((meta_1['seed'], meta_files[0].parent.name))  
                continue

    print("="*20)
    print("="*20)
    print("OVERVIEW OF COLLECTED FILES")
    print("="*20)
    print("="*20)
    # sortt by lang
    sorted_count = dict(sorted(count.items(), key=lambda x: (x[0][1], x[0][0])))
    prev_lang = None
    for k,v in sorted_count.items():
        if not prev_lang or k[1] != prev_lang:
            print("="*20)
            print(f"LANG {k[1]}")
            print("="*20)
            prev_lang = k[1]

        if len(k) == 4 and k[2] in ['clf', "slm"]:
            print(f"LANG: {k[1]} | MODEL {k[0]} | TYPE {k[2]} | LOSS {'ATL' if k[3] else 'VAN'}: {len(v)} run(s) -> {sorted(v, key=lambda x: str(x[0]))}")
            if len(v) > 3:
                print("WARNING: TOO MANY RUNS!")
            if set([run[0] for run in v]) != set([2025, 2026, 'hp']):
                print("WARNING: INCORRECT NUMBER OF RUNS!")
            print("")
        elif len(k) == 4:
            print(f"LANG: {k[1]} | MODEL {k[0]} | {'Few-shot' if k[2] else 'Zero-shot'} | {'Verbose' if k[3] else 'Instruct'}: {len(v)} runs -> {sorted(v, key=lambda x: str(x[0]))}")
            print("")
        elif len(k) == 2:
            print(f"LANG: {k[1]} | MODEL {k[0]}: {len(v)} runs -> {sorted(v, key=lambda x: str(x[0]))}")
            if set([run[0] for run in v]) != set([2025, 2026, 42]):
                print("WARNING: INCORRECT NUMBER OF RUNS!")
            print("")
    print("="*20)

    # generate averages
    out_rows = defaultdict(dict)
    for (model_name, pabel), metrics in rows.items():
        for lang, v in metrics.items():
            if isinstance(v, list) and len(v) > 0:
                avg_v = sum(v) / len(v)
                out_rows[(model_name, pabel)][lang] = avg_v
            else:
                out_rows[(model_name, pabel)][lang] = None

    # print("FINAL ROWS")
    # for k,v in out_rows.items():
    #     print(f"{k}: {v}")
    #     print()
    # return out_rows

    # sort rows


    final_rows = {}
    # LLMs
    llms = {k: v for k, v in out_rows.items() if k[1][0] == "LLMs"}
    llms = dict(sorted(llms.items(), key=lambda x: (x[0][1][1]), reverse=True))
    final_rows.update(llms)
    # PLMs
    plms = {k: v for k, v in out_rows.items() if k[1][0] == "PLMs"}
    plms = dict(sorted(plms.items(), key=lambda x: (x[0][0])))
    final_rows.update(plms)
    # SLMs
    slms = {k: v for k, v in out_rows.items() if k[1][0] == "SLMs"}
    slms = dict(sorted(slms.items(), key=lambda x: (x[0][1][1], x[0][0])))
    final_rows.update(slms)

    return final_rows

def latex_table(rows, context):

    lang_max = {l: float("-inf") for l in LANG_ORDER}
    lang_second_max = {l: float("-inf") for l in LANG_ORDER}
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
    header += " & " + " & ".join([f"\\textbf{{{l}}}" for l in LANGS['high']]) + " & \\textbf{Avg} & "  + " & ".join([f"\\textbf{{{l}}}" for l in LANGS['medium']]) + " & \\textbf{Avg} & " + " & ".join([f"\\textbf{{{l}}}" for l in LANGS['low']]) + " & \\textbf{Avg} \\\\"
    
    table += "\\begin{tabular}{" + colspec + "}\n"
    table += header + "\n"
    table += "\\hline\n"
    
    prev_main_header=None
    prev_minor_header=None
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

        # MAJOR HEADER LINES
        if prev_main_header is None and panel[0] == "LLMs":
                # table += "\\hline\n"
                table += f"\\rowcolor{{lightgray}}\\multicolumn{{{len(LANG_ORDER)+4}}}{{c}}{{\\textbf{{Decoder-based LLMs}}}} \\\\\n"
                table += "\\hline\n"
        if prev_main_header and prev_main_header != panel[0]:
            if panel[0] == "PLMs":
                table += "\\hline\n"
                table += f"\\rowcolor{{lightgray}}\\multicolumn{{{len(LANG_ORDER)+4}}}{{c}}{{\\textbf{{Encoder-based PLMs}}}} \\\\\n"
                table += "\\hline\n"
            elif panel[0] == "SLMs":
                table += "\\hline\n"    
                table += f"\\rowcolor{{lightgray}}\\multicolumn{{{len(LANG_ORDER)+4}}}{{c}}{{\\textbf{{Decoder-based SLMs}}}} \\\\\n"   
                table += "\\hline\n"
        
        # MINOR HEADER LINES
        if prev_minor_header is None and panel[1] == "zero":
            table += f"\\textbf{{Zero-shot}} \\\\\n"
            table += "\\hline\n"

        if prev_minor_header == "zero" and panel[1] == "few":
            table += f"\\textbf{{Few-shot}} \\\\\n"
            table += "\\hline\n"

        if prev_minor_header != panel[1] and panel[1] not in ["default", "zero", "few"]:
            if panel[1] == "1VAN":
                panel_name = "Full-token Loss (FTL)"
            elif panel[1] == "2ATL":
                panel_name = "Target-only Loss (TOL)"
            elif panel[1] == "3Classifier":
                panel_name = "Encoder-style"
            table += f"\\textbf{{{panel_name}}} \\\\\n"
            table += "\\hline\n"
        table += f"{model_name} & " + " & ".join(cells) + " \\\\\n"
    
        prev_main_header = panel[0]
        prev_minor_header = panel[1]

    table += "\\hline\n"
    table += "\\end{tabular}\n"

    print("\n\n")
    print(table)
    print("\n\n")

def main():

    configs = {
        "metric": "accuracy",
    }

    # load all models
    rows = load_all_models(configs, SLM_DIR)

    # print latex table
    latex_table(rows, configs)
if __name__ == "__main__":
    main()
