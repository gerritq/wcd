import os
import sys
from collections import defaultdict
from pathlib import Path
from argparse import Namespace
from utils import load_metrics, find_best_metric_from_hyperparameter_search, LANGS, LANG_ORDER, MODEL_DISPLAY_NAMES
import numpy as np
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
        variant = "2(verbose)"
    else:
        variant = "1(instruct)"

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

    # average over runs / robustness
    out_rows = defaultdict(dict)
    out_stds = defaultdict(dict)
    for (model_name, panel), metrics in rows.items():
        for lang, v in metrics.items():
            if isinstance(v, list) and len(v) > 0:
                avg_v = np.mean(v)
                std = np.std(v)
                out_rows[(model_name, panel)][lang] = avg_v
                out_stds[(model_name, panel)][lang] = std
            else:
                out_rows[(model_name, panel)][lang] = None
                out_stds[(model_name, panel)][lang] = None


    # generate resource-level averages
    model_panel_avgs = {}
    for model_key, metrics in out_rows.items():
        model_panel_avgs[model_key] = {}
        for resource_set_name, resource_set in LANGS.items():
            tmp_metrics = []
            for l in resource_set:
                if l in metrics.keys() and metrics[l] is not None:
                    tmp_metrics.append(metrics[l])
            if len(tmp_metrics) > 0:
                model_panel_avgs[model_key][resource_set_name] = sum(tmp_metrics) / len(tmp_metrics)
            else:
                model_panel_avgs[model_key][resource_set_name] = None

    print("\n\nResource-level Averages:\n")
    for model_name, avgs in model_panel_avgs.items():
        avg_strs = []
        for resource_set_name in ['high', 'medium', 'low']:
            avg = avgs[resource_set_name]
            avg_strs.append(f"{resource_set_name}: {avg:.3f}" if avg is not None else f"{resource_set_name}: None")
        print(f"{model_name}: " + " | ".join(avg_strs))
    print("\n\n")

    # generate overall averages
    all_models = set(out_rows.keys())
    overall_avg = {}

    for model_key in all_models:
        tmp_metrics = []
        for model_key_inner, metrics in out_rows.items():
            if model_key != model_key_inner:
                continue
            for _, v in metrics.items():
                if v is not None:
                    tmp_metrics.append(v)

        if len(tmp_metrics) > 0:
            overall_avg[model_key] = sum(tmp_metrics) / len(tmp_metrics)
        else:
            overall_avg[model_key] = None
        
    print("\n\n")
    print("Overall Averages:")
    # sort by model type and name
    overall_avg = dict(sorted(overall_avg.items(), key=lambda x: (x[0][1][0], x[0][0])))
    for model_name, avg in overall_avg.items():
        print(f"{model_name}: {avg:.3f}" if avg is not None else f"{model_name}: None")
    print("\n\n")
        

    final_rows = {}
    # LLMs
    llms_zero = {k: v for k, v in out_rows.items() if k[1] == ("LLMs", "zero")}
    llms_zero = dict(sorted(llms_zero.items(), key=lambda x: (x[0][0]), reverse=False))
    llms_few = {k: v for k, v in out_rows.items() if k[1] == ("LLMs", "few")}
    llms_few = dict(sorted(llms_few.items(), key=lambda x: (x[0][0]), reverse=False))
    final_rows.update(llms_zero)
    final_rows.update(llms_few)
    
    
    # PLMs
    plms = {k: v for k, v in out_rows.items() if k[1][0] == "PLMs"}
    plms = dict(sorted(plms.items(), key=lambda x: (x[0][0])))
    final_rows.update(plms)
    # SLMs
    # for each model, keep only the best variant
    unique_slm_models = set([k[0] for k in out_rows.keys() if k[1][0] == "SLMs"])
    slm_keys_to_keep = []
    for model in unique_slm_models:
        variants = {k: v for k, v in overall_avg.items() if k[0] == model and k[1][0] == "SLMs"}
        best_variant = max(variants.items(), key=lambda x: x[1] if x[1] is not None else float("-inf"))[0]
        slm_keys_to_keep.append(best_variant)

    slms = {k: v for k, v in out_rows.items() if k in slm_keys_to_keep}
    # sort SLMs by panel type and model name
    slms = dict(sorted(slms.items(), key=lambda x: (x[0][0])))
    final_rows.update(slms)

    # print("\n\n")
    # print(slm_keys_to_keep)
    # print("\n\n")

    return final_rows,overall_avg, model_panel_avgs, out_stds

def latex_table(rows, overall_avg, model_panel_avgs, out_stds):


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
        
    max_panel_avg = {"high": float("-inf"), "medium": float("-inf"), "low": float("-inf")}
    second_max_panel_avg = {"high": float("-inf"), "medium": float("-inf"), "low": float("-inf")}
    for model_key, avgs in model_panel_avgs.items():
        for resource_set_name, avg in avgs.items():
            if avg is None:
                continue
            if avg > max_panel_avg[resource_set_name]:
                second_max_panel_avg[resource_set_name] = max_panel_avg[resource_set_name]
                max_panel_avg[resource_set_name] = avg
            elif avg > second_max_panel_avg[resource_set_name]:
                second_max_panel_avg[resource_set_name] = avg

    overall_max = float("-inf")
    overall_second_max = float("-inf")
    for model_key, avg in overall_avg.items():
        if avg is None:
            continue
        if avg > overall_max:
            overall_second_max = overall_max
            overall_max = avg
        elif avg > overall_second_max:
            overall_second_max = avg

    # print LaTeX table
    table = "\n\n"
    colspec = "l" + "c" * (len(LANG_ORDER) + 4)
    
    header = " & \\multicolumn{" + str(len(LANGS['high'])+1) + "}{c}{\\textbf{High Resource}} & \\multicolumn{" + str(len(LANGS['medium'])+1) + "}{c}{\\textbf{Medium Resource}} & \\multicolumn{" + str(len(LANGS['low'])+1) + "}{c}{\\textbf{Low Resource}} \\\\"
    header += "\\cmidrule(lr){2-" + str(len(LANGS['high'])+2) + "}  \\cmidrule(lr){" + str(len(LANGS['high'])+3) + "-" + str(len(LANGS['high'])+len(LANGS['medium'])+3) + "}  \\cmidrule(lr){" + str(len(LANGS['high'])+len(LANGS['medium'])+4) + "-" + str(len(LANGS['high'])+len(LANGS['medium'])+len(LANGS['low'])+4) + "}"
    header += " & " + " & ".join([f"\\textbf{{{l}}}" for l in LANGS['high']]) + " & \\textbf{Avg} & "  + " & ".join([f"\\textbf{{{l}}}" for l in LANGS['medium']]) + " & \\textbf{Avg} & " + " & ".join([f"\\textbf{{{l}}}" for l in LANGS['low']]) + " & \\textbf{Avg} & \\textbf{Avg} \\\\"
    
    table += "\\begin{tabular}{" + colspec + "}\n"
    table += header + "\n"
    table += "\\toprule\n"
    
    prev_main_header=None
    prev_minor_header=None
    for (model_name, panel), metrics in rows.items():
        cells = []
        for resource, resource_set in [('high', LANGS['high']), ('medium', LANGS['medium']), ('low', LANGS['low'])]:
            for l in resource_set:
                if l not in metrics.keys():
                    v = None
                else:
                    v = metrics[l]
                    std = out_stds[(model_name, panel)][l]
                if isinstance(v, (int, float)) and v == lang_max[l]:
                    cells.append(f"\\shortstack{{\\textbf{{{v*100:.2f}}} \\\\ \\scriptsize{{(\\pm{std*100:.2f})}}}}" if std is not None and panel[0] in ["SLMs", "PLMs"] else f"\\textbf{{{v*100:.2f}}}")
                elif isinstance(v, (int, float)) and v == lang_second_max[l]:
                    cells.append(f"\\shortstack{{\\underline{{{v*100:.2f}}} \\\\ \\scriptsize{{(\\pm{std*100:.2f})}}}}" if std is not None and panel[0] in ["SLMs", "PLMs"] else f"\\underline{{{v*100:.2f}}}")
                else:
                    cells.append(f"\\shortstack{{{v*100:.2f} \\\\ \\scriptsize{{(\\pm{std*100:.2f})}}}}" if isinstance(v, (int, float)) and std is not None and panel[0] in ["SLMs", "PLMs"] else (f"{v*100:.2f}" if isinstance(v, (int, float)) else "--"))
            
            # average cell  
            avg = model_panel_avgs[(model_name, panel)][resource]
            if isinstance(avg, (int, float)) and avg == max_panel_avg[resource]:
                cells.append(f"\\cellcolor{{gray!25}} \\textbf{{{avg*100:.2f}}}")
            elif isinstance(avg, (int, float)) and avg == second_max_panel_avg[resource]:
                cells.append(f"\\cellcolor{{gray!25}} \\underline{{{avg*100:.2f}}}")
            else:
                cells.append(f"\\cellcolor{{gray!25}} {avg*100:.2f}")

        # overall average cell
        model_avg = overall_avg[(model_name, panel)]
        if isinstance(model_avg, (int, float)):
            if model_avg == overall_max:
                cells.append(f"\\textbf{{{model_avg*100:.2f}}}")
            elif model_avg == overall_second_max:
                cells.append(f"\\underline{{{model_avg*100:.2f}}}")
            else :
                cells.append(f"{model_avg*100:.2f}")
        else:
            cells.append(f"--")
        

        # MAJOR HEADER LINES
        if prev_main_header is None and panel[0] == "LLMs":
                # table += "\\hline\n"
                table += f"\\multicolumn{{{len(LANG_ORDER)+5}}}{{c}}{{\\textbf{{Decoder-based LLMs}}}} \\\\\n"
                table += "\\midrule\n"
        if prev_main_header and prev_main_header != panel[0]:
            if panel[0] == "PLMs":
                table += "\\midrule\n"
                table += f"\\multicolumn{{{len(LANG_ORDER)+5}}}{{c}}{{\\textbf{{Encoder-based PLMs}}}} \\\\\n"
                table += "\\midrule\n"
            elif panel[0] == "SLMs":
                table += "\\midrule\n"    
                table += f"\\multicolumn{{{len(LANG_ORDER)+5}}}{{c}}{{\\textbf{{Decoder-based SLMs}}}} \\\\\n"   
                table += "\\midrule\n"
        
        # MINOR HEADER LINES
        if prev_minor_header is None and panel[1] == "zero":
            table += f"\\textit{{Zero-shot}} \\\\\n"
            table += "\\midrule\n"

        if prev_minor_header == "zero" and panel[1] == "few":
            table += "\\midrule\n"
            table += f"\\textit{{Few-shot}} \\\\\n"
            table += "\\midrule\n"
        # if prev_minor_header != panel[1] and panel[1] not in ["default", "zero", "few"]:
        #     if panel[1] == "1VAN":
        #         panel_name = "Full-token Loss (FTL)"
        #     elif panel[1] == "2ATL":
        #         panel_name = "Target-only Loss (TOL)"
        #     elif panel[1] == "3Classifier":
        #         panel_name = "Encoder-style"
        #     table += f"\\textbf{{{panel_name}}} \\\\\n"
        #     table += "\\hline\n"
        
        variant_name = ""
        if panel[0] == "SLMs":
            
            if panel[1] == "1VAN":
                variant_name = " (FTL)"
            elif panel[1] == "2ATL":
                variant_name = " (TOL)"
            elif panel[1] == "3Classifier":
                variant_name = " (ES)"

        if panel[0] == "LLMs":
            model_name = model_name.replace("1(instruct)", "(instruct)").replace("2(verbose)", "(verbose)")

        table += f"{model_name}{variant_name} & " + " & ".join(cells) + " \\\\\n"
    
        prev_main_header = panel[0]
        prev_minor_header = panel[1]

    table += "\\bottomrule\n"
    table += "\\end{tabular}\n"

    print("\n\n")
    print(table)
    print("\n\n")

def main():

    configs = {
        "metric": "f1",
    }

    # load all models
    rows, overall_avg, model_panel_avgs, out_stds = load_all_models(configs, SLM_DIR)

    # print latex table
    print("="*80)
    print(f"Metric: {configs['metric']}")
    print("="*80)
    latex_table(rows, overall_avg, model_panel_avgs, out_stds)
if __name__ == "__main__":
    main()
