import os
from collections import defaultdict
from pathlib import Path
from utils import load_metrics, find_best_metric_from_hyperparameter_search, LANGS, LANG_ORDER, MODEL_DISPLAY_NAMES

# ------------------------------------------------------------------------------------------
# configs
# ------------------------------------------------------------------------------------------

BASE_DIR = os.getenv("BASE_WCD")
SLM_DIR = os.path.join(BASE_DIR, "data/exp1")

COUNT = defaultdict(list)

def load_slms(configs: dict, 
                  meta_files: list[Path],
                  rows: dict,
                  ) -> dict[str, dict]:
    
    meta_1 = load_metrics(meta_files[0])
    
    # panel
    if meta_1['model_type'] == "clf":
        panel = "3ES"
    else:
        if meta_1["atl"]:
            panel = "2TOL"
        else:
            panel = "1FTL"
    
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

            if meta_1["model_type"] in ["slm", "clf"]:
                rows = load_slms(configs=configs, rows=rows, meta_files=meta_files)
                hp_seed = "hp" if len(meta_files) == 6 else meta_1['seed']
                count[(meta_1['model_name'], meta_1['lang'], meta_1['model_type'], meta_1['atl'])].append((hp_seed, meta_files[0].parent.name))  
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
    for (model_name, panel), metrics in rows.items():
        for lang, v in metrics.items():
            if isinstance(v, list) and len(v) > 0:
                avg_v = sum(v) / len(v)
                out_rows[(model_name, panel)][lang] = avg_v
            else:
                out_rows[(model_name, panel)][lang] = None

    # print("FINAL ROWS")
    # for k,v in out_rows.items():
    #     print(f"{k}: {v}")
    #     print()
    # return out_rows

    # sort rows


    final_rows = {}
    # Panel 1: Aya
    llms = {k: v for k, v in out_rows.items() if  "aya" in k[0].lower()}
    llms = dict(sorted(llms.items(), key=lambda x: (x[0][1])))
    final_rows.update(llms)
    # PLMs
    plms = {k: v for k, v in out_rows.items() if "llama" in k[0].lower()}
    plms = dict(sorted(plms.items(), key=lambda x: (x[0][1])))
    final_rows.update(plms)
    # SLMs
    slms = {k: v for k, v in out_rows.items() if "qwen" in k[0].lower()}
    slms = dict(sorted(slms.items(), key=lambda x: (x[0][1])))
    final_rows.update(slms)

    return final_rows

def latex_table(rows):

    n_panels = 3

    
    all_models = set([k[0] for k in rows.keys()])
    all_panels = set([k[1] for k in rows.keys()])

    max_langs = {(model_name, lang): 0.0 for model_name in all_models for lang in LANG_ORDER}
    max_avg = {(model_name, panel, resource): [] for model_name in all_models for resource in ['high', 'medium', 'low'] for panel in all_panels}

    for model in all_models:

        for lang in LANG_ORDER:
            for panel in all_panels:
                key = (model, panel)
                if key in rows.keys():
                    v = rows[key][lang]

                    if isinstance(v, (int, float)):

                        # avgs
                        if lang in LANGS['high']:
                            max_avg[(model, panel, 'high')].append(v)
                        elif lang in LANGS['medium']:
                            max_avg[(model, panel, 'medium')].append(v)
                        elif lang in LANGS['low']:
                            max_avg[(model, panel, 'low')].append(v)
                        if v > max_langs[(model, lang)]:
                            max_langs[(model, lang)] = v

    # generate max avgs
    
    max_avg = {(model, panel, resource): (sum(v) / len(v) if len(v) > 0 else 0.0) for (model, panel, resource), v in max_avg.items()}
    max_avg = {(model, resource): max([max_avg[(model, panel, resource)] for panel in all_panels]) for model in all_models for resource in ['high', 'medium', 'low']}

    # print LaTeX table
    table = "\n\n"
    colspec = "ll" + "c" * (len(LANG_ORDER) + 3)
    
    header = " & & \\multicolumn{" + str(len(LANGS['high'])+1) + "}{c}{\\textbf{High Resource}} & \\multicolumn{" + str(len(LANGS['medium'])+1) + "}{c}{\\textbf{Medium Resource}} & \\multicolumn{" + str(len(LANGS['low'])+1) + "}{c}{\\textbf{Low Resource}} \\\\"
    header += "\\cmidrule(lr){3-" + str(len(LANGS['high'])+3) + "}  \\cmidrule(lr){" + str(len(LANGS['high'])+4) + "-" + str(len(LANGS['high'])+len(LANGS['medium'])+4) + "}  \\cmidrule(lr){" + str(len(LANGS['high'])+len(LANGS['medium'])+5) + "-" + str(len(LANGS['high'])+len(LANGS['medium'])+len(LANGS['low'])+5) + "}"
    header += "\\textbf{Model} & \\textbf{SFT Variant} & " + " & ".join([f"\\textbf{{{l}}}" for l in LANGS["high"]]) + "& \\textbf{Avg}" + " & " + " & ".join([f"\\textbf{{{l}}}" for l in LANGS["medium"]]) + "& \\textbf{Avg}" + " & " + " & ".join([f"\\textbf{{{l}}}" for l in LANGS["low"]]) + "& \\textbf{Avg} \\\\"

    table += "\\begin{tabular}{" + colspec + "}\n"
    table += header + "\n"
    table += "\\toprule\n"

    
    prev_model=None
    for (model_name, panel), metrics in rows.items():
        cells = []
        for resource, resource_set in [('high', LANGS['high']), ('medium', LANGS['medium']), ('low', LANGS['low'])]:
            for l in resource_set:
                if l not in metrics.keys():
                    v = None
                else:
                    if metrics[l] == max_langs[(model_name, l)]:
                        v = f"\\textbf{{{metrics[l]*100:.2f}}}"
                    elif isinstance(metrics[l], (int, float)):  
                        v = f"{metrics[l]*100:.2f}"
                    else:
                        v = "--"
                        
                cells.append(v)
                
            # average cell  
            lang_values = [metrics[l] for l in resource_set if l in metrics.keys() and isinstance(metrics[l], (int, float))]
            if len(lang_values) > 0:
                avg = sum(lang_values) / len(lang_values)
            
                if avg == max_avg[(model_name, resource)]:
                    cells.append(f"\\cellcolor{{gray!25}} \\textbf{{{avg*100:.2f}}}")
                else:
                    cells.append(f"\\cellcolor{{gray!25}} {avg*100:.2f}")
            else:   
                cells.append("--")
        
        if prev_model is None:
            table += f"\\multirow{{{n_panels}}}{{*}}{{{model_name}}} & " + f" {panel[1:]} & " + " & ".join(cells) + " \\\\\n"
        elif prev_model != model_name:
            table += "\\midrule\n"
            table += f"\\multirow{{{n_panels}}}{{*}}{{{model_name}}} & " + f" {panel[1:]} & " + " & ".join(cells) + " \\\\\n"
        else:
            table += f" & " + f"{panel[1:]} & "+ " & ".join(cells) + " \\\\\n"
            
        prev_model = model_name

    table += "\\bottomrule\n"
    table += "\\end{tabular}\n"

    print(table)

def main():

    configs = {
        "metric": "accuracy",
    }

    # load all models
    rows = load_all_models(configs, SLM_DIR)

    # print latex table

    print("="*80)
    print(f"Metric: {configs['metric']}")
    print("="*80)
    latex_table(rows)
if __name__ == "__main__":
    main()
