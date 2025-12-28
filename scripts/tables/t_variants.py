import os
from collections import defaultdict
from pathlib import Path
from utils import load_metrics, find_best_metric_from_hyperparameter_search, LANGS, LANG_ORDER, MODEL_DISPLAY_NAMES
import pandas as pd

import sys

# ------------------------------------------------------------------------------------------
# configs
# ------------------------------------------------------------------------------------------

BASE_DIR = os.getenv("BASE_WCD")
SLM_DIR = os.path.join(BASE_DIR, "data/exp1")

COUNT = defaultdict(list)


def get_resource(lang: str) -> str:
    for resource, langs in LANGS.items():
        if lang in langs:
            return resource
    return "unknown"


def load_slms(configs: dict, 
                  meta_files: list[Path],
                  rows: dict,
                  ) -> dict[str, dict]:
    
    meta_1 = load_metrics(meta_files[0])
    
    # panel
    if meta_1['model_type'] == "clf":
        variant = "ES"
    else:
        if meta_1["atl"]:
            variant = "TOL"
        else:
            variant = "FTL"
    
    model_name = MODEL_DISPLAY_NAMES[meta_1["model_name"]]
    key = (model_name, variant)

    if len(meta_files) == 1:    
        if meta_1['seed'] in [2025, 2026]:
            best_metric = meta_1["test_metrics"][-1]['metrics'][configs['metric']]

            rows.append({'model_name': model_name,
                         'variant': variant,
                         'lang': meta_1["lang"],
                         'seed': meta_1['seed'],
                         'metric': best_metric,
                         'run_dir': meta_files[0].parent.name,
                         })
            COUNT[(model_name, meta_1['model_type'], meta_1['lang'], meta_1['atl'])].append((meta_1['seed'], meta_files[0].parent.name))  
            
        
    if len(meta_files) == 6:
        best_metric = find_best_metric_from_hyperparameter_search(all_meta_file_paths=meta_files, metric=configs['metric'])
        rows.append({'model_name': model_name,
                     'variant': variant,
                     'lang': meta_1["lang"],
                     'seed': "hp",
                     'metric': best_metric,
                     'run_dir': meta_files[0].parent.name,
                     })

        COUNT[(model_name, meta_1['model_type'], meta_1['lang'], meta_1['atl'])].append(("hp", meta_files[0].parent.name))
    return rows


def load_all_models(configs: dict, path: str) -> dict[str, dict]:
    root = Path(path)
    rows = []
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

    # print("="*20)
    # print("="*20)
    # print("OVERVIEW OF COLLECTED FILES")
    # print("="*20)
    # print("="*20)
    # # sortt by lang
    # sorted_count = dict(sorted(count.items(), key=lambda x: (x[0][1], x[0][0])))
    # prev_lang = None
    # for k,v in sorted_count.items():
    #     if not prev_lang or k[1] != prev_lang:
    #         print("="*20)
    #         print(f"LANG {k[1]}")
    #         print("="*20)
    #         prev_lang = k[1]

    #     if len(k) == 4 and k[2] in ['clf', "slm"]:
    #         print(f"LANG: {k[1]} | MODEL {k[0]} | TYPE {k[2]} | LOSS {'ATL' if k[3] else 'VAN'}: {len(v)} run(s) -> {sorted(v, key=lambda x: str(x[0]))}")
    #         if len(v) > 3:
    #             print("WARNING: TOO MANY RUNS!")
    #         if set([run[0] for run in v]) != set([2025, 2026, 'hp']):
    #             print("WARNING: INCORRECT NUMBER OF RUNS!")
    #         print("")
    #     elif len(k) == 4:
    #         print(f"LANG: {k[1]} | MODEL {k[0]} | {'Few-shot' if k[2] else 'Zero-shot'} | {'Verbose' if k[3] else 'Instruct'}: {len(v)} runs -> {sorted(v, key=lambda x: str(x[0]))}")
    #         print("")
    #     elif len(k) == 2:
    #         print(f"LANG: {k[1]} | MODEL {k[0]}: {len(v)} runs -> {sorted(v, key=lambda x: str(x[0]))}")
    #         if set([run[0] for run in v]) != set([2025, 2026, 42]):
    #             print("WARNING: INCORRECT NUMBER OF RUNS!")
    #         print("")
    # print("="*20)

    return rows


def create_df(rows: list):
    df = pd.DataFrame(rows)

    # CHECK TABLE
    df_check = (df
                .groupby(['model_name', 'variant', 'lang'])
                .agg(
                    metric_mean=("metric", "mean"),
                    metric_std=("metric", "std"),
                    metric_count=("metric", "count"),
                    metrics_lst=("metric", list),
                    sources=("run_dir", list),
                    seeds_lst=("seed", list),
        )
        .reset_index())
    
    # sort and sace
    df_check = df_check.sort_values(by=['model_name', 'variant', 'lang'])
    df_check.to_excel("checks/atl.xlsx", index=False)

    print(df_check.head())

    # MAIN PIVOT TABLE

    # prtivot table for latex
    pivot = df_check.pivot_table(index=['model_name', 'variant'], columns='lang', values=['metric_mean', 'metric_std'])
    
    # remove multi columns
    pivot.columns = [f"{col[0]}_{col[1]}" for col in pivot.columns]

    # set index to columns
    pivot = pivot.reset_index()

    # multiply all values by 100
    for col in pivot.columns:
        if col.startswith("metric_mean_") or col.startswith("metric_std_"):
            pivot[col] = pivot[col] * 100.0

    # order variants 
    pivot['variant'] = pd.Categorical(pivot['variant'], categories=['FTL', 'TOL', 'ES'], ordered=True)
    pivot = pivot.sort_values(by=['model_name', 'variant'])

    # AVG PER RESOURCE LEVEL TABLE

    # merge resouce level to df_check
    df_check['resource'] = df_check['lang'].apply(get_resource)
    df_avg = (df_check
              .groupby(['model_name', 'variant', 'resource'])
              .agg(
                  metric_mean=("metric_mean", "mean"),
                  metric_std=("metric_mean", "std"),
                  metric_count=("metric_mean", "count"),
              )
              .reset_index())
    
    df_avg = df_avg.sort_values(by=['model_name', 'variant', 'resource'])

    # multiply by 100
    df_avg['metric_mean'] = df_avg['metric_mean'] * 100.0
    df_avg['metric_std'] = df_avg['metric_std'] * 100.0

    print("="*60)
    print("AVERAGE METRICS PER RESOURCE LEVEL")
    print(df_avg.head(5))
    print("="*60)

    print("="*60)
    print("MAIN TABLE PIVOT")
    print(pivot.head(5))
    print("="*60)

    
    return pivot, df_avg


def latex_table(pivot, df_avg):

    # print LaTeX table
    table = "\n\n"
    colspec = "ll" + "C" * (len(LANG_ORDER) + 3)
    
    header = " & & \\multicolumn{" + str(len(LANGS['high'])+1) + "}{c}{\\textbf{High Resource}} & \\multicolumn{" + str(len(LANGS['medium'])+1) + "}{c}{\\textbf{Medium Resource}} & \\multicolumn{" + str(len(LANGS['low'])+1) + "}{c}{\\textbf{Low Resource}} \\\\"
    header += "\\cmidrule(lr){3-" + str(len(LANGS['high'])+3) + "}  \\cmidrule(lr){" + str(len(LANGS['high'])+4) + "-" + str(len(LANGS['high'])+len(LANGS['medium'])+4) + "}  \\cmidrule(lr){" + str(len(LANGS['high'])+len(LANGS['medium'])+5) + "-" + str(len(LANGS['high'])+len(LANGS['medium'])+len(LANGS['low'])+5) + "}"
    header += "\\textbf{Model} & \\textbf{SFT} & " + " & ".join([f"\\textbf{{{l}}}" for l in LANGS["high"]]) + "& \\textbf{Avg}" + " & " + " & ".join([f"\\textbf{{{l}}}" for l in LANGS["medium"]]) + "& \\textbf{Avg}" + " & " + " & ".join([f"\\textbf{{{l}}}" for l in LANGS["low"]]) + "& \\textbf{Avg} \\\\"

    table += "\\begin{tabular}{" + colspec + "}\n"
    table += header + "\n"
    table += "\\toprule\n"

    prev_model = None
    for _, row in pivot.iterrows():
        model_name = row['model_name']
        variant = row['variant']
        cells = []
        for resource, resource_set in [('high', LANGS['high']), ('medium', LANGS['medium']), ('low', LANGS['low'])]:
            for l in resource_set:
                
                max_lang_metric = pivot[(pivot['model_name'] == model_name)][f"metric_mean_{l}"].max()
                mean_col = f"metric_mean_{l}"
                std_col = f"metric_std_{l}"
                if mean_col in row and not pd.isna(row[mean_col]):
                    if row[mean_col] == max_lang_metric:
                        # make only value bold, not the std
                        v = f"\\textbf{{{row[mean_col]:.2f}}} \\scriptsize{{$(\\pm {row[std_col]:.2f})$}}"
                    else:
                        v = f"{row[mean_col]:.2f}  \\scriptsize{{$(\\pm{row[std_col]:.2f})$}}"  
                else:
                    v = "--"
                cells.append(v)
                
            # average cell  
            avg_metric = df_avg[(df_avg['model_name'] == model_name) & (df_avg['variant'] == variant) & (df_avg['resource'] == resource)]['metric_mean']
            avg_std = df_avg[(df_avg['model_name'] == model_name) & (df_avg['variant'] == variant) & (df_avg['resource'] == resource)]['metric_std']

            max_avg_metric = df_avg[(df_avg['resource'] == resource) & (df_avg['model_name'] == model_name)]['metric_mean'].max()
            if not avg_metric.empty and avg_metric.values[0] == max_avg_metric:
                cells.append(f"\\cellcolor{{gray!25}} \\textbf{{{avg_metric.values[0]:.2f}}} \\scriptsize{{$(\\pm{avg_std.values[0]:.2f})$}}" if not avg_metric.empty else "--")
            else:
                cells.append(f"\\cellcolor{{gray!25}} {avg_metric.values[0]:.2f} \\scriptsize{{$(\\pm{avg_std.values[0]:.2f})$}}" if not avg_metric.empty else "--")
            
        if prev_model is None:
            table += f"\\multirow{{3}}{{*}}[-1.8ex]{{{model_name}}} & " + f" {variant} & " + " & ".join(cells) + " \\\\\n"
        elif prev_model != model_name:
            table += "\\midrule\n"
            table += f"\\multirow{{3}}{{*}}[-1.8ex]{{{model_name}}} & " + f" {variant} & " + " & ".join(cells) + " \\\\\n"
        else:
            table += f" & " + f"{variant} & "+ " & ".join(cells) + " \\\\\n"
            
        prev_model = model_name

    table += "\\bottomrule\n"
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
    df = create_df(rows)

    # print latex table

    print("="*80)
    print(f"Metric: {configs['metric']}")
    print("="*80)
    latex_table(df[0], df[1])
if __name__ == "__main__":
    main()
