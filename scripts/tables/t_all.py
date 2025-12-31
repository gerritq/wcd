import os
import sys
from collections import defaultdict
from pathlib import Path
from argparse import Namespace
from utils import load_metrics, find_best_metric_from_hyperparameter_search, LANGS, LANG_ORDER, MODEL_DISPLAY_NAMES
import numpy as np
import pandas as pd

# ------------------------------------------------------------------------------------------
# configs
# ------------------------------------------------------------------------------------------

BASE_DIR = os.getenv("BASE_WCD")
SLM_DIR = os.path.join(BASE_DIR, "data/exp1")

METRIC = "f1" 
COUNT = defaultdict(list)

def get_resource(lang: str) -> str:
    for resource, langs in LANGS.items():
        if lang in langs:
            return resource
    return "unknown"

def llm_variant_display_name(x: str) -> str:
    if x in ["few_verbose", "zero_verbose"]:
        return "Verbose"
    if x in ["few_instruct", "zero_instruct"]:
        return "Instruct"

def load_llms(rows: dict,
              meta_1: dict,
              meta_files: list[Path],
              ) -> dict[str, dict]:
    

    variant = ""
    if meta_1['shots'] == True and meta_1['verbose'] == True:
        variant = "few_verbose"
    elif meta_1['shots'] == True and meta_1['verbose'] == False:
        variant = "few_instruct"
    elif meta_1['shots'] == False and meta_1['verbose'] == True:
        variant = "zero_verbose"
    else:
        variant = "zero_instruct"

    # COLLECT 
    rows.append({"model_name": MODEL_DISPLAY_NAMES[meta_1["model_name"]],
                 "model_type": "LLM",
                 "variant": variant,
                 "metric": meta_1["test_metrics"][METRIC],
                 "lang": meta_1["lang"],
                 'run_dir': meta_files[0].parent.name,
                 "seed": 42,
                 })
    
    return rows

def load_slms(meta_files: list[Path],
                  rows: dict,
                  ) -> dict[str, dict]:
    
    meta_1 = load_metrics(meta_files[0])
    
    
    if len(meta_files) == 1:    
        if meta_1['seed'] in [2025, 2026]:
            best_metric = meta_1["test_metrics"][-1]['metrics'][METRIC]

            rows.append({"model_name": MODEL_DISPLAY_NAMES[meta_1["model_name"]],
                         "model_type": "SLM",
                         "variant": "ES" if meta_1["model_type"] == "clf" else ("TOL" if meta_1["atl"] else "FTL"),
                         "metric": best_metric,
                         "lang": meta_1["lang"],
                         "run_dir": meta_files[0].parent.name,
                         "seed": meta_1["seed"],
                        })
        
    if len(meta_files) == 6:
        best_metric = find_best_metric_from_hyperparameter_search(all_meta_file_paths=meta_files, metric=METRIC)
        rows.append({"model_name": MODEL_DISPLAY_NAMES[meta_1["model_name"]],
                        "model_type": "SLM",
                        "variant": "ES" if meta_1["model_type"] == "clf" else ("TOL" if meta_1["atl"] else "FTL"),
                        "metric": best_metric,
                        "lang": meta_1["lang"],
                        "run_dir": meta_files[0].parent.name,
                        "seed": "hp",
                    })
    return rows

def load_plms(meta_files: list[Path],
                  rows: dict,
                  ) -> dict[str, dict]:
    
    meta_1 = load_metrics(meta_files[0])
        
    if len(meta_files) == 6:
        best_metric = find_best_metric_from_hyperparameter_search(all_meta_file_paths=meta_files, metric=METRIC)
        rows.append({"model_name": MODEL_DISPLAY_NAMES[meta_1["model_name"]],
                    "model_type": "PLM",
                    "variant": "default",
                    "metric": best_metric,
                    "lang": meta_1["lang"],
                    "run_dir": meta_files[0].parent.name,
                    "seed": meta_1["seed"],
                    })

    return rows


def load_all_models(path: str) -> dict[str, dict]:
    root = Path(path)
    rows = []

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
                rows = load_llms(rows=rows, meta_1=meta_1, meta_files=meta_files)
                continue

            if meta_1["model_type"] in ["slm", "clf"]:
                rows = load_slms(rows=rows, meta_files=meta_files)
                continue

            if meta_1["model_type"] == "plm":
                rows = load_plms(rows=rows, meta_files=meta_files)
                continue

    return rows
  
def create_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)

    # CHECK TABLE
    df_check = (df
                .groupby(['model_type', 'model_name', 'variant', 'lang'])
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
    df_check = df_check.sort_values(by=['model_type', 'model_name', 'variant', 'lang'])
    df_check.to_excel(f"checks/all_check_{METRIC}.xlsx", index=False)

    print("="*80)
    print("CHECK DF")
    print("="*80)
    print(df_check.head())

    # ACG per resource level
    # merge resouce level to df_check
    df_check['resource'] = df_check['lang'].apply(get_resource)
    df_avg = (df_check
              .groupby(['model_type', 'model_name', 'variant', 'resource'])
              .agg(
                  metric_mean=("metric_mean", "mean"),
                  metric_std=("metric_mean", "std"),
                  metric_count=("metric_mean", "count"),
              )
              .reset_index())
    
    df_avg = df_avg.sort_values(by=['model_type', 'model_name', 'variant', 'resource'])

    # multiply all values by 100
    for col in df_avg.columns:
        if col.startswith("metric_mean") or col.startswith("metric_std"):
            df_avg[col] = df_avg[col] * 100.0

    print("="*80)
    print("AVG DF")
    print("="*80)
    print(df_avg.head())


    # MODEL AVG
    df_model_avg = (df
                .groupby(['model_type', 'model_name', 'variant'])
                .agg(
                    metric_mean=("metric", "mean"),
                    metric_std=("metric", "std")
                ).reset_index())
    
    # multiply all values by 100
    for col in df_model_avg.columns:
        if col.startswith("metric_mean") or col.startswith("metric_std"):
            df_model_avg[col] = df_model_avg[col] * 100.0

    print("="*80)
    print("MODEL AVG DF")
    print("="*80)
    print(df_model_avg.head())


    # MAIN PIVOT TABLE
    pivot = df_check.pivot_table(index=['model_type', 'model_name', 'variant'], columns='lang', values=['metric_mean', 'metric_std'])
    
    # remove multi columns
    pivot.columns = [f"{col[0]}_{col[1]}" for col in pivot.columns]

    # set index to columns
    pivot = pivot.reset_index()

    # multiply all values by 100
    for col in pivot.columns:
        if col.startswith("metric_mean_") or col.startswith("metric_std_"):
            pivot[col] = pivot[col] * 100.0

    # KEEP ONLY BEST PERFORMING SLM PER VARIANT (by overall mean)
    best_slm_variants = (
        df_model_avg[df_model_avg["model_type"] == "SLM"]
        .sort_values(["model_name", "metric_mean"], ascending=[True, False])
        .drop_duplicates(subset=["model_name"], keep="first")[["model_name", "variant"]]
    )

    # Filter pivot:
    pivot = pivot[
        (pivot["model_type"] != "SLM")
        | pivot.set_index(["model_name", "variant"]).index.isin(
            best_slm_variants.set_index(["model_name", "variant"]).index
        )
    ].copy()

    
    # sort model_types
    pivot['model_type'] = pd.Categorical(pivot['model_type'], categories=["LLM", "PLM", "SLM"], ordered=True)
    pivot['variant'] = pd.Categorical(pivot['variant'], categories=["zero_instruct", "zero_verbose", "few_instruct", "few_verbose", "FTL", "TOL", "ES", "default"], ordered=True)
    pivot = pivot.sort_values(by=['model_type', 'model_name', 'variant'])

    
    # sort llms only
    pivot['_name_key'] = pivot['model_name']
    pivot[['_shots_key', '_prompt_key']] = pivot['variant'].astype(str).str.split('_', n=1, expand=True)

    llm = pivot['model_type'] == 'LLM'

    pivot['_variant_key'] = None
    pivot.loc[llm, '_variant_key'] = pivot.loc[llm, '_shots_key'].map({'zero': 0, 'few': 1})

    pivot = (
        pivot
        .sort_values(['model_type', '_variant_key', '_name_key'], na_position='last')
        .drop(columns=['_name_key', '_variant_key', '_shots_key', '_prompt_key'])
    )

    print("="*80)
    print("PIVOT DF")
    print("="*80)
    print(pivot.head())


    return pivot, df_avg, df_model_avg

def latex_table(df, df_avg, df_model_avg):
    # print LaTeX table
    table = "\n\n"
    colspec = "l" + "C" * (len(LANG_ORDER) + 4)
    
    header = " & \\multicolumn{" + str(len(LANGS['high'])+1) + "}{c}{\\textbf{High Resource}} & \\multicolumn{" + str(len(LANGS['medium'])+1) + "}{c}{\\textbf{Medium Resource}} & \\multicolumn{" + str(len(LANGS['low'])+1) + "}{c}{\\textbf{Low Resource}} \\\\"
    header += "\\cmidrule(lr){2-" + str(len(LANGS['high'])+2) + "}  \\cmidrule(lr){" + str(len(LANGS['high'])+3) + "-" + str(len(LANGS['high'])+len(LANGS['medium'])+3) + "}  \\cmidrule(lr){" + str(len(LANGS['high'])+len(LANGS['medium'])+4) + "-" + str(len(LANGS['high'])+len(LANGS['medium'])+len(LANGS['low'])+4) + "}"
    header += " & " + " & ".join([f"\\textbf{{{l}}}" for l in LANGS['high']]) + " & \\textbf{Avg} & "  + " & ".join([f"\\textbf{{{l}}}" for l in LANGS['medium']]) + " & \\textbf{Avg} & " + " & ".join([f"\\textbf{{{l}}}" for l in LANGS['low']]) + " & \\textbf{Avg} & \\textbf{Avg} \\\\"
    
    table += "\\begin{tabular}{" + colspec + "}\n"
    table += header + "\n"
    table += "\\toprule\n"

    prev_main_header=None
    prev_minor_header=None
    for _, row in df.iterrows():
        model_type = row['model_type']
        model_name = row['model_name']
        variant = row['variant']
        cells = []
        for resource, resource_set in [('high', LANGS['high']), ('medium', LANGS['medium']), ('low', LANGS['low'])]:
            for l in resource_set:
                mean_col = f"metric_mean_{l}"
                std_col = f"metric_std_{l}"
                v = row[mean_col] if mean_col in row else None
                std = row[std_col] if std_col in row else None
                
                # max lang column
                max_lang = df[f"metric_mean_{l}"].max()
                second_max_lang = df[df[f"metric_mean_{l}"] < max_lang][f"metric_mean_{l}"].max()

                # cell formatting
                if v == max_lang:
                    v_cell = f"\\textbf{{{v:.2f}}}"
                elif v == second_max_lang:
                    v_cell = f"\\underline{{{v:.2f}}}"
                else:
                    v_cell = f"{v:.2f}"
                
                if model_type in ["SLM", "PLM"]:
                    cells.append(f"{v_cell} \\scriptsize{{($\\pm{std:.2f}$)}}" if std is not None else v_cell)
                else:
                    cells.append(v_cell)

            
            # average cell
            max_avg = df_avg[df_avg['resource'] == resource]['metric_mean'].max()
            second_max_avg = df_avg[(df_avg['resource'] == resource) & (df_avg['metric_mean'] < max_avg)]['metric_mean'].max()

            avg_row = df_avg[(df_avg['model_name'] == model_name) & (df_avg['variant'] == variant) & (df_avg['resource'] == resource)]['metric_mean'].values[0]
            avg_std = df_avg[(df_avg['model_name'] == model_name) & (df_avg['variant'] == variant) & (df_avg['resource'] == resource)]['metric_std'].values[0]

            if avg_row == max_avg:
                v_cell = f"\\cellcolor{{gray!25}} \\textbf{{{avg_row:.2f}}}"
            elif avg_row == second_max_avg:
                v_cell = f"\\cellcolor{{gray!25}} \\underline{{{avg_row:.2f}}}"
            else:
                v_cell = f"\\cellcolor{{gray!25}} {avg_row:.2f}"
            
        
            cells.append(f"{v_cell} \\scriptsize{{($\\pm{avg_std:.2f}$)}}")
            

        # add overall average cell
        model_avg_row = df_model_avg[(df_model_avg['model_name'] == model_name) & (df_model_avg['variant'] == variant)]['metric_mean'].values[0]
        model_avg_std = df_model_avg[(df_model_avg['model_name'] == model_name) & (df_model_avg['variant'] == variant)]['metric_std'].values[0]

        overall_max = df_model_avg['metric_mean'].max()
        overall_second_max = df_model_avg[df_model_avg['metric_mean'] < overall_max]['metric_mean'].max()
        if model_avg_row == overall_max:
            v_cell = f"\\textbf{{{model_avg_row:.2f}}}"
        elif model_avg_row == overall_second_max:
            v_cell = f"\\underline{{{model_avg_row:.2f}}}"
        else:
            v_cell = f"{model_avg_row:.2f}" 
        
        
        cells.append(f"{v_cell} \\scriptsize{{($\\pm{model_avg_std:.2f}$)}}")

        
        # major header
        if prev_main_header is None and model_type == "LLM":
                table += f"\\multicolumn{{{len(LANG_ORDER)+5}}}{{c}}{{\\textbf{{Decoder-based LLMs}}}} \\\\\n"
                table += "\\midrule\n"
        if prev_main_header and prev_main_header != model_type:
            if model_type == "PLM":
                table += "\\midrule\n"
                table += f"\\multicolumn{{{len(LANG_ORDER)+5}}}{{c}}{{\\textbf{{Encoder-based PLMs}}}} \\\\\n"
                table += "\\midrule\n"
            elif model_type == "SLM":
                table += "\\midrule\n"    
                table += f"\\multicolumn{{{len(LANG_ORDER)+5}}}{{c}}{{\\textbf{{Decoder-based SLMs}}}} \\\\\n"   
                table += "\\midrule\n"

        # minor header
        if prev_minor_header is None and variant in ["zero_instruct", "zero_verbose"]:
            table += f"\\textit{{Zero-shot}} \\\\\n"
            table += "\\midrule\n"
        if prev_minor_header and prev_minor_header.startswith("zero") and variant.startswith("few"):
            table += "\\midrule\n"
            table += f"\\textit{{Few-shot}} \\\\\n"
            table += "\\midrule\n"

        if model_type == "LLM":
            display_name = f"{model_name} ({llm_variant_display_name(variant)})"
        elif model_type == "PLM":
            display_name = f"{model_name}"
        else:   
            display_name = f"{model_name} ({variant})"
        table += f"{display_name} & " + " & ".join(cells) + " \\\\\n"
        
        prev_main_header = model_type
        prev_minor_header = variant

    table += "\\bottomrule\n"
    table += "\\end{tabular}\n"
    print("\n\n")
    print(table)
    print("\n\n")


def main():

    # load all models
    rows = load_all_models(SLM_DIR)
    # create dataframe
    df = create_df(rows)

    # print latex table
    print("="*80)
    print(f"Metric: {METRIC}")
    print("="*80)
    latex_table(df[0], df[1], df[2])
if __name__ == "__main__":
    main()
