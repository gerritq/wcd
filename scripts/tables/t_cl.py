import os
from pdb import run

import json
import glob
from collections import defaultdict
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from utils import MODEL_DISPLAY_NAMES, LANG_ORDER, LANGS, load_metrics
import pandas as pd

# ----------------------------------------------------------------------
# Configs
# ----------------------------------------------------------------------

BASE_DIR = os.getenv("BASE_WCD")
EXP2_DIR = os.path.join(BASE_DIR, "data/exp2/eval")

METRIC="f1"
COUNT = defaultdict(list)

SHOTS = [0, 50, 100, 250, 500]

def load_zero(rows: list[dict],
              meta_1: dict,
              ) -> dict[str, dict]:

    # COLLECT
    model_name = MODEL_DISPLAY_NAMES[meta_1["model_name"]]
    if meta_1["model_type"] == "clf":
        model_name += f" (ES) {1 if meta_1['lower_lr'] else 0}"
    if meta_1["model_type"] == "slm" and meta_1['atl'] == True:
        model_name += f" (TOL) {1 if meta_1['lower_lr'] else 0}"
    
    rows.append({"model_name": model_name,
                 "lang": meta_1["lang"],
                 "lang_setting": meta_1["lang_setting"],
                 "shots": 0,
                 "metric": meta_1["test_metrics_0_shot"][METRIC]
                })
    return rows


def load_few(rows: list[dict],
             all_meta_files: list[Path],
              ) -> dict[str, dict]:

    # COLLECT
    for meta_file in all_meta_files:
        meta_1 = load_metrics(meta_file)    
        model_name = MODEL_DISPLAY_NAMES[meta_1["model_name"]]
        if meta_1["model_type"] == "clf":
            model_name += f" (ES) {1 if meta_1['lower_lr'] else 0}"
        if meta_1["model_type"] == "slm" and meta_1['atl'] == True:
            model_name += f" (TOL) {1 if meta_1['lower_lr'] else 0}"
        
        rows.append({"model_name": model_name,
                    "lang": meta_1["lang"],
                    "lang_setting": meta_1["lang_setting"],
                    "shots": meta_1["training_size"],
                    "metric": meta_1["test_metrics"][-1]['metrics'][METRIC]
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

            if len(meta_files) == 1:
                meta_1 = load_metrics(meta_files[0])
                rows = load_zero(rows, meta_1)

            if len(meta_files) == 4:
                rows = load_few(rows=rows, all_meta_files=meta_files)


    return rows

def create_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)

    pivot = (
    df
    .pivot_table(
        index=["lang", "model_name"],
        columns=["lang_setting", "shots"],
        values="metric",
        aggfunc="mean"
    )
    .sort_index(axis=1, level=[0, 1])  # optional: clean column order
    )

    
    pivot.columns = [
        c if isinstance(c, str) else f"{c[0]}_{c[1]}"
        for c in pivot.columns
    ]
    pivot = pivot.reset_index() 

    id_cols = ["lang", "model_name"]
    trans_cols = [c for c in pivot.columns if c.startswith("translation_")]
    main_cols  = [c for c in pivot.columns if c.startswith("main_")]

    pivot = pivot[id_cols + trans_cols + main_cols]

    # sort LANG_ORDER
    pivot['lang'] = pd.Categorical(pivot['lang'], categories=LANG_ORDER, ordered=True)
    pivot = pivot.sort_values(['lang', 'model_name']).reset_index(drop=True)

    return pivot

def generate_figure(df: pd.DataFrame):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    plt.rcParams.update({"font.size": 12})

    fig = plt.figure(
        figsize=(12, 6),
        constrained_layout=True,
    )

    lang_settings = ["translation", "main"]
    row_titles = ["Pseudo-parallel", "In-the-wild"]
    col_titles = ["Medium Resource Languages", "Low Resource Languages"]

    x = SHOTS

    # ------------------------------------------------------------------
    # Consistent styling
    # ------------------------------------------------------------------
    models = sorted(df["model_name"].unique())
    cmap = cm.get_cmap("tab10", len(models))
    color_map = {m: cmap(i) for i, m in enumerate(models)}

    marker_map = {
        "translation": "o",
        "main": "s",
    }

    # ------------------------------------------------------------------
    # Subfigures
    # ------------------------------------------------------------------
    subfigs = fig.subfigures(nrows=2, ncols=1)

    for row_i, (subfig, lang_setting) in enumerate(zip(subfigs, lang_settings)):
        subfig.suptitle(row_titles[row_i], fontsize=20)

        axs = subfig.subplots(nrows=1, ncols=2, sharex=True, sharey=True)

        for col_i, resource in enumerate(["medium", "low"]):
            ax = axs[col_i]

            langs = LANGS[resource]
            df_subset = df[df["lang"].isin(langs)]

            for model_name in models:
                df_model = df_subset[df_subset["model_name"] == model_name]
                if df_model.empty:
                    continue

                y = [df_model.get(f"{lang_setting}_{shot}").mean() for shot in x]

                ax.plot(
                    x,
                    y,
                    marker=marker_map[lang_setting],
                    color=color_map[model_name],
                    linewidth=2,
                    markersize=6,
                    label=model_name,
                )

            # column titles only on top row
            if row_i == 0:
                ax.set_title(col_titles[col_i], fontsize=16)

            # x label only bottom row
            if row_i == len(lang_settings) - 1:
                ax.set_xlabel("Number of Shots")

            # y label only left column
            if col_i == 0:
                ax.set_ylabel(f"{METRIC.upper()} score")

            ax.set_xticks(x)
            ax.grid(True)
            ax.set_axisbelow(True)

    # ------------------------------------------------------------------
    # Shared legend
    # ------------------------------------------------------------------
    handles, labels = subfigs[0].axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(models),
        bbox_to_anchor=(0.5, -0.10),
        frameon=True,
        shadow=True,
    )

    fig.savefig("plots/main_fig.pdf", format="pdf", bbox_inches="tight")
            

def latex_table(df: pd.DataFrame) -> str:
    table = "\n\n"
    colspec = "ll" + "c" * (len(df.columns) - 2)

    header = " & & \\multicolumn{" + str(len(SHOTS)) + "}{c}{\\textbf{Pseudo Parallel Data}} & \\multicolumn{" + str(len(SHOTS)) + "}{c}{\\textbf{In-the-wild}} \\\\ \n"
    header += "\\cmidrule(lr){3-" + str(2 + len(SHOTS)) + "} \\cmidrule(lr){" + str(3 + len(SHOTS)) + "-" + str(2 + 2*len(SHOTS)) + "} \n"
    header += "Language & Model & $k=0$ & $k=50$ & $k=100$ & $k=250$ & $k=500$ & $k=0$ & $k=50$ & $k=100$ & $k=250$ & $k=500$ \\\\ \\hline\n"

    table += "\\begin{tabular}{" + colspec + "}\n"
    table += header

    prev_lang = None
    prev_resource = None
    for _, row in df.iterrows():

        if prev_lang != row['lang'] and prev_lang is not None:
            table += "\\cmidrule(lr){2-" + str(len(df.columns)) + "}\n"

        if row['lang'] in LANGS['medium'] and prev_resource is None:
            line = f"\\textbf{{Medium Resource }} " + " & " * (len(df.columns)-1) + " \\\\\n"
            line += "\\hline\n"
            prev_resource = "medium"
            table += line

        if row['lang'] in LANGS['low'] and prev_resource == "medium":
            line = "\\hline\n"
            line += f"\\textbf{{Low Resource }} " + " & " * (len(df.columns)-1) + " \\\\\n"
            line += "\\hline\n"
            prev_resource = "low"
            table += line

        if prev_lang is None or prev_lang != row['lang']:
            n_lang_rows = df[df['lang'] == row['lang']].shape[0]
            line = f"\\multirow{{{n_lang_rows}}}{{*}}{{{row['lang']}}} & {row['model_name']} "
        else:
            line = f" & {row['model_name']} "
        for shot in SHOTS:
            line += f"& {row[f'translation_{shot}']:.2f} "
        for shot in SHOTS:
            line += f"& {row[f'main_{shot}']:.2f} "
        line += "\\\\\n"
        table += line

        
        prev_lang = row['lang']
    table += "\\bottomrule\n"
    table += "\\end{tabular}\n"
    table += "\n\n"
    
    table = table.replace("nan", "-")
    print("\n\n")
    print(table)
    print("\n\n")

def main():
    rows = load_all_models(path=EXP2_DIR)

    pivot = create_df(rows)
    print(pivot)
    print(pivot.columns)

    latex_table(pivot)

    generate_figure(pivot)
if __name__ == "__main__":
    main()
