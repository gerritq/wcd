import os
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from utils import MODEL_DISPLAY_NAMES, LANG_ORDER, LANGS, load_metrics
import pandas as pd
import sys

# ----------------------------------------------------------------------
# Configs
# ----------------------------------------------------------------------

BASE_DIR = os.getenv("BASE_WCD")
EXP2_DIR = os.path.join(BASE_DIR, "data/exp2/eval")

METRIC="f1"
COUNT = defaultdict(list)
SHOTS = [0, 50, 100, 250, 500]


# ----------------------------------------------------------------------
# 0-shot 
# ----------------------------------------------------------------------

def get_resource(lang: str) -> str:
    for resource, langs in LANGS.items():
        if lang in langs:
            return resource
    return "unknown"

# rename model by rm lr

def sft_names(model_name: str) -> str:
        
    if "(ES)" in model_name:
        return "ES"
    if "(TOL)" in model_name:
        return "TOL"
    if "(FTL)" in model_name:
        return "FTL"
    return model_name

def rename_model(model_name: str) -> str:
    if "||" in model_name:
        base, lr = model_name.split("||")
        return f"{base.strip()}"
    return model_name

def load_zero(rows: list[dict],
              meta_file: Path,
              meta_1: dict,
              ) -> dict[str, dict]:

    # COLLECT
    model_name = MODEL_DISPLAY_NAMES[meta_1["model_name"]]
    if meta_1["model_type"] == "clf":
        model_name += f" (ES) || {meta_1['learning_rate']}"
    if meta_1["model_type"] == "slm" and meta_1['atl'] == True:
        model_name += f" (TOL) || {meta_1['learning_rate']}"
    if meta_1["model_type"] == "slm" and meta_1['atl'] == False:
        model_name += f" (FTL) || {meta_1['learning_rate']}"
    
    try:
        rows.append({"model_name": model_name,
                    "lang": meta_1["lang"],
                    "lang_setting": meta_1["lang_setting"],
                    "shots": 0,
                    "metric": meta_1["test_metrics_0_shot"][METRIC],
                    "source": meta_file.parent.name
                    })
    except:
        pass

    return rows


def create_zero_shot_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)

    zero_results_check = (
        df[df["shots"] == 0]
        .groupby(["lang_setting", "lang", "model_name"])
        .agg(
            metric_mean=("metric", "mean"),
            metric_count=("metric", "count"),
            sources=("source", list),
            metrics_lst=("metric", list),
        )
        .reset_index()
    )

    print("="*60)
    print("="*60)
    print("CHECK ZERO-SHOT RESULTS")    
    print(zero_results_check)
    print("="*60)
    print("="*60)

    # CREATE 0SHOT PIVOT
    zero_results = (
        df[df["shots"] == 0]
        .pivot_table(
            index=["lang_setting", "lang"],
            columns=["model_name"],
            values=["metric"],
            aggfunc="mean"
        )
    )

    zero_results.columns = zero_results.columns.droplevel(0)
    zero_results = zero_results.reset_index().rename(columns={df.index.name:'index'})

    zero_results["resource"] = zero_results["lang"].apply(get_resource)


    # group by lang setting and resource
    zero_results = (zero_results
                    .reset_index(drop=True).groupby(["lang_setting", "resource"])
                    .mean(numeric_only=True).reset_index()
                    )


    # SORT COLUMNS
    def model_priority(col: str) -> tuple:
        col_l = col.lower()
        if "ftl" in col_l:
            return (1, col)
        if "tol" in col_l:
            return (2, col)
        if "es" in col_l:
            return (3, col)
        if "mbert" in col_l:
            return (0, col)
        return (3, col)
    model_cols = sorted(
        [c for c in zero_results.columns if c not in ["lang_setting", "resource"]],
        key=model_priority
    )
    # get sft names
    cols = ["lang_setting", "resource"] + model_cols
    zero_results = zero_results[cols]

    # rename mbert column
    zero_results = zero_results.rename(columns={"mBert": "mBERT"})

    # rename llama
    rename_dict = {col: sft_names(col) for col in model_cols}
    zero_results = zero_results.rename(columns=rename_dict)

    # SORT ROWS
    zero_results['lang_setting'] = pd.Categorical(zero_results['lang_setting'], categories=['translation', 'main'], ordered=True)
    zero_results['resource'] = pd.Categorical(zero_results['resource'], categories=['medium', 'low'], ordered=True)

    zero_results = zero_results.sort_values(['lang_setting', 'resource']).reset_index(drop=True)

    # reset index
    zero_results = zero_results.reset_index(drop=True)

    # rename values
    zero_results['lang_setting'] = zero_results['lang_setting'].replace({
        'translation': 'Parallel',
        'main': 'Non-Parallel'
    })
    zero_results['resource'] = zero_results['resource'].replace({
        'medium': 'Medium',
        'low': 'Low'
    })
    return zero_results

def zero_shot_latex_table(df: pd.DataFrame) -> str:
    table = "\n\n"
    colspec = "ll" + "c" * (len(df.columns) - 2)

    header = " & & & \\multicolumn{" + str(len(df.columns) - 3) + "}{c}{\\textbf{SFT Variant}} \\\\ \n"
    header += "\\cmidrule(lr){4-" + str(len(df.columns)) + "} \n"
    header += "\\textbf{{Data}} & \\textbf{{Resource}} "

    for model in df.columns[2:]:
        header += f"& \\textbf{{{model}}}"
    header += "\\\\ \\toprule\n"

    table += "\\begin{tabular}{" + colspec + "}\n"
    table += header

    prev_lang_setting = None
    for _, row in df.iterrows():

        if prev_lang_setting != row['lang_setting'] and prev_lang_setting is not None:
            table += "\\cmidrule(lr){2-" + str(len(df.columns)) + "}\n"

        if prev_lang_setting is None or prev_lang_setting != row['lang_setting']:
            n_setting_rows = df[df['lang_setting'] == row['lang_setting']].shape[0]
            line = f"\\multirow{{{n_setting_rows}}}{{*}}{{{row['lang_setting']}}} & {row['resource']} "
        else:
            line = f" & {row['resource']} "
        for model in df.columns[2:]:
            if "mbert" in model.lower():
                line += f"& {row[model]*100:.2f} "
            else:
                diff = row[model] - row['mBERT']
                line += f"& {row[model]*100:.2f} (\\posneg{{{diff*100:.2f}}}) "

        line += "\\\\\n"
        table += line

        prev_lang_setting = row['lang_setting']

    table += "\\bottomrule\n"
    table += "\\end{tabular}\n"
    table += "\n\n"
    
    print("\n\n")
    print("ZERO-SHOT TABLE")
    print(table)
    print("\n\n")

# ----------------------------------------------------------------------
# Few shot
# ----------------------------------------------------------------------

def load_few(rows: list[dict],
             all_meta_files: list[Path],
              ) -> dict[str, dict]:

    # COLLECT
    for meta_file in all_meta_files:
        meta_1 = load_metrics(meta_file)    
        model_name = MODEL_DISPLAY_NAMES[meta_1["model_name"]]
        if meta_1["model_type"] == "clf":
            model_name += f" (ES) || {meta_1['learning_rate']}"
        if meta_1["model_type"] == "slm" and meta_1['atl'] == True:
            model_name += f" (TOL) || {meta_1['learning_rate']}"
        if meta_1["model_type"] == "slm" and meta_1['atl'] == False:
            model_name += f" (FTL) || {meta_1['learning_rate']}"
            
        rows.append({"model_name": model_name,
                    "lang": meta_1["lang"],
                    "lang_setting": meta_1["lang_setting"],
                    "shots": meta_1["training_size"],
                    "metric": meta_1["test_metrics"][-1]['metrics'][METRIC],
                    "source": meta_file.parent.name
                    })
        
    return rows

def create_all_results_df(rows: list[dict]) -> pd.DataFrame:
 
    df = pd.DataFrame(rows)


    all_results_check = (
        df
        .groupby(["lang_setting", "lang", "model_name", "shots"])
        .agg(
            metric_mean=("metric", "mean"),
            metric_count=("metric", "count"),
            sources=("source", list),
            metrics_lst=("metric", list),
        )
        .reset_index()
    )

    print("="*60)
    print("="*60)
    print("ALL RESULTS CHECK")    
    print(all_results_check)
    print("="*60)
    print("="*60)

    all_results_check.to_excel("checks/all_results_check.xlsx")

    # CRAETE ALL RESULTS PIVOT
    all_results = (df.pivot_table(
            index=["lang", "model_name"],
            columns=["lang_setting", "shots"],
            values="metric",
            aggfunc="mean"
        )
        .sort_index(axis=1, level=[0, 1])
    )

    
    all_results.columns = [
        c if isinstance(c, str) else f"{c[0]}_{c[1]}"
        for c in all_results.columns
    ]
    all_results = all_results.reset_index() 
    id_cols = ["lang", "model_name"]
    trans_cols = [c for c in all_results.columns if c.startswith("translation_")]
    main_cols  = [c for c in all_results.columns if c.startswith("main_")]

    all_results = all_results[id_cols + trans_cols + main_cols]

    # sort LANG_ORDER
    all_results['lang'] = pd.Categorical(all_results['lang'], categories=LANG_ORDER, ordered=True)
    all_results = all_results.sort_values(['lang', 'model_name']).reset_index(drop=True)
    return all_results


def zero_shot_helper_df(rows: list[dict], lang_setting: str = "main") -> pd.DataFrame:
    df = pd.DataFrame(rows)

    # zero shot helper tables
    df_zero = df[(df["shots"] == 0) & (df["lang_setting"] == lang_setting) & (df["model_name"].str.contains("llama", case=False))]
    zero_results = (    
        df_zero
        .pivot_table(
            index=["model_name", "lang"],
            values="metric",
            aggfunc="mean"
        )
    )
    zero_results = zero_results.reset_index().rename(columns={df.index.name:'index',
                                                              "metric": "shot_0"})
    
    # rename model names
    zero_results["model_name"] = zero_results["model_name"].apply(rename_model)

    # sort
    zero_results['lang'] = pd.Categorical(zero_results['lang'], categories=LANG_ORDER, ordered=True)
    zero_results = zero_results.sort_values(['model_name', 'lang']).reset_index(drop=True)

    return zero_results

def create_lr_few_shot_df(rows: list[dict], lang_setting: str = "main") -> pd.DataFrame:
    
    df = pd.DataFrame(rows)

    # get zero shot helper
    zero_results = zero_shot_helper_df(rows, lang_setting=lang_setting)
    
    # FILTER TO FEW-SHOT ONLY
    df_shots = df[
        (df["lang_setting"] == "main") & (df["model_name"].str.contains("llama", case=False) & (df["shots"] > 0))
    ]
    df_shots = df_shots.drop(columns=["lang_setting"])

    few_shot_results_check = (
        df_shots
        .groupby(["model_name", "lang", "shots"])
        .agg(
            metric_mean=("metric", "mean"),
            metric_count=("metric", "count"),
            sources=("source", list),
            metrics_lst=("metric", list),
        )
        .reset_index()
    )

    print("="*60)
    print("="*60)
    print("FEW-SHOT RESULTS CHECK")    
    print(few_shot_results_check)
    print("="*60)
    print("="*60)

    # CREATE FEW-SHOT PIVOT
    few_shot_results = (df_shots
        .pivot_table(
            index=["model_name","lang"],
            columns=["shots"],
            values="metric",
            aggfunc="mean"
        )
    )

    # rename columns
    few_shot_results.columns = [f"shot_{c}" for c in few_shot_results.columns]
    few_shot_results = few_shot_results.reset_index().rename(columns={df.index.name:'index'})

    

    # add resource level
    few_shot_results["resource"] = few_shot_results["lang"].apply(get_resource)

    # split moedl name to base and lr
    def split_model_lr(model_name: str) -> tuple:
        base, lr = model_name.split("||")
        return base.strip(), lr.strip()    
    few_shot_results[["model_name", "learning_rate"]] = few_shot_results["model_name"].apply(split_model_lr).tolist()

    # make learning rate float
    few_shot_results["learning_rate"] = few_shot_results["learning_rate"].astype(float)

    # add zero shot (needs clean model neame wo lr)
    few_shot_results = few_shot_results.merge(zero_results, on=["model_name", "lang"], how="left")

    # group by model and resource
    few_shot_results = (few_shot_results
                    .reset_index(drop=True).groupby(["resource", "model_name", "learning_rate"])
                    .mean(numeric_only=True).reset_index()
                    )
    
    # rename to keep only sft strategy names
    few_shot_results["model_name"] = few_shot_results["model_name"].apply(sft_names)
    
        # sort
    few_shot_results['resource'] = pd.Categorical(few_shot_results['resource'], categories=['medium', 'low'], ordered=True)
    few_shot_results['model_name'] = pd.Categorical(few_shot_results['model_name'], categories=['FTL', 'TOL', 'ES'], ordered=True)
    few_shot_results = (
        few_shot_results
            .sort_values(
                ['resource', 'model_name', 'learning_rate'],
                ascending=[True, True, False]
            )
            .reset_index(drop=True)
    )

    # generate differences
    for shot in SHOTS[1:]:
        few_shot_results[f'diff_shot_{shot}'] = few_shot_results[f'shot_{shot}'] - few_shot_results['shot_0']

    # generate avg diff
    diff_cols = [f'diff_shot_{shot}' for shot in SHOTS[1:]]
    few_shot_results['avg_diff'] = few_shot_results[diff_cols].mean(axis=1)


    print("="*60)
    print("FEW-SHOT RESULTS PIVOT")
    print(few_shot_results)
    print("="*60)

    return few_shot_results


def all_learning_rate_latex_table(df: pd.DataFrame, lang_setting: str = "main") -> str:
    
    shot_length = len(SHOTS) - 1 + 1 # - 1 to excl zero and + 1 for avg column
    table = "\n\n"
    colspec = "ll" + "c" * shot_length

    header = " & & \\multicolumn{" + str(shot_length-1) + "}{c}{\\textbf{" + ("Parallel Data" if lang_setting == "translation" else "Non-parallel Data") + "}} \\\\ \n"
    header += "\\cmidrule(lr){3-" + str(2 + shot_length-1) + "} \n"
    header += "\\textbf{SFT Variant}   & \\textbf{\\makecell{Learning \\\\ Rate}} & $k=50$ & $k=100$ & $k=250$ & $k=500$ & \\textbf{Avg} \\\\ \\toprule\n"

    table += "\\begin{tabular}{" + colspec + "}\n"
    table += header

    prev_resource = None
    prev_models = None
    for _, row in df.iterrows():

        # Panels
        if prev_resource is None:
            table += f"\\textbf{{Medium Resource }} " + " & " * (shot_length + 1) + " \\\\\n"
            table += "\\midrule\n"
        if prev_resource != row['resource'] and prev_resource is not None:
            table += "\\midrule\n"
            table += f"\\textbf{{Low Resource }} " + " & " * (shot_length + 1) + " \\\\\n"
            table += "\\midrule\n"

        # Lines
        if prev_models is None or prev_models != row['model_name']:
            n_model_rows = df[(df['model_name'] == row['model_name']) & (df['resource'] == row['resource'])].shape[0]
            line = f"\\multirow{{{n_model_rows}}}{{*}}{{{row['model_name']}}} & {row['learning_rate']:.0e} "
            if prev_models and row['resource'] == prev_resource:
                table += "\\cmidrule(lr){2-" + str(2 + shot_length) + "}\n"
        else:
            line = f" & {row['learning_rate']:.0e} "

        # add shots with diff
        for shot in SHOTS[1:]:
            diff = row[f'diff_shot_{shot}']
            line += f"& {row[f'shot_{shot}']*100:.2f} (\\posneg{{{diff*100:.2f}}}) "
            
        # add avg
        if row['avg_diff'] == df[(df['model_name'] == row['model_name']) & (df['resource'] == row['resource'])]['avg_diff'].max():
            line += f"& \\textbf{{{row['avg_diff']*100:.2f}}} \\\\ \n"
        else:
            line += f"& {row['avg_diff']*100:.2f} \\\\ \n"
       
        table += line   

        
        prev_resource = row['resource']
        prev_models = row['model_name']

    table += "\\bottomrule\n"
    table += "\\end{tabular}\n"
    table += "\n\n"
    
    table = table.replace("nan", "-")
    print("\n\n")
    print(table)
    print("\n\n")
# ----------------------------------------------------------------------
# Loader
# ----------------------------------------------------------------------

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
                rows = load_zero(rows=rows, meta_file=meta_files[0], meta_1=meta_1)

            if len(meta_files) == 4:
                rows = load_few(rows=rows, all_meta_files=meta_files)


    return rows

# ----------------------------------------------------------------------
# Main figures and tables
# ----------------------------------------------------------------------



def generate_main_figure(df: pd.DataFrame):

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
            

def all_results_latex_table(df: pd.DataFrame) -> str:
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

    # zero shot
    # zero_shot_df = create_zero_shot_df(rows)
    # zero_shot_latex_table(zero_shot_df)

    # few-shot lr table
    # lang_setting = "main"
    # few_shot_lr_df = create_lr_few_shot_df(rows, lang_setting=lang_setting)
    # all_learning_rate_latex_table(few_shot_lr_df, lang_setting=lang_setting)

    
    # # all results
    pivot = create_all_results_df(rows)
    all_results_latex_table(pivot)  
    generate_main_figure(pivot)
    
if __name__ == "__main__":
    main()
