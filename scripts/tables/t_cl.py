
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from utils import MODEL_DISPLAY_NAMES, LANG_ORDER, LANGS, load_metrics
import pandas as pd
import sys
import numpy as np

# ----------------------------------------------------------------------
# Configs
# ----------------------------------------------------------------------

BASE_DIR = os.getenv("BASE_WCD")
EXP2_DIR = os.path.join(BASE_DIR, "data/exp2/eval")

METRIC="f1"  # accuracy | f1_macro | f1_weighted | exact_match
COUNT = defaultdict(list)
SHOTS = [0, 50, 100, 250, 500]

LANG_SETTING = "main"  # main | translation
ITEM = "LR"  # zero | LR | FEW

# ----------------------------------------------------------------------
# COLLECTORS AND HELPERS
# ----------------------------------------------------------------------

def get_resource(lang: str) -> str:
    for resource, langs in LANGS.items():
        if lang in langs:
            return resource
    return "unknown"

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
        model_name += f" (ES)"
    if meta_1["model_type"] == "slm" and meta_1['atl'] == True:
        model_name += f" (TOL)"
    if meta_1["model_type"] == "slm" and meta_1['atl'] == False:
        model_name += f" (FTL)"
    
    try:
        rows.append({"model_name": model_name,
                    "lang": meta_1["lang"],
                    "lang_setting": meta_1["lang_setting"],
                    "shots": 0,
                    "metric": meta_1["test_metrics_0_shot"][METRIC],
                    "source": meta_file.parent.name,
                    "seed": meta_1["seed"],
                    "learning_rate": float(meta_1['learning_rate'])
                    })
    except:
        pass

    return rows

def load_few(rows: list[dict],
             all_meta_files: list[Path],
              ) -> dict[str, dict]:

    # COLLECT
    for meta_file in all_meta_files:
        meta_1 = load_metrics(meta_file)    
        model_name = MODEL_DISPLAY_NAMES[meta_1["model_name"]]
        if meta_1["model_type"] == "clf":
            model_name += f" (ES)"
        if meta_1["model_type"] == "slm" and meta_1['atl'] == True:
            model_name += f" (TOL)"
        if meta_1["model_type"] == "slm" and meta_1['atl'] == False:
            model_name += f" (FTL)"
            
        rows.append({"model_name": model_name,
                    "lang": meta_1["lang"],
                    "lang_setting": meta_1["lang_setting"],
                    "shots": meta_1["training_size"],
                    "metric": meta_1["test_metrics"][-1]['metrics'][METRIC],
                    "source": meta_file.parent.name,
                    "seed": meta_1["seed"],
                    "learning_rate": float(meta_1['learning_rate'])
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
                rows = load_zero(rows=rows, meta_file=meta_files[0], meta_1=meta_1)

            if len(meta_files) == 4:
                rows = load_few(rows=rows, all_meta_files=meta_files)


    return rows


# ----------------------------------------------------------------------
# 0-shot 
# ----------------------------------------------------------------------

def zero_shot_helper_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)

    # zero shot helper tables
    df_zero = df[(df["shots"] == 0)]
    zero_results = (    
        df_zero
        .pivot_table(
            index=["lang_setting", "model_name", "lang"],
            values="metric",
            aggfunc=["mean", "std"]
        )
    )

    # flatten column names
    zero_results.columns = [
        f"{stat}_{col}" for stat, col in zero_results.columns
    ]
    
    # rename metrics to 0_shot
    zero_results = zero_results.reset_index().rename(columns={"mean_metric":'shot_0_mean', "std_metric":'shot_0_std'})
    
    # rename model names
    zero_results["model_name"] = zero_results["model_name"].apply(rename_model)

    # add resource level
    zero_results["resource"] = zero_results["lang"].apply(get_resource)


    # sort
    zero_results['lang'] = pd.Categorical(zero_results['lang'], categories=LANG_ORDER, ordered=True)
    zero_results = zero_results.sort_values(['model_name', 'lang']).reset_index(drop=True)

    return zero_results


def create_zero_shot_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)

    zero_results_check = (
        df[df["shots"] == 0]
        .groupby(["lang_setting", "model_name", "lang"])
        .agg(
            metric_mean=("metric", "mean"),
            metric_std=("metric", "std"),
            metric_count=("metric", "count"),
            sources=("source", list),
            metrics_lst=("metric", list),
            seeds_lst=("seed", list),
        )
        .sort_values(['lang_setting', 'lang', 'model_name']).reset_index()
    )

    print("="*60)
    print("="*60)
    print("CHECK ZERO-SHOT RESULTS")    
    print(zero_results_check)
    print("="*60)
    print("="*60)

    # save
    zero_results_check.to_excel(f"checks/zero_check_{METRIC}.xlsx")

    # CREATE 0SHOT PIVOT
    zero_results = (
        df[df["shots"] == 0]
        .pivot_table(
            index=["lang_setting", "lang"],
            columns=["model_name"],
            values=["metric"],
            aggfunc=["mean", "std"]
        )
    )

    # flatten column names and reset index
    zero_results.columns = [
        f"{model}_{stat}"
        for stat, _, model in zero_results.columns
    ]
    zero_results = zero_results.reset_index().rename(columns={df.index.name:'index'})

    print(zero_results)
    
    # add resource level
    zero_results["resource"] = zero_results["lang"].apply(get_resource)

    
    # group by lang setting and resource
    zero_results_average = (zero_results
                            .reset_index(drop=True).groupby(["lang_setting", "resource"])
                            .mean(numeric_only=True).reset_index()
                            )

    # SORT ROWS
    zero_results_average['lang_setting'] = pd.Categorical(zero_results_average['lang_setting'], categories=['translation', 'main'], ordered=True)
    zero_results_average['resource'] = pd.Categorical(zero_results_average['resource'], categories=['medium', 'low'], ordered=True)
    zero_results_average = zero_results_average.sort_values(['lang_setting', 'resource']).reset_index(drop=True)

    # reset index
    zero_results_average = zero_results_average.reset_index(drop=True)

    # rename values
    zero_results_average['lang_setting'] = zero_results_average['lang_setting'].replace({
        'translation': 'Parallel',
        'main': 'Non-Parallel'
    })
    zero_results_average['resource'] = zero_results_average['resource'].replace({
        'medium': 'Medium',
        'low': 'Low'
    })

    # multiply by 100
    for col in zero_results_average.columns:
        if col != 'lang_setting' and col != 'resource':
            zero_results_average[col] = zero_results_average[col] * 100.0

    print("="*60)
    print("ZERO-SHOT RESULTS PIVOT")
    print(zero_results_average[[col for col in zero_results_average.columns if not col.endswith('std')]].head())
    print("="*60)

    return zero_results_average

def zero_shot_plot(
    df,
    panel_col="lang_setting",
    x_col="resource",
    resource_order=("Medium", "Low"),
    panel_order=("Parallel", "Non-Parallel"),
    model_order=(
        ("mBert", "mBert_mean", "mBert_std"),
        ("Llama3-8B (FTL)", "Llama3-8B (FTL)_mean", "Llama3-8B (FTL)_std"),
        ("Llama3-8B (TOL)", "Llama3-8B (TOL)_mean", "Llama3-8B (TOL)_std"),
        ("Llama3-8B (ES)", "Llama3-8B (ES)_mean", "Llama3-8B (ES)_std"),
    ),
    capsize=5,
    figsize=(8.2, 3.6),
    alpha=0.8,
):
    from matplotlib.ticker import MultipleLocator
    outpath = f"plots/zero_shot_{METRIC}.pdf"

    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 14,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
    })

    fig, axes = plt.subplots(1, len(panel_order), figsize=figsize, sharey=True)
    if len(panel_order) == 1:
        axes = [axes]

    # -----------------------------
    # Spacing controls (main change)
    # -----------------------------
    group_gap = 0.6   # smaller -> less space between resource groups (Low/Medium)
    intra_gap = 1.35  # larger  -> more space between bars within a group

    x = np.arange(len(resource_order)) * group_gap
    n_models = len(model_order)

    bar_width = 0.08
    offsets = (
        (np.arange(n_models) - (n_models - 1) / 2)
        * bar_width
        * intra_gap
    )
    # -----------------------------

    for ax, panel in zip(axes, panel_order):
        sub = df[df[panel_col] == panel].set_index(x_col)

        # store positions + values for line plotting
        values_by_resource = {r: [] for r in resource_order}
        xpos_by_resource = {r: [] for r in resource_order}

        for i, (label, mean_col, std_col) in enumerate(model_order):
            means = [float(sub.loc[r, mean_col]) for r in resource_order]
            stds  = [float(sub.loc[r, std_col])  for r in resource_order]

            x_pos = x + offsets[i]

            # Bars
            ax.bar(
                x_pos,
                means,
                width=bar_width,
                yerr=stds,
                capsize=capsize,
                alpha=alpha,
                label=label,
                zorder=3,
            )

            # collect points per resource (for within-group lines)
            for xi, r, m in zip(x_pos, resource_order, means):
                xpos_by_resource[r].append(xi)
                values_by_resource[r].append(m)

        # draw lines within each resource group
        for r in resource_order:
            ax.plot(
                xpos_by_resource[r],
                values_by_resource[r],
                linestyle="--",
                marker="o",
                linewidth=1,
                markersize=4,
                color="gr" \
                "ay",
                alpha=0.6,
                zorder=4,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(resource_order)
        ax.tick_params(axis="x", pad=2)
        ax.set_title(panel)

        ax.grid(
            axis="y",
            linestyle="--",
            linewidth=0.6,
            alpha=0.4,
            zorder=0,
        )

        ax.yaxis.set_minor_locator(MultipleLocator(10))
        ax.grid(
            axis="y",
            which="minor",
            linestyle=":",
            linewidth=0.5,
            alpha=0.3,
            zorder=0,
        )

    axes[0].set_ylabel(f"{METRIC.replace('_', ' ').title()}")
    fig.supxlabel("Resource level", y=0.15)

    # common legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(model_order),
        frameon=True,
        bbox_to_anchor=(0.5, 0.01),
    )

    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(outpath, bbox_inches="tight")



        
# ----------------------------------------------------------------------
# Few-shot
# ----------------------------------------------------------------------

# def create_few_shot_df(rows: list[dict]) -> pd.DataFrame:
    
    
#     df = pd.DataFrame(rows)

#     # ------------------------
#     # CHECK TABLE
#     # ------------------------
#     few_results_check = (
#         df[(df["shots"] > 0) & (df['learning_rate'] != 0.000005)]
#         .groupby(["lang_setting", "model_name", "lang", "shots", "learning_rate"])
#         .agg(
#             metric_mean=("metric", "mean"),
#             metric_std=("metric", "std"),
#             metric_count=("metric", "count"),
#             sources=("source", list),
#             metrics_lst=("metric", list),
#             seeds_lst=("seed", list),
#         )
#         .sort_values(['lang_setting', 'lang', 'model_name', "shots", "learning_rate"]).reset_index()
#     )

#     print("="*60)
#     print("="*60)
#     print("CHECK FEW-SHOT RESULTS")    
#     print(few_results_check)
#     print("="*60)
#     print("="*60)

#     # save
#     few_results_check.to_excel("checks/few_shot_results_check.xlsx")

#     # ------------------------
#     # Lang-setting- Resource AVERAGE PIVOT
#     # ------------------------
    
#     # get zero shot helper
#     zero_results = zero_shot_helper_df(rows)

#     # group lang setting and resource level
#     zero_results = zero_results.groupby(["lang_setting", "resource", "model_name"]).mean(numeric_only=True).reset_index()

#     print("="*60)
#     print("="*60)
#     print("CHECK ZERO-SHOT FOR FEW-SHOT MERGE")    
#     print(zero_results)
#     print("="*60)
#     print("="*60)
    
#     df_shots = df[
#         ((df["shots"] > 0)) & (df['learning_rate'] != 0.000005)
#         ]
    
#     # add resource level
#     df_shots["resource"] = df_shots["lang"].apply(get_resource)

#     df_shots = (df_shots
#         .pivot_table(
#             index=["lang_setting", "model_name", "resource", "learning_rate"],
#             columns=["shots"],
#             values="metric",
#             aggfunc=["mean", "std"] #"count"
#         ).reset_index()
#     )

#     # flatten column names
#     df_shots.columns = [f"{c[0]}_{c[1]}" if c[1] else c[0] for c in df_shots.columns]

#     # add zero-shot values
#     df_shots = df_shots.merge(zero_results,
#                             on=["model_name", "resource", "lang_setting"],
#                             how="left")

#     # ------------------------
#     # LR Table
#     # ------------------------

#     df_learning_rate = df_shots[df_shots["model_name"].str.contains("Llama", case=False)]
#     # Genete differences between 0-shot and few-shot
#     for shot in SHOTS[1:]:
#         df_learning_rate[f'diff_shot_{shot}'] = df_learning_rate[f'mean_{shot}'] - df_learning_rate['shot_0_mean']
    
#     # Add average diff
#     diff_cols = [f'diff_shot_{shot}' for shot in SHOTS[1:]]
#     df_learning_rate['avg_diff'] = df_learning_rate[diff_cols].mean(axis=1)
#     # srop mean,std
#     df_learning_rate = df_learning_rate.drop(columns=[col for col in df_learning_rate.columns if col.startswith("std_")])
    
#     # multiply all scores by 100
#     for col in df_learning_rate.columns:
#         if col not in ["lang_setting", "model_name", "resource", "learning_rate"]:
#             df_learning_rate[col] = df_learning_rate[col] * 100.0

#     # Sory
#     df_learning_rate['resource'] = pd.Categorical(df_learning_rate['resource'], categories=['medium', 'low'], ordered=True)
#     df_learning_rate['model_name'] = pd.Categorical(df_learning_rate['model_name'].apply(sft_names), categories=['FTL', 'TOL', 'ES'], ordered=True)
#     df_learning_rate = df_learning_rate.sort_values(['lang_setting', 'resource', 'model_name', 'learning_rate'],
#                                                     ascending=[True, True, True, False]).reset_index(drop=True)
#     print("="*60)
#     print("FEW-SHOT LEARNING RATE DF")
#     print(df_learning_rate)
#     print("="*60)

#     # return df_learning_rate
#     # ------------------------
#     # PLOT Table
#     # ------------------------
#     for shot in SHOTS[1:]:
#         df_shots[f'diff_shot_{shot}'] = df_shots[f'mean_{shot}'] - df_shots['shot_0_mean']
#     df_shots['avg_diff'] = df_shots[[f'diff_shot_{shot}' for shot in SHOTS[1:]]].mean(axis=1)

#     # drop diff and std
#     df_shots = df_shots.drop(columns=[col for col in df_shots.columns if col.startswith("std_") or col.startswith("diff_shot_")])

#     # group and keep where avg_diff is the highest per model/resource/lang_setting
#     df_shots = (
#             df_shots
#             .groupby(["lang_setting", "resource", "model_name"])
#             .apply(lambda x: x.loc[x["avg_diff"].idxmax()])
#             .reset_index(drop=True)
#             .drop(columns=["avg_diff", "learning_rate"]
#         )
#         )
    
#     print("="*60)
#     print("FEW-SHOT PLOT DF")
#     print(df_shots)
#     print("="*60)

#     # ------------------------
#     # FULL Table
#     # ------------------------

#     full_table = df[(df["shots"] > 0) & (df['learning_rate'] != 0.000005)].copy()
    
#     # add resource
#     full_table["resource"] = full_table["lang"].apply(get_resource)

#     full_table = (full_table
#         .pivot_table(
#             index=["lang_setting", "resource", "lang", "model_name", "learning_rate"],
#             columns=["shots"],
#             values="metric",
#             aggfunc=["mean", "std"] #"count"
#         ).reset_index()
#     )

#     full_table.columns = [f"{c[0]}_{c[1]}" if c[1] else c[0] for c in full_table.columns]

#     # merge zero-shot mean and std
#     full_table = full_table.merge(zero_results, on=["model_name", "resource", "lang_setting"], how="left")
#     # rename and reaorder
#     full_table = full_table.rename(columns={"shot_0_mean": "mean_0", "shot_0_std": "std_0"})
#     column_order = [f"mean_{shot}" for shot in SHOTS] + [f"std_{shot}" for shot in SHOTS]
#     full_table = full_table[["lang_setting", "resource", "lang", "model_name", "learning_rate"] + column_order]
    

#     # pivot to have main and trnslation side by side
#     mean_cols = [c for c in full_table.columns if c.startswith("mean_")]
#     std_cols  = [c for c in full_table.columns if c.startswith("std_")]
#     value_cols = mean_cols + std_cols

#     pivoted = (
#         full_table
#         .pivot_table(
#             index=["resource", "lang", "model_name", "learning_rate"],
#             columns="lang_setting",
#             values=value_cols,
#             aggfunc="mean",   # safe if duplicates exist
#         )
#     )

#     pivoted.columns = [
#     f"{metric}_{setting}"
#     for metric, setting in pivoted.columns
#     ]
#     pivoted = pivoted.reset_index()

#     # order columns
#     ordered_cols = ["resource", "lang", "model_name", "learning_rate"]
#     for setting in ["translation", "main"]:
#         for shot in SHOTS:
#             ordered_cols.append(f"mean_{shot}_{setting}")
#         for shot in SHOTS:
#             ordered_cols.append(f"std_{shot}_{setting}")

#     pivoted = pivoted[ordered_cols]

#     # order roows
#     pivoted['resource'] = pd.Categorical(pivoted['resource'], categories=['medium', 'low'], ordered=True)
#     pivoted['model_name'] = pd.Categorical(pivoted['model_name'].apply(sft_names), categories=['FTL', 'TOL', 'ES', "mBert"], ordered=True)
#     pivoted = pivoted.sort_values(['resource', 'lang', 'model_name']).reset_index(drop=True)

#     # multiply all mean and std by 100
#     for col in pivoted.columns:
#         if col.startswith("mean_") or col.startswith("std_"):
#             pivoted[col] = pivoted[col] * 100.0

#     return df_shots, df_learning_rate, pivoted

def few_shot_main_table_df(rows: list[dict], lang_setting: str = "main") -> pd.DataFrame:
    df = pd.DataFrame(rows)


    # ======================================================
    # CHECK TABLE
    # ======================================================
    few_results_check = (
        df[(df['learning_rate'] != 5e-6)]
        .groupby(["lang_setting", "model_name", "lang", "shots", "learning_rate"])
        .agg(
            metric_mean=("metric", "mean"),
            metric_std=("metric", "std"),
            metric_count=("metric", "count"),
            sources=("source", list),
            metrics_lst=("metric", list),
            seeds_lst=("seed", list),
        )
        .sort_values(['lang_setting', 'lang', 'model_name', "shots", "learning_rate"]).reset_index()
    )

    # print("="*60)
    # print("="*60)
    # print("CHECK FEW-SHOT RESULTS")    
    # print(few_results_check)
    # print("="*60)
    # print("="*60)


    # ======================================================
    # Helper df zero
    # ======================================================

    df_zero_helper = df.copy()
    df_zero_helper = df_zero_helper[
        (df_zero_helper["shots"] == 0)
        & (df_zero_helper['learning_rate'] != 5e-6)
        & (df_zero_helper['lang_setting'] == lang_setting)
    ]

    df_zero_helper = (df_zero_helper.
                      groupby(["model_name", "lang"],
                              ).agg(
                        zero_mean=("metric", "mean"),
                      ).reset_index()
    )

    # clean column,s
    df_zero_helper.columns = ["model_name", "lang", "zero_shot_mean"]

    print("ZERO HELPER")
    print(df_zero_helper)


    # ======================================================
    # Pick best lr per model and resource level
    # ======================================================
    df_zero = df.copy()

    df_zero = (
        df_zero[
            (df_zero["shots"] == 0)
            & (df_zero["learning_rate"] != 5e-6)
            & (df_zero["lang_setting"] == lang_setting)
        ]
        .groupby(["model_name", "lang", "shots"])["metric"]
        .agg(["mean", "std"])
        .reset_index()
    )



    print("ZERO")
    print(df_zero)

    # ======================================================
    # Pick best lr per model and resource level
    # ======================================================
    df_lr = df.copy()
    df_lr = df_lr[(df_lr['lang_setting'] == lang_setting) & (df_lr['learning_rate'] != 5e-6) & (df_lr['shots'] > 0)]
    df_lr['resource'] = df_lr['lang'].apply(get_resource)
    df_lr = (df_lr
                .groupby(["model_name", "resource", "learning_rate"])
                .agg(metric_mean=("metric", "mean"),
                )
    )

    # flatten columns
    df_lr = df_lr.reset_index()
    df_lr.columns = ["model_name", "resource", "learning_rate", "lr_avg"]

    # pick highest lr per model name  and reousrce
    df_lr = df_lr.loc[
        df_lr.groupby(["model_name", "resource"])["lr_avg"].idxmax()
    ].reset_index(drop=True)

    print("BEST LR")
    print(df_lr)
    # ======================================================
    # Gen main table using the best lr per model-resource
    # ======================================================
    df_main = df.copy()
    df_main['resource'] = df['lang'].apply(get_resource)
    df_main = df_main[(df_main['lang_setting'] == lang_setting) & (df_main['learning_rate'] != 5e-6) & (df_main['shots'] > 0)]
    df_main = df_main.dropna(subset=['lang_setting'])

    
    df_main = (df_main
               .groupby(["model_name", "resource", "lang", "shots", "learning_rate", "seed"])
               .agg(metric_mean=("metric", "mean"),
                    metric_std=("metric", "std")
                ))

    # flatten columns
    df_main = df_main.reset_index()
    df_main.columns = ["model_name", "resource", "lang", "shots", "learning_rate", "seed", "metric_mean", "metric_std"]

    # select only rows with best lr per model and resource
    df_main = df_main.merge(df_lr, on=["model_name", "resource", "learning_rate"], how="inner")

    # merge zero shot values
    df_main = df_main.drop(columns=['lr_avg', "learning_rate", "resource"])  
    df_main= df_main.merge(
                                df_zero_helper, on=["model_name", "lang"], how="left")
    
    
    df_main['diff'] = df_main['metric_mean'] - df_main['zero_shot_mean']

    df_main = (df_main
               .groupby(["model_name", "lang", "shots"])['diff']
               .agg(['mean', 'std'])
               .reset_index())

    df_main = pd.concat([df_main, df_zero], ignore_index=True, axis=0)

    # pivot
    df_main = (df_main 
                .pivot_table(
                    index=["model_name", "shots"],  
                    columns=["lang"],
                    values=["mean", "std"],
                    aggfunc=["mean"]
                ).reset_index()
    )

    # clean columns
    df_main.columns = [
        "_".join([str(x) for x in c if x])
        for c in df_main.columns
    ]

    df_main.columns = [col.replace("mean_mean", "mean").replace("mean_std", "std") for col in df_main.columns]

    # all times 100
    for col in df_main.columns:
        if col not in ["model_name", "shots"]:
            df_main[col] = df_main[col] * 100.0


    # test shot means
    # for resource, lang in LANGS.items():
    #     if resource == "high":
    #         continue

    #     lang_cols = [f"mean_{l}" for l in lang]
    #     df_subset = df_main[["model_name", "shots"] + lang_cols]
        
    #     # row mean over all mean_langs
    #     df_subset = df_subset.assign(lang_mean=df_subset[lang_cols].mean(axis=1))
    #     print(f"RESOURCE: {resource}")
    #     print(df_subset)
        
    # add resource means
    for resource, lang in LANGS.items():
        if resource == "high":
            continue

        lang_cols = [f"mean_{l}" for l in lang]
        df_main[f"{resource}_mean"] = df_main[lang_cols].mean(axis=1)

    
    print("FINAL MAIN")
    print(df_main)

    df_main['model_name'] = pd.Categorical(df_main['model_name'].apply(sft_names), categories=['FTL', 'TOL', 'ES', "mBert"], ordered=True)
    df_main = df_main.sort_values(['model_name', 'shots']).reset_index(drop=True)


    return df_main

    sys.exit(0)


    #=====================================
    #=====================================
    #=====================================
    #=====================================
    #=====================================

    # now pivot to have langs as columns
    df_main = df_main.drop(columns=['resource', 'learning_rate', 'lr_avg'])
    df_main = (df_main
               .pivot_table(
                index=["model_name", "shots"],
                columns=["lang"],
                values="metric_mean",
                aggfunc=["mean"] #"count"
        ).reset_index())
    
    # clean columns
    df_main.columns = [f"{c[0]}_{c[1]}" if c[1] else c[0] for c in df_main.columns]

    # # add zero shot values as rows not columns
    df_main = pd.concat([df_main, df_zero], ignore_index=True, axis=0)

    print("PIVOT MAIN")
    print(df_main)
    sys.exit(0)

    df_main = (df_main
        .drop(columns=['lang_setting'])       
        .pivot_table(
            index=["model_name", "shots", "learning_rate"],
            columns=["lang"],
            values="metric",
            aggfunc=["mean"] #"count"
        ).reset_index()
    )

    # clean columns
    df_main.columns = [f"{c[0]}_{c[1]}" if c[1] else c[0] for c in df_main.columns]




    # score each (model, lr, seed) by averaging across shots + langs
    # score each (model, lr) by averaging across shots + langs
    lr_score = (
        df_pivot[df_pivot["shots"] != 0]
        .groupby(["model_name", "resource", "learning_rate"])[lang_cols]
        .mean()          # avg over shots (and seeds if present)
        .mean(axis=1)    # avg over langs
        .rename("lr_score")
        .reset_index()
    )


    # save
    few_results_check.to_excel(f"checks/few_shot_check_{METRIC}.xlsx")

    # filter and privor to get mean per model/shots/learning_rate/seed (seed! important as we want to get variation over seeds later)
    df_pivot = (df[(df['lang_setting'] == lang_setting) & (df['learning_rate'] != 0.000005)]
        .drop(columns=['lang_setting'])       
        .pivot_table(
            index=["model_name", "shots", "learning_rate"],
            columns=["lang"],
            values="metric",
            aggfunc=["mean"] #"count"
        ).reset_index()
    )
    
    # clean columns
    df_pivot.columns = [f"{c[0]}_{c[1]}" if c[1] else c[0] for c in df_pivot.columns]
    
   

    # pick best LR per model
    best_lr = lr_score.loc[
        lr_score.groupby("model_name")["lr_score"].idxmax()
    ][["model_name", "learning_rate"]].reset_index(drop=True)

    print("best")
    print(best_lr)


    print("PIVOT")
    print(df_pivot)

    # keep ALL shots rows, but only for the selected LR (per model+seed)
    df_pivot = df_pivot.merge(best_lr, on=["model_name", "learning_rate"], how="inner")

    print(df_pivot)
    sys.exit(0)

    numeric_cols = [col for col in df_pivot.columns if col.startswith("mean_")]

    # Get shot 0 values for each model and seed
    zero_shot = df_pivot[df_pivot['shots'] == 0][['model_name', 'seed'] + numeric_cols].copy()
    zero_shot = zero_shot.rename(columns={col: f"{col}_zero" for col in numeric_cols})

    # Merge back on model_name and seed
    result_df = df_pivot.merge(zero_shot, on=['model_name', 'seed'])

    # Calculate differences only for non-zero shots, keep original for shot 0
    for col in numeric_cols:
        result_df[col] = result_df.apply(
            lambda row: row[col] if row['shots'] == 0 else row[col] - row[f"{col}_zero"],
            axis=1
        )


    # Drop the _zero columns
    result_df = result_df.drop(columns=[f"{col}_zero" for col in numeric_cols])
    
    # now average over seeds
    result_df = (result_df
                .groupby(["model_name", "shots"])[numeric_cols]
                .agg(['mean', 'std'])
                .reset_index(drop=False)
        )

    # flatten columns
    result_df.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in result_df.columns]
    result_df.columns = [col.replace("mean_", "") for col in result_df.columns]


    # create resource group delta averages
    low_resource_langs = [col for col in result_df.columns if col.endswith("_mean") and col[:2] in LANGS['low']]
    medium_resource_langs = [col for col in result_df.columns if col.endswith("_mean") and col[:2] in LANGS['medium']]

    result_df['low_mean'] = result_df[low_resource_langs].mean(axis=1)
    result_df['medium_mean'] = result_df[medium_resource_langs].mean(axis=1)

    # multiply all by 100
    for col in result_df.columns:
        if col not in ["model_name", "shots"]:
            result_df[col] = result_df[col] * 100.0

    # sort rows by model and shots
    result_df['model_name'] = pd.Categorical(result_df['model_name'].apply(sft_names), categories=['FTL', 'TOL', 'ES', "mBert"], ordered=True)
    result_df = result_df.sort_values(['model_name', 'shots']).reset_index(drop=True)

    print("="*60)
    print("FEW-SHOT MAIN TABLE DF")
    print(result_df)
    print("="*60)
    
    return result_df

# OG
# def few_shot_main_table_df(rows: list[dict], lang_setting: str = "main") -> pd.DataFrame:
#     df = pd.DataFrame(rows)


#     # ------------------------
#     # CHECK TABLE
#     # ------------------------
#     few_results_check = (
#         df[(df['learning_rate'] != 0.000005)]
#         .groupby(["lang_setting", "model_name", "lang", "shots", "learning_rate"])
#         .agg(
#             metric_mean=("metric", "mean"),
#             metric_std=("metric", "std"),
#             metric_count=("metric", "count"),
#             sources=("source", list),
#             metrics_lst=("metric", list),
#             seeds_lst=("seed", list),
#         )
#         .sort_values(['lang_setting', 'lang', 'model_name', "shots", "learning_rate"]).reset_index()
#     )

#     print("="*60)
#     print("="*60)
#     print("CHECK FEW-SHOT RESULTS")    
#     print(few_results_check)
#     print("="*60)
#     print("="*60)

#     # save
#     few_results_check.to_excel(f"checks/few_shot_check_{METRIC}.xlsx")

#     # filter and privor to get mean per model/shots/learning_rate/seed (seed! important as we want to get variation over seeds later)
#     df_pivot = (df[(df['lang_setting'] == lang_setting) & (df['learning_rate'] != 0.000005)]
#         .drop(columns=['lang_setting'])       
#         .pivot_table(
#             index=["model_name", "shots", "learning_rate", "seed"],
#             columns=["lang"],
#             values="metric",
#             aggfunc=["mean"] #"count"
#         ).reset_index()
#     )
    
#     # clean columns
#     df_pivot.columns = [f"{c[0]}_{c[1]}" if c[1] else c[0] for c in df_pivot.columns]
    
#     # gen lang mean across seeds!
#     # mean over all langs and seed (not sure whether perfect but ok for now)
#     lang_cols = [col for col in df_pivot.columns if col.startswith("mean_")]
#     df_pivot['metric_mean'] = df_pivot.groupby(['model_name', 'shots', 'learning_rate'])[lang_cols].transform('mean').mean(axis=1)

#     # use this mean to pick best learning rate per model and shots; keep seeds
#     df_pivot = (df_pivot
#                 .groupby(["model_name", "shots", "seed"])
#                 .apply(lambda x: x.loc[x["metric_mean"].idxmax()])
#                 .drop(columns=["metric_mean", "learning_rate"])
#                 .reset_index(drop=True))

#     numeric_cols = [col for col in df_pivot.columns if col.startswith("mean_")]

#     # Get shot 0 values for each model and seed
#     zero_shot = df_pivot[df_pivot['shots'] == 0][['model_name', 'seed'] + numeric_cols].copy()
#     zero_shot = zero_shot.rename(columns={col: f"{col}_zero" for col in numeric_cols})

#     # Merge back on model_name and seed
#     result_df = df_pivot.merge(zero_shot, on=['model_name', 'seed'])

#     # Calculate differences only for non-zero shots, keep original for shot 0
#     for col in numeric_cols:
#         result_df[col] = result_df.apply(
#             lambda row: row[col] if row['shots'] == 0 else row[col] - row[f"{col}_zero"],
#             axis=1
#         )


#     # Drop the _zero columns
#     result_df = result_df.drop(columns=[f"{col}_zero" for col in numeric_cols])
    
#     # now average over seeds
#     result_df = (result_df
#                 .groupby(["model_name", "shots"])[numeric_cols]
#                 .agg(['mean', 'std'])
#                 .reset_index(drop=False)
#         )

#     # flatten columns
#     result_df.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in result_df.columns]
#     result_df.columns = [col.replace("mean_", "") for col in result_df.columns]


#     # create resource group delta averages
#     low_resource_langs = [col for col in result_df.columns if col.endswith("_mean") and col[:2] in LANGS['low']]
#     medium_resource_langs = [col for col in result_df.columns if col.endswith("_mean") and col[:2] in LANGS['medium']]

#     result_df['low_mean'] = result_df[low_resource_langs].mean(axis=1)
#     result_df['medium_mean'] = result_df[medium_resource_langs].mean(axis=1)

#     # multiply all by 100
#     for col in result_df.columns:
#         if col not in ["model_name", "shots"]:
#             result_df[col] = result_df[col] * 100.0

#     # sort rows by model and shots
#     result_df['model_name'] = pd.Categorical(result_df['model_name'].apply(sft_names), categories=['FTL', 'TOL', 'ES', "mBert"], ordered=True)
#     result_df = result_df.sort_values(['model_name', 'shots']).reset_index(drop=True)

#     print("="*60)
#     print("FEW-SHOT MAIN TABLE DF")
#     print(result_df)
#     print("="*60)
    
#     return result_df

# OG
# def  few_shot_table(df: pd.DataFrame, lang_setting: str) -> None:
    
#     n_langs = 12 # 10 langs + 2 avgs
    
#     table = "\n\n"
#     colspec = "lc" + "c" * (n_langs) # 10 langs + avgs

#     panel_name = "Non-Parallel Data" if lang_setting == "main" else "Parallel Data"
#     header = (
#         " & & "
#         + f"\\multicolumn{{12}}{{c}}{{\\textbf{{{panel_name}}}}} \\\\ \n"
#     )
#     header += "\\cmidrule(lr){3-14} \n"
#     header += " & & \\multicolumn{" + str(n_langs // 2) + "}{c}{\\textbf{Medium-Resource}} & \\multicolumn{" + str(n_langs // 2) + "}{c}{\\textbf{Low-Resource}} \\\\ \n"
#     header += "\\cmidrule(lr){3-8} \\cmidrule(lr){9-14} \n"
#     header += (
#         "\\textbf{Model} & \\textbf{Shots} & "
#         + " & ".join(f"\\textbf{{{lang}}}" for lang in (LANGS["medium"]))
#         + " & $\\Delta$ \\textbf{Avg} & "
#         + " & ".join(f"\\textbf{{{lang}}}" for lang in (LANGS["low"]))
#         + " & $\\Delta$ \\textbf{Avg} "
#         + " \\\\ \n"
#     )
#     header += "\\toprule\n"

#     table += "\\begin{tabular}{" + colspec + "}\n"
#     table += header

#     prev_model = None
#     for _, row in df.iterrows():

#         if prev_model != row['model_name'] and prev_model is not None:
#             table += "\\cmidrule(lr){2-" + str(n_langs+2) + "}\n"

#         if prev_model is None or prev_model != row['model_name']:
#             n_model_rows = df[df['model_name'] == row['model_name']].shape[0]
#             model_name = row['model_name'] if "mBert" in row['model_name'] else "Llama3-8B " + f"({row['model_name']})"
#             line = f"\\multirow{{{n_model_rows}}}{{*}}{{{model_name}}} & \\cellcolor{{gray!25}} {row['shots']} "
#         else:
#             if row['shots'] == 50:
#                 line = f" & $\\Delta$ \\; {row['shots']}"
#             else:
#                 line = f" & $\\Delta$ {row['shots']}"

#         # MEDIUM
#         for lang in (LANGS["medium"]):
#             if row['shots'] == 0:
#                 line += f"&  {row[f'{lang}_mean']:.2f} \\scriptsize{{($\\pm${row[f'{lang}_std']:.2f})}}"
#             else:
#                 line += f"&  \\posneg{{{row[f'{lang}_mean']:.2f}}} \\scriptsize{{($\\pm${row[f'{lang}_std']:.2f})}}"
        
#         # add medium avg
#         if row['shots'] == 0:
#             line += f" &  "
#         else:
#             line += f" &  \\posneg{{{row['medium_mean']:.2f}}} "
        
#         # LOW
#         for lang in (LANGS["low"]):
#             if row['shots'] == 0:
#                 line += f"&  {row[f'{lang}_mean']:.2f} \\scriptsize{{($\\pm${row[f'{lang}_std']:.2f})}}"
#             else:
#                 line += f"&  \\posneg{{{row[f'{lang}_mean']:.2f}}} \\scriptsize{{($\\pm${row[f'{lang}_std']:.2f})}}"

#         # add low avg
#         if row['shots'] == 0:
#             line += f" &  "
#         else:
#             line += f" &  \\posneg{{{row['low_mean']:.2f}}} "

#         line += "\\\\\n"
#         table += line

        
#         prev_model = row['model_name']

#     table += "\\bottomrule\n"
#     table += "\\end{tabular}\n"
#     table += "\n\n"
    
#     print("="*80)
#     print(f"{lang_setting.upper()} FEW-SHOT TABLE")
#     print("="*80)
#     print("\n\n")
#     print(table)
#     print("\n\n")
#     print("="*80)

def  few_shot_table(df: pd.DataFrame, lang_setting: str) -> None:
    
    n_langs = 12 # 10 langs + 2 avgs
    
    table = "\n\n"
    colspec = "lc" + "c" * (n_langs) # 10 langs + avgs

    panel_name = "Non-Parallel Data" if lang_setting == "main" else "Parallel Data"
    header = (
        " & & "
        + f"\\multicolumn{{12}}{{c}}{{\\textbf{{{panel_name}}}}} \\\\ \n"
    )
    header += "\\cmidrule(lr){3-14} \n"
    header += " & & \\multicolumn{" + str(n_langs // 2) + "}{c}{\\textbf{Medium-Resource}} & \\multicolumn{" + str(n_langs // 2) + "}{c}{\\textbf{Low-Resource}} \\\\ \n"
    header += "\\cmidrule(lr){3-8} \\cmidrule(lr){9-14} \n"
    header += (
        "\\textbf{Model} & \\textbf{Shots} & "
        + " & ".join(f"\\textbf{{{lang}}}" for lang in (LANGS["medium"]))
        + " & $\\Delta$ \\textbf{Avg} & "
        + " & ".join(f"\\textbf{{{lang}}}" for lang in (LANGS["low"]))
        + " & $\\Delta$ \\textbf{Avg} "
        + " \\\\ \n"
    )
    header += "\\toprule\n"

    table += "\\begin{tabular}{" + colspec + "}\n"
    table += header

    prev_model = None
    for _, row in df.iterrows():

        if prev_model != row['model_name'] and prev_model is not None:
            table += "\\cmidrule(lr){2-" + str(n_langs+2) + "}\n"

        if prev_model is None or prev_model != row['model_name']:
            n_model_rows = df[df['model_name'] == row['model_name']].shape[0]
            model_name = row['model_name'] if "mBert" in row['model_name'] else "Llama3-8B " + f"({row['model_name']})"
            line = f"\\multirow{{{n_model_rows}}}{{*}}{{{model_name}}} & \\cellcolor{{gray!25}} {row['shots']} "
        else:
            if row['shots'] == 50:
                line = f" & $\\Delta$ \\; {row['shots']}"
            else:
                line = f" & $\\Delta$ {row['shots']}"

        # MEDIUM
        for lang in (LANGS["medium"]):
            if row['shots'] == 0:
                line += f"&  {row[f'mean_{lang}']:.2f} \\scriptsize{{($\\pm${row[f'std_{lang}']:.2f})}}"
            else:
                line += f"&  \\posneg{{{row[f'mean_{lang}']:.2f}}} \\scriptsize{{($\\pm${row[f'std_{lang}']:.2f})}}"
        
        # add medium avg
        if row['shots'] == 0:
            line += f" &  "
        else:
            line += f" &  \\posneg{{{row['medium_mean']:.2f}}} "
        
        # LOW
        for lang in (LANGS["low"]):
            if row['shots'] == 0:
                line += f"&  {row[f'mean_{lang}']:.2f} \\scriptsize{{($\\pm${row[f'std_{lang}']:.2f})}}"
            else:
                line += f"&  \\posneg{{{row[f'mean_{lang}']:.2f}}} \\scriptsize{{($\\pm${row[f'std_{lang}']:.2f})}}"

        # add low avg
        if row['shots'] == 0:
            line += f" &  "
        else:
            line += f" &  \\posneg{{{row['low_mean']:.2f}}} "

        line += "\\\\\n"
        table += line

        
        prev_model = row['model_name']

    table += "\\bottomrule\n"
    table += "\\end{tabular}\n"
    table += "\n\n"
    
    print("="*80)
    print(f"{lang_setting.upper()} FEW-SHOT TABLE")
    print("="*80)
    print("\n\n")
    print(table)
    print("\n\n")
    print("="*80)

# ----------------------------------------------------------------------
# Few-shot LR
# ----------------------------------------------------------------------

# def few_shot_lr_table_df(rows: list[dict]) -> pd.DataFrame:
    

#     df = pd.DataFrame(rows)

#     print
#     df_shots = df[
#         (df["shots"] > 0)
#         & (df["learning_rate"] != 5e-6)
#     ]
#     # add resource level

#     df_shots["resource"] = df_shots["lang"].apply(get_resource)

#     df_shots = (df_shots
#         .pivot_table(
#             index=["lang_setting", "model_name", "resource", "lang", "learning_rate"],
#             columns=["shots"],
#             values="metric",
#             aggfunc=["mean"] #"count"
#         ).reset_index()
#     ) 

#     # flatten column names
#     df_shots.columns = [f"{c[0]}_{c[1]}" if c[1] else c[0] for c in df_shots.columns]
#     # get zero shot helper
#     zero_results = zero_shot_helper_df(rows)
#     # add zero-shot values
#     df_shots = df_shots.merge(zero_results,
#                             on=["lang_setting", "resource", "lang", "model_name"],
#                             how="left")

#     # rename
#     df_shots = df_shots.rename(columns={"shot_0_mean":'mean_0', "shot_0_std":'std_0'})

#     # # drop std
#     # df_shots = df_shots.drop(columns=[col for col in df_shots.columns if col.startswith("std_")])
    

#     print(df_shots)
#     print("AFTER")
#     df_shots = df_shots.pivot_table(
#         index=["model_name", "resource", "lang", "learning_rate", "seed"],
#         columns="lang_setting", 
#         values=[col for col in df_shots.columns if col.startswith("mean") or col.startswith("std")],
#         aggfunc="first"
#     ).reset_index()

#     # flatten columns
#     df_shots.columns = [
#         f"{metric}_{setting}" if setting else metric
#         for metric, setting in df_shots.columns
#     ]


#     # gen differences
#     for lang_setting in df['lang_setting'].unique():
#         for shot in SHOTS[1:]:
#             df_shots[f'diff_shot_{shot}_{lang_setting}'] = df_shots[f'mean_{shot}_{lang_setting}'] - df_shots[f'mean_0_{lang_setting}']

#         df_shots[f'diff_avg_{lang_setting}'] = df_shots[[f'diff_shot_{shot}_{lang_setting}' for shot in SHOTS[1:]]].mean(axis=1)
#     # df_shots = df_shots.drop(columns=[col for col in df_shots.columns if col.startswith("std_")])
    

#     # mean over langs and seeds
#     df_shots = (df_shots
#                 .groupby(["model_name", "resource", "learning_rate"])
#                 .mean(numeric_only=True)
#                 .reset_index()
#         )
    

#         # multiply all scores by 100
#     for col in df_shots.columns:
#         if col not in ["model_name", "resource", "learning_rate"]:
#             df_shots[col] = df_shots[col] * 100.0

#     # # drop seed and trans
#     # df_shots.to_excel("checks/TEST.xlsx")

#     # sys.exit(0)

#     # Sort rows
#     df_shots['resource'] = pd.Categorical(df_shots['resource'], categories=['medium', 'low'], ordered=True)
#     df_shots['model_name'] = pd.Categorical(df_shots['model_name'].apply(sft_names), categories=['FTL', 'TOL', 'ES', "mBert"], ordered=True)
#     df_shots = df_shots.sort_values(['resource', 'model_name', 'learning_rate'],
#                                                     ascending=[True, True, False]).reset_index(drop=True)
    

#     # print(df_shots)
#     # # Add averages
#     # for lang_setting in df['lang_setting'].unique():        
#     #     for shot in SHOTS[1:]:
#     #         df_shots[f'diff_shot_{shot}_{lang_setting}'] = df_shots[f'mean_{shot}_{lang_setting}'] - df_shots[f'mean_0_{lang_setting}']
    
#     #     diff_cols = [f'diff_shot_{shot}_{lang_setting}' for shot in SHOTS[1:]]
#     #     df_shots[f'diff_avg_{lang_setting}'] = df_shots[diff_cols].mean(axis=1)

#     print("="*60)
#     print("FEW-SHOT LEARNING RATE DF")
#     print(df_shots)
#     print("="*60)

#     print(df_shots[[col for col in df_shots.columns if col.startswith("std") ]])

#     return df_shots


# OG
def few_shot_lr_table_df(rows: list[dict]) -> pd.DataFrame:
    

    df = pd.DataFrame(rows)

    print
    df_shots = df[
        (df["shots"] > 0)
        & (df["learning_rate"] != 5e-6)
    ]
    # add resource level

    df_shots["resource"] = df_shots["lang"].apply(get_resource)

    df_shots = (df_shots
        .pivot_table(
            index=["lang_setting", "model_name", "resource", "learning_rate"],
            columns=["shots"],
            values="metric",
            aggfunc=["mean", "std"] #"count"
        ).reset_index()
    )

    # flatten column names
    df_shots.columns = [f"{c[0]}_{c[1]}" if c[1] else c[0] for c in df_shots.columns]
    # get zero shot helper
    zero_results = zero_shot_helper_df(rows)
    # group by resource
    zero_results = zero_results.groupby(["lang_setting", "resource", "model_name"]).mean(numeric_only=True).reset_index()

    

    # add zero-shot values
    df_shots = df_shots.merge(zero_results,
                            on=["lang_setting", "resource", "model_name"],
                            how="left")

    # rename
    df_shots = df_shots.rename(columns={"shot_0_mean":'mean_0', "shot_0_std":'std_0'})

    # # drop std
    # df_shots = df_shots.drop(columns=[col for col in df_shots.columns if col.startswith("std_")])
    

    print(df_shots)
    print("AFTER")
    df_shots = df_shots.pivot_table(
        index=["model_name", "resource", "learning_rate"],
        columns="lang_setting", 
        values=[col for col in df_shots.columns if col.startswith("mean") or col.startswith("diff") or col.startswith("std")],
        aggfunc="first"
    ).reset_index()

    # flatten columns
    df_shots.columns = [
        f"{metric}_{setting}" if setting else metric
        for metric, setting in df_shots.columns
    ]

    # Sort rows
    df_shots['resource'] = pd.Categorical(df_shots['resource'], categories=['medium', 'low'], ordered=True)
    df_shots['model_name'] = pd.Categorical(df_shots['model_name'].apply(sft_names), categories=['FTL', 'TOL', 'ES', "mBert"], ordered=True)
    df_shots = df_shots.sort_values(['resource', 'model_name', 'learning_rate'],
                                                    ascending=[True, True, False]).reset_index(drop=True)
    
    # multiply all scores by 100
    for col in df_shots.columns:
        if col not in ["model_name", "resource", "learning_rate"]:
            df_shots[col] = df_shots[col] * 100.0

    print(df_shots)
    # Add averages
    for lang_setting in df['lang_setting'].unique():        
        for shot in SHOTS[1:]:
            df_shots[f'diff_shot_{shot}_{lang_setting}'] = df_shots[f'mean_{shot}_{lang_setting}'] - df_shots[f'mean_0_{lang_setting}']
    
        diff_cols = [f'diff_shot_{shot}_{lang_setting}' for shot in SHOTS[1:]]
        df_shots[f'diff_avg_{lang_setting}'] = df_shots[diff_cols].mean(axis=1)

    print("="*60)
    print("FEW-SHOT LEARNING RATE DF")
    print(df_shots)
    print("="*60)

    return df_shots

def few_shot_learning_rate_table(df: pd.DataFrame) -> str:
    
    shot_length = len(SHOTS) -1 + 1 # - 1 to excl zero and + 1 for avg column
    table = "\n\n"
    colspec = "ll" + "c" * ((shot_length) + 1)

    header = (
        " & & "
        + f"\\multicolumn{{{shot_length}}}{{c}}{{\\textbf{{Shots}}}} \\\\ \n"
    )
    header += "\\cmidrule(lr){3-" + str(2 + shot_length) + "} \n"
    header += "\\textbf{Model}   & \\textbf{LR} & $k=50$ & $k=100$ & $k=250$ & $k=500$ & \\textbf{Avg} $\\Delta$ \\\\ \\toprule\n"

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
            
            if row['model_name'] != "mBert":
                model_name  = "Llama 3-8B (" + sft_names(row['model_name']) + ")"
            else:
                model_name = row['model_name']

            line = f"\\multirow{{{n_model_rows}}}{{*}}{{{model_name}}} & {row['learning_rate']:.0e} "
            if prev_models and row['resource'] == prev_resource:
                table += "\\cmidrule(lr){2-" + str(2 + shot_length) + "}\n"
        else:
            line = f" & {row['learning_rate']:.0e} "

        # add shots with diff
        for shot in SHOTS[1:]:
            diff = row[f'diff_shot_{shot}_main']
            line += f"& \\makecell{{{row[f'mean_{shot}_main']:.2f} \\\\[-3pt] \\small{{($\\pm${{{row[f'std_{shot}_main']:.2f}}})}}}}  \\small{{(\\posneg{{{diff:.2f}}})}} "
            
        # add avg
        
        if row[f'diff_avg_main'] == df[(df['model_name'] == row['model_name']) & (df['resource'] == row['resource'])][f'diff_avg_main'].max():
            line += f"& \\textbf{{{row[f'diff_avg_main']:.2f}}} \\\\ \n"
        else:
            line += f"& {row[f'diff_avg_main']:.2f} \\\\ \n"

        table += line   

        
        prev_resource = row['resource']
        prev_models = row['model_name']

    table += "\\bottomrule\n"
    table += "\\end{tabular}\n"
    table += "\n\n"
    
    # table = table.replace("nan", "-")
    print("\n\n")
    print(table)
    print("\n\n")

# ----------------------------------------------------------------------
# Misc but old
# ----------------------------------------------------------------------

# def all_results_latex_table(df: pd.DataFrame) -> str:
#     print(df.columns)
#     table = "\n\n"
#     colspec = "ll" + "c" * (len(df.columns) - 1)

#     header = " & & & \\multicolumn{" + str(len(SHOTS)) + "}{c}{\\textbf{Parallel}} & \\multicolumn{" + str(len(SHOTS)) + "}{c}{\\textbf{Non-Parallel}} \\\\ \n"
#     header += "\\cmidrule(lr){4-" + str(3 + len(SHOTS)) + "} \\cmidrule(lr){" + str(4 + len(SHOTS)) + "-" + str(3 + 2*len(SHOTS)) + "} \n"
#     header += "\\textbf{Lang} & \\textbf{Model} & \\textbf{LR} & $k=0$ & $k=50$ & $k=100$ & $k=250$ & $k=500$ & $k=0$ & $k=50$ & $k=100$ & $k=250$ & $k=500$ \\\\ \n"
#     header += "\\toprule\n"

#     table += "\\begin{longtable}{" + colspec + "}\n"
#     table += header

#     prev_lang = None
#     prev_model = None
#     prev_resource = None
#     for _, row in df.iterrows():

#         if prev_lang != row['lang'] and prev_lang is not None:
#             table += "\\cmidrule(lr){2-" + str(len(df.columns)) + "}\n"

#         if row['resource'] == "medium" and prev_resource is None:
#             line = f"\\multicolumn{{{len(df.columns)}}}{{c}}{{\\textbf{{Medium Resource}}}} \\\\\n"
#             line += "\\midrule\n"
#             prev_resource = "medium"
#             table += line

#         if row['resource'] == "low" and prev_resource == "medium":
#             line = f"\\multicolumn{{{len(df.columns)}}}{{c}}{{\\textbf{{Low Resource}}}} \\\\\n"
#             line += "\\midrule\n"
#             prev_resource = "low"
#             table += line

#         if prev_lang is None or prev_lang != row['lang']:
#             n_lang_rows = df[df['lang'] == row['lang']].shape[0]
#             line = f"\\multirow{{{n_lang_rows}}}{{*}}{{{row['lang']}}} & {row['model_name']} & {row['learning_rate']:.0e} "
#         else:
#             line = f" & {row['model_name']} & {row['learning_rate']:.0e} "


#         for shot in SHOTS:
#             line += f"& {row[f'mean_{shot}_translation']:.2f} ($\\pm${row[f'std_{shot}_translation']:.2f})"
#         for shot in SHOTS:
#             line += f"& {row[f'mean_{shot}_main']:.2f} ($\\pm${row[f'std_{shot}_main']:.2f})"
#         line += "\\\\\n"
#         table += line

        
#         prev_lang = row['lang']
#         prev_resource = row['resource']

#     table += "\\bottomrule\n"
#     table += "\\end{longtable}\n"
#     table += "\n\n"
    
#     table = table.replace("nan", "-")
#     print("\n\n")
#     print(table)
#     print("\n\n")

# def generate_main_figure(df: pd.DataFrame):

#     # rename shot_0
#     df = df.rename(columns={"shot_0_mean": "mean_0"})

#     plt.rcParams.update({"font.size": 12})

#     fig = plt.figure(
#         figsize=(12, 6),
#         constrained_layout=True,
#     )

#     lang_settings = ["translation", "main"]
#     row_titles = ["Parallel", "Non-Parallel"]
#     col_titles = ["Medium Resource Languages", "Low Resource Languages"]

#     x = SHOTS

#     # ------------------------------------------------------------------
#     # Consistent styling
#     # ------------------------------------------------------------------
#     models = sorted(df["model_name"].unique())
#     cmap = cm.get_cmap("tab10", len(models))
#     color_map = {m: cmap(i) for i, m in enumerate(models)}
#     markers = ['o', 's', '^', 'v', 'D', '*', '+', 'x'][:len(models)]

#     # color and marker map per models
#     marker_map = {model: markers[i] for i, model in enumerate(models)}
#     color_map = {model: cmap(i) for i, model in enumerate(models)}

#     # ------------------------------------------------------------------
#     # Subfigures
#     # ------------------------------------------------------------------
#     subfigs = fig.subfigures(nrows=2, ncols=1)

#     for row_i, (subfig, lang_setting) in enumerate(zip(subfigs, lang_settings)):
#         subfig.suptitle(row_titles[row_i], fontsize=20)

#         axs = subfig.subplots(nrows=1, ncols=2, sharex=True, sharey=True)

#         df_subset = df[df["lang_setting"] == lang_setting]

#         for col_i, resource in enumerate(["medium", "low"]):
#             ax = axs[col_i]

#             df_subset_resource = df_subset[df_subset["resource"] == resource]

#             for model_name in models:
#                 df_model = df_subset_resource[df_subset_resource["model_name"] == model_name]
#                 if df_model.empty:
#                     continue

#                 y = [df_model[f'mean_{shot}'] for shot in SHOTS]

#                 ax.plot(
#                     x,
#                     y,
#                     marker=marker_map[model_name],
#                     color=color_map[model_name],
#                     linewidth=2,
#                     markersize=6,
#                     label=model_name,
#                 )

#             # column titles only on top row
#             if row_i == 0:
#                 ax.set_title(col_titles[col_i], fontsize=16)

#             # x label only bottom row
#             if row_i == len(lang_settings) - 1:
#                 ax.set_xlabel(r"$N$ Target Language Samples")

#             # y label only left column
#             if col_i == 0:
#                 ax.set_ylabel(f"{METRIC.upper()} score")

#             ax.set_xticks(x)
#             ax.grid(True)
#             ax.set_axisbelow(True)

#     # ------------------------------------------------------------------
#     # Shared legend
#     # ------------------------------------------------------------------
#     handles, labels = subfigs[0].axes[0].get_legend_handles_labels()
#     fig.legend(
#         handles,
#         labels,
#         loc="lower center",
#         ncol=len(models),
#         bbox_to_anchor=(0.5, -0.10),
#         frameon=True,
#         shadow=True,
#     )

#     fig.savefig(f"plots/main_fig_{METRIC}.pdf", format="pdf", bbox_inches="tight")
            

def main():

    rows = load_all_models(path=EXP2_DIR)

    if ITEM == "zero":
        zero_shot_df = create_zero_shot_df(rows)
        zero_shot_plot(zero_shot_df)


    if ITEM == "LR" or ITEM == "FEW":
        if ITEM == "LR":
            df_learning_rate = few_shot_lr_table_df(rows)
            print("="*60)
            print("METRIC: ", METRIC)
            print("="*60)
            few_shot_learning_rate_table(df_learning_rate)

        if ITEM == "FEW":
            df = few_shot_main_table_df(rows, lang_setting=LANG_SETTING)
            print("="*60)
            print("METRIC: ", METRIC)
            print("="*60)
            few_shot_table(df, lang_setting=LANG_SETTING)
        
if __name__ == "__main__":
    main()
