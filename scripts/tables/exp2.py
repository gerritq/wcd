import os
from pdb import run

import json
import glob
from collections import defaultdict
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

from itertools import cycle

NON_MONO_MARKERS = cycle(["s", "^", "v", "D", "P", "X", "*"])

# ----------------------------------------------------------------------
# Configs
# ----------------------------------------------------------------------
BASE_DIR = os.getenv("BASE_WCD")
EXP1_DIR = os.path.join(BASE_DIR, "data/exp1")
EXP2_DIR = os.path.join(BASE_DIR, "data/exp2")

LAST_EPOCH = True

def collect_monoliongual_metrics(configs: dict,) -> dict:
    lang_dir = os.path.join(EXP1_DIR, configs["test_lang"])
    # get all meta files
    all_lang_runs = [f for f in os.listdir(lang_dir)
                      if os.path.isdir(os.path.join(lang_dir, f))]
    evals = []
    for run_dir in all_lang_runs:
        meta_paths = glob.glob(os.path.join(lang_dir, run_dir, "meta_*.json"))
        
        # skipt those; as we only look for one meta per run
        if len(meta_paths) == 0 or len(meta_paths) > 1:
            continue
        
        meta = load_meta(meta_paths[0])
        
        if "training_size" not in meta:
            continue

        if meta['training_size'] in [200, 400, 600, 800] and meta['model_type'] in configs['model_type']:            
            
            if LAST_EPOCH:
                evals.append((meta['training_size'],
                              meta['dev_metrics'][-1]['metrics']['accuracy'],
                              meta['test_metrics'][-1]['metrics']['accuracy'],))
            else:
                best_epoch = find_best_epoch(meta, metric="accuracy")
                evals.append((meta['training_size'],
                              best_epoch['dev_metric'],
                              best_epoch['test_metric'],))
            
    evals = sorted(evals, key=lambda x: x[0])
    out = {"training_langs": [configs["test_lang"]], "evals": evals, "run_dir": lang_dir}
    print("="*20)
    print("MONOLINGUAL EVALS")
    print(out)
    print("="*20)
    return out

def collect_best_monolingual_baseline(configs: dict,) -> dict:
    lang_dir = os.path.join(EXP1_DIR, configs["test_lang"])
    # get all meta files
    all_lang_runs = [f for f in os.listdir(lang_dir)
                      if os.path.isdir(os.path.join(lang_dir, f))]
    
    best_acc = 0.0
    for run_dir in all_lang_runs:
        meta_paths = glob.glob(os.path.join(lang_dir, run_dir, "meta_*.json"))
        for meta_path in meta_paths:
            meta = load_meta(meta_path)

            if meta['model_type'] in configs['model_type']:
                if meta['training_size'] == 5000: 
                    meta = load_meta(meta_path)
                    best_epoch = find_best_epoch(meta, metric="accuracy")
                    test_acc = best_epoch['test_metric']
                    if test_acc > best_acc:
                        best_acc = test_acc
    configs['mono_baseline'] = best_acc
    return configs

def load_meta(meta_path: str):
    with open(meta_path, "r") as f:
        meta = json.load(f)
    return meta

def find_best_epoch(meta: dict, metric: str) -> dict:
    best_epoch = None
    best_dev_metric = float("-inf")

    dev_metrics = meta.get("dev_metrics", [])
    test_metrics = meta.get("test_metrics", [])

    for dev_entry in dev_metrics:
        epoch = dev_entry["epoch"]
        dev_metric = dev_entry["metrics"][metric]

        if dev_metric > best_dev_metric:
            test_entry = next(
                (t for t in test_metrics if t["epoch"] == epoch),
                None
            )
            if test_entry is None:
                continue

            best_dev_metric = dev_metric
            best_epoch = {
                "epoch": epoch,
                "dev_metric": dev_metric,
                "test_metric": test_entry["metrics"][metric],
            }

    return best_epoch

def get_metas(run_dict: dict) -> list[tuple[int, float, float]]:

    evals = []

    all_metas = glob.glob(os.path.join(run_dict["run_dir"], "meta_*.json"))

    for meta_path in all_metas:
        print(meta_path)

        meta = load_meta(meta_path)

        # if met path is meta_1
        if 'meta_1' in os.path.basename(meta_path):
            training_size = 0
        else:
            training_size = meta['training_size']

        if LAST_EPOCH:
            best_dev = meta['dev_metrics'][-1]['metrics']['accuracy']
            test_acc = meta['test_metrics'][-1]['metrics']['accuracy']
            evals.append((training_size, best_dev, test_acc))
        else:
            best_epoch = find_best_epoch(meta, metric="accuracy")
            evals.append((training_size, best_epoch['dev_metric'], best_epoch['test_metric']))
    
    run_dict['evals'] = evals
    return run_dict

from itertools import cycle

NON_MONO_MARKERS = cycle(["s", "^", "v", "D", "P", "X", "*"])

# ... all your existing code above this stays the same ...


def plot_2_stage_ft(all_evals: list, configs: dict):
    """
    Produce an individual plot for a single target language and save it
    to plots/exp2_{lang}_{model}.png (original behaviour).
    """
    title = f"Target Language: {configs['test_lang']}"
    model = "slm" if "slm" in configs['model_type'] else "clf"
    out_path = os.path.join("plots", f"exp2_{configs['test_lang']}_{model}.png")
    os.makedirs("plots", exist_ok=True)

    # fresh marker cycle to keep markers stable per plot
    non_mono_markers = cycle(["s", "^", "v", "D", "P", "X", "*"])

    plt.figure()

    for run_dict in all_evals:
        training_langs = run_dict['training_langs']
        evals = run_dict['evals']
        xs = [ts for ts, _, _ in evals]
        ys = [test for _, _, test in evals]
        xs, ys = zip(*sorted(zip(xs, ys)))

        if len(training_langs) == 1 and training_langs[0] == configs['test_lang']:
            # Monolingual FT — fixed style
            plt.plot(
                xs, ys,
                marker="o",
                color="black",
                linestyle="--",
                linewidth=2,
                alpha=0.4,
                label="Monolingual FT",
            )
        else:
            # Multilingual / 2-stage runs — different marker
            marker = next(non_mono_markers)
            plt.plot(
                xs, ys,
                marker=marker,
                linestyle="--",
                linewidth=2,
                alpha=0.7,
                label="1st-stage: " + ", ".join(training_langs),
            )

    plt.axhline(
        y=configs['mono_baseline'],
        color='gray',
        linestyle=':',
        linewidth=2,
        alpha=0.8,
        label='Monolingual Baseline',
    )

    plt.xticks([0, 200, 400, 600, 800])
    plt.title(title)
    plt.xlabel("Target-Language Fine-Tuning Samples")
    plt.ylabel("Accuracy")
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
        frameon=True,
    )
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_2_stage_ft_on_ax(ax, all_evals: list, configs: dict):
    """
    Plot the same data as plot_2_stage_ft, but into a given Axes (for the grid figure).
    """
    ax.set_title(configs["test_lang"].upper())

    # local marker cycle so each subplot has its own set
    non_mono_markers = cycle(["s", "^", "v", "D", "P", "X", "*"])

    for run_dict in all_evals:
        training_langs = run_dict['training_langs']
        evals = run_dict['evals']
        xs = [ts for ts, _, _ in evals]
        ys = [test for _, _, test in evals]
        xs, ys = zip(*sorted(zip(xs, ys)))

        if len(training_langs) == 1 and training_langs[0] == configs['test_lang']:
            ax.plot(
                xs, ys,
                marker="o",
                color="black",
                linestyle="--",
                linewidth=2,
                alpha=0.4,
                label="Monolingual FT",
            )
        else:
            marker = next(non_mono_markers)
            ax.plot(
                xs, ys,
                marker=marker,
                linestyle="--",
                linewidth=2,
                alpha=0.7,
                label="1st-stage: " + ", ".join(training_langs),
            )

    ax.axhline(
        y=configs['mono_baseline'],
        color='gray',
        linestyle=':',
        linewidth=2,
        alpha=0.8,
        label='Monolingual Baseline',
    )

    ax.set_xticks([0, 200, 400, 600, 800])
    ax.set_xlabel("Target-Language FT Samples")
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=7, frameon=True, loc="lower right")


def plot_grid_figure(results: dict):
    """
    Create a single 2x3 figure:
      - row 1: no, ro, bg  (Within-family transfer)
      - row 2: vi, id, tr  (Distant transfer)
    `results` is a dict: lang -> (all_evals, config)
    """
    os.makedirs("plots", exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(10, 6), sharey=True)

    within_langs = ["no", "ro", "bg"]
    distant_langs = ["vi", "id", "tr"]

    # row 1: within-family
    for col, lang in enumerate(within_langs):
        ax = axes[0, col]
        all_evals, cfg = results[lang]
        plot_2_stage_ft_on_ax(ax, all_evals, cfg)
        if col == 0:
            ax.set_ylabel("Accuracy")

    # row 2: distant transfer
    for col, lang in enumerate(distant_langs):
        ax = axes[1, col]
        all_evals, cfg = results[lang]
        plot_2_stage_ft_on_ax(ax, all_evals, cfg)
        if col == 0:
            ax.set_ylabel("Accuracy")

    fig.text(0.5, 0.97, "Within-family transfer", ha="center", va="top", fontsize=12)
    fig.text(0.5, 0.50, "Distant transfer", ha="center", va="top", fontsize=12)

    fig.tight_layout(rect=[0.05, 0.05, 0.95, 0.93])
    plt.savefig(os.path.join("plots", "exp2_grid.png"), dpi=300)
    plt.close()

# def plot_2_stage_ft(all_evals: list, configs: dict):
#     """Plot training size vs dev/test accuracy for all runs."""
    
#     title = f"Target Language: {configs['test_lang']}"
#     model = "slm" if "slm" in configs['model_type'] else "clf"
#     out_path = os.path.join("plots", f"exp2_{configs['test_lang']}_{model}.png")
#     plt.figure()
    
#     for run_dict in all_evals:
#         training_langs = run_dict['training_langs']
#         evals = run_dict['evals']
#         xs = [ts for ts, _, _ in evals]
#         ys = [test for _, _, test in evals]
#         xs, ys = zip(*sorted(zip(xs, ys)))
#         if len(training_langs) == 1 and training_langs[0] == configs['test_lang']:
#             # Monolingual FT — fixed style
#             plt.plot(
#                 xs, ys,
#                 marker="o",
#                 color="black",      # always same color
#                 linestyle="--", 
#                 linewidth=2,
#                 alpha=0.4,
#                 label="Monolingual Fine-Tuning"
#             )
#         else:
#             # Multilingual or 2-stage runs — different style
#             marker = next(NON_MONO_MARKERS)
#             plt.plot(
#                 xs, ys,
#                 marker=marker,
#                 linestyle="--",
#                 linewidth=2,
#                 alpha=0.7,
#                 label="1st-stage Training: " + ", ".join(training_langs)
#             )
    
#     plt.axhline(
#         y=configs['mono_baseline'],
#         color='gray',
#         linestyle=':',
#         linewidth=2,
#         alpha=0.8,
#         label='Monolingual Baseline',
#     )
    
#     plt.xticks([0, 200, 400, 600, 800])
#     plt.title(title)
#     plt.xlabel("Target-Language Fine-Tuning Samples")
#     plt.ylabel("Accuracy")
#     plt.legend(
#         loc="upper center",
#         bbox_to_anchor=(0.5, -0.15),
#         ncol=2,
#         frameon=True, 
#     )
#     plt.grid(True, linestyle='--', alpha=0.5)
#     plt.tight_layout(rect=[0, 0.05, 1, 1])
#     plt.savefig(out_path, dpi=300)
#     plt.close()

def select_runs(configs: dict) -> list[str]:
    
    select_runs = []
    # find all lang folder
    all_langs = [
        f for f in os.listdir(EXP2_DIR)
        if os.path.isdir(os.path.join(EXP2_DIR, f))
        if f != "smoke_test"
    ]
    for lang in all_langs:
        
        if lang != configs["test_lang"]:
            continue

        # check if dir exists
        lang_dir = os.path.join(EXP2_DIR, f"{lang}")
        print(lang_dir)

        all_runs = [
            f for f in os.listdir(lang_dir)
            if os.path.isdir(os.path.join(lang_dir, f))
            if f != "smoke_test"
        ]


        for run in all_runs:
            # load meta_1
            run_dir = os.path.join(lang_dir, run)
            if not os.path.exists(run_dir):
                print(f"Directory {run_dir} does not exist. Skipping.")
                continue
            meta_1_path = os.path.join(run_dir, "meta_1.json")
            meta_1 = load_meta(meta_1_path)
            
            # Skip if not fitting configs
            if meta_1['model_type'] not in configs["model_type"]:
                continue
            if meta_1['training_langs'] not in configs["training_langs"]:
                continue
            if ("atl" in configs and meta_1['atl'] != configs['atl']):
                continue


            select_runs.append({"run_dir": run_dir,
                                "training_langs": meta_1['training_langs'],})
    return select_runs

# def main():
#     # configs = [{"test_lang": "bg",
#     #            "training_langs": [["en"], ["en", "ru"], ["ru"]],
#     #            "model_type": ["clf", "classifier", "cls"]},
#     #           {"test_lang": "no",
#     #            "training_langs": [["en"], ["en", "nl"], ["nl"]],
#     #            "model_type": ["clf", "classifier", "cls"]},
#     #            {"test_lang": "ro",
#     #            "training_langs": [["en"], ["en", "it"], ['it']],
#     #            "model_type": ["clf", "classifier", "cls"]},]

#     configs = [{"test_lang": "bg",
#                "training_langs": [["en"], ["en", "ru"], ["ru"]],
#                "model_type": ["slm"],
#                "atl": True},
#               {"test_lang": "no",
#                "training_langs": [["en"], ["en", "nl"], ["nl"]],
#                "model_type": ["slm"],
#                "atl": True},
#                {"test_lang": "ro",
#                "training_langs": [["en"], ["en", "it"], ['it']],
#                "model_type": ["slm"],
#                "atl": True},
#                {"test_lang": "vi",
#                "training_langs": [["en"]],
#                "model_type": ["slm"],
#                "atl": True},
#                {"test_lang": "id",
#                "training_langs": [["en"]],
#                "model_type": ["slm"],
#                "atl": True},
#                {"test_lang": "tr",
#                "training_langs": [["en"]],
#                "model_type": ["slm"],
#                "atl": True},]
    
#     for config in configs:
#         # collec monolignual metrics to get baselines
#         config = collect_best_monolingual_baseline(config)
#         # collect monoligual model over training sizes
#         mono_model = collect_monoliongual_metrics(config)
#         print(config)
#         selected_runs = select_runs(configs=config)
#         print("="*20)
#         print(f"Selected {len(selected_runs)} runs:")
#         for run_dict in selected_runs:
#             print(run_dict["run_dir"])
#         print("="*20)
#         all_evals = []
#         for run_dict in selected_runs:
#             evals = get_metas(run_dict=run_dict)
#             print(evals)
#             all_evals.append(evals)
#         print(all_evals)
#         all_evals.append(mono_model)
#         plot_2_stage_ft(all_evals=all_evals, configs=config)




def main():
    configs = [
        {"test_lang": "bg",
         "training_langs": [["en"], ["en", "ru"], ["ru"]],
         "model_type": ["slm"],
         "atl": True},
        {"test_lang": "no",
         "training_langs": [["en"], ["en", "nl"], ["nl"]],
         "model_type": ["slm"],
         "atl": True},
        {"test_lang": "ro",
         "training_langs": [["en"], ["en", "it"], ['it']],
         "model_type": ["slm"],
         "atl": True},
        {"test_lang": "vi",
         "training_langs": [["en"]],
         "model_type": ["slm"],
         "atl": True},
        {"test_lang": "id",
         "training_langs": [["en"]],
         "model_type": ["slm"],
         "atl": True},
        {"test_lang": "tr",
         "training_langs": [["en"]],
         "model_type": ["slm"],
         "atl": True},
    ]

    results = {}  # lang -> (all_evals, config)

    for config in configs:
        # collect monolingual baseline
        config = collect_best_monolingual_baseline(config)
        # collect monolingual model over training sizes
        mono_model = collect_monoliongual_metrics(config)

        selected_runs = select_runs(configs=config)
        print("=" * 20)
        print(f"Selected {len(selected_runs)} runs:")
        for run_dict in selected_runs:
            print(run_dict["run_dir"])
        print("=" * 20)

        all_evals = []
        for run_dict in selected_runs:
            evals = get_metas(run_dict=run_dict)
            all_evals.append(evals)

        all_evals.append(mono_model)

        # store for grid
        results[config["test_lang"]] = (all_evals, config)

        # individual plot for this language
        plot_2_stage_ft(all_evals=all_evals, configs=config)

    # single 2x3 figure
    plot_grid_figure(results)


if __name__ == "__main__":
    main()
