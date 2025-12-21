import os
import re
import json
import matplotlib.pyplot as plt
import argparse

BASE_DIR = os.getenv("BASE_WCD")
EXP2_DIR = os.path.join(BASE_DIR, "data/exp2")

def load_metrics(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_losses(meta_dir: str, meta_number: str):

    base_dir = os.path.dirname(meta_dir)
    out_path = os.path.join(base_dir, f"meta_{meta_number}_loss.pdf")
    # if os.path.isfile(out_path):
    #     print(f"Loss plot for meta {meta_number} already exists")
    #     return
    
    with open(meta_dir, "r") as f:
        log = json.load(f)

    try:
        ts_og = log['training_size']
        ts_2 = log['training_data_n']
        lang = log['lang']
        explanation = log['explanation']
    except:
        ts_og = log['epoch_1']['training_size']
        ts_2 = log['epoch_1']['training_data_n']
        lang = log['epoch_1']['lang']
        explanation = log['epoch_1']['explanation']

    train_epochs, train_losses = [], []
    dev_epochs, dev_losses = [], []

    for entry in log.get("log_history", []):
        epoch = entry.get("epoch")
        if epoch is None:
            continue

        if "loss" in entry:
            train_epochs.append(epoch)
            train_losses.append(entry["loss"])

        if "eval_loss" in entry:
            dev_epochs.append(epoch)
            dev_losses.append(entry["eval_loss"])


    plt.figure()

    if train_epochs:
        plt.plot(train_epochs, train_losses, marker="o", label="train loss")
    if dev_epochs:
        plt.plot(dev_epochs, dev_losses, marker="s", linestyle="--", label="dev loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(f"L: {lang} - TS: {ts_og}/{ts_2} - EX: {explanation}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    
    all_folders = [
                   os.path.join(EXP2_DIR, f) for f in os.listdir(EXP2_DIR)
                   if os.path.isdir(os.path.join(EXP2_DIR, f))
                  ]

    for language_dir in all_folders:
        all_runs =  [
                    os.path.join(language_dir, f) for f in os.listdir(language_dir)
                    if os.path.isdir(os.path.join(language_dir, f))
                    ]
        for run_dir in all_runs:
            print(run_dir)
            all_metas = [
                os.path.join(run_dir, f)
                for f in os.listdir(run_dir)
                if os.path.isfile(os.path.join(run_dir, f)) and f.endswith(".json")
            ]
            for meta_dir in all_metas:
                filename = os.path.basename(meta_dir)
                meta_number = filename.split("_")[-1].replace(".json", "")
                print(filename, "--", meta_number)
                get_losses(meta_dir, meta_number)
        
if __name__ == "__main__":
    main()