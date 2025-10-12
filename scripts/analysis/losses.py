import os
import json
from collections import defaultdict
import matplotlib.pyplot as plt

LANG = "en"
MODEL = "model_1"   

BASE_DIR = os.getenv("BASE_WCD")
META_PATH = os.path.join(BASE_DIR, f"data/models/{MODEL}/", "meta.json")
OUT_PATH = os.path.join(BASE_DIR, f"data/out/{LANG}_{MODEL}_loss.pdf")

def main():

    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    train_logs = meta["train_losses"]
    train_logs = sorted(meta["train_losses"], key=lambda x: x["epoch"])

    epochs = [rec["epoch"] for rec in train_logs]
    losses = [rec["loss"] for rec in train_logs]

    plt.plot(epochs, losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Training loss")
    plt.title(f"Training loss over epochs — {LANG} - {MODEL}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=200)
    print(f"Saved plot to: {OUT_PATH}")

if __name__ == "__main__":
    main()