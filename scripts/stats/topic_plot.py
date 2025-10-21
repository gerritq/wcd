import os
import json
import argparse
from collections import Counter
import matplotlib.pyplot as plt

# DIRs
BASE_DIR = os.getenv("BASE_WCD")
IN_DIR = os.path.join(BASE_DIR, "data/sents")
OUT_DIR = os.path.join(BASE_DIR, "data/out")

# ARGPARSE
parser = argparse.ArgumentParser()
parser.add_argument("--lang", type=str, required=True)
args = parser.parse_args()

def load_data(lang: str):
    path = os.path.join(IN_DIR, f"{lang}_sents.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    data = load_data(args.lang)

    # unique titles (avoid duplicates)
    seen = set()
    rows = []
    for x in data:
        key = (x["title"], x["source"], x["topic"], x["label"])
        if key not in seen:
            seen.add(key)
            rows.append(x)

    # counters: topic -> count per source and label
    counts = {
        "fa": {0: Counter(), 1: Counter()},
        "random": {0: Counter(), 1: Counter()},
    }

    for x in rows:
        topic = x["topic"].split(".")[0]  # top-level category
        src = x["source"]
        lbl = int(x["label"])
        counts[src][lbl][topic] += 1

    # plotting
    for src in ["fa", "random"]:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"Top 10 topics — source: {src.upper()}")

        for i, lbl in enumerate([0, 1]):
            top_topics = counts[src][lbl].most_common(10)
            if not top_topics:
                axes[i].set_title(f"Label {lbl} — No data")
                continue

            topics, vals = zip(*top_topics)
            axes[i].barh(topics, vals, color="skyblue")
            axes[i].invert_yaxis()
            axes[i].set_title(f"Label {lbl}")
            axes[i].set_xlabel("Count")

        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"{src}.pdf"), dpi=300, bbox_inches="tight")
        plt.show()

if __name__ == "__main__":
    main()