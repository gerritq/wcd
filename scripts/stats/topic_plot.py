import os
import json
import argparse
from collections import Counter
import matplotlib.pyplot as plt

# DIRs
BASE_DIR = os.getenv("BASE_WCD")
IN_DIR = os.path.join(BASE_DIR, "data/sents")
OUT_DIR = os.path.join(BASE_DIR, "data/out/topics")
os.makedirs(OUT_DIR, exist_ok=True)

# ARGPARSE
parser = argparse.ArgumentParser()
parser.add_argument("--lang", type=str, required=True)
args = parser.parse_args()

def load_data(lang: str):
    path = os.path.join(IN_DIR, f"{lang}_sents.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    rows = load_data(args.lang)

    counts = {
        "fa": {0: Counter(), 1: Counter()},
        "random": {0: Counter(), 1: Counter()},
    }

    for x in rows:
        topic = x.get("topic")
        if not topic:
            continue
        topic = topic.replace("*", "")
        src = x.get("source")
        lbl = int(x.get("label", 0))
        counts[src][lbl][topic] += 1

    # fa vs random, lables combined
    fa_counts = Counter()
    rand_counts = Counter()
    for lbl in [0, 1]:
        fa_counts += counts["fa"][lbl]
        rand_counts += counts["random"][lbl]

    # top 10 topics overall
    all_topics = (fa_counts + rand_counts).most_common(10)
    topics = [t for t, _ in all_topics]
    fa_vals = [fa_counts[t] for t in topics]
    rand_vals = [rand_counts[t] for t in topics]

    plt.figure(figsize=(10, 6))
    x = range(len(topics))
    width = 0.35
    plt.bar([i - width/2 for i in x], fa_vals, width, label="Featured Articles", color="royalblue")
    plt.bar([i + width/2 for i in x], rand_vals, width, label="Random Articles", color="lightcoral")
    plt.xticks(x, topics, rotation=45, ha="right")
    plt.ylabel("Number of Claims")
    plt.title(f"Claim Distribution by Topic — {args.lang.upper()} (FA vs Random)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{args.lang}_fa_vs_random.pdf"), dpi=300, bbox_inches="tight")
    plt.show()
    # ==================================================================

    # ========= ORIGINAL: Per-source plots, split by label =========
    for src in ["fa", "random"]:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"Top 10 topics — source: {src.upper()}")

        for i, lbl in enumerate([0, 1]):
            top_topics = counts[src][lbl].most_common(10)
            if not top_topics:
                axes[i].set_title(f"Label {lbl} — No data")
                axes[i].axis("off")
                continue

            topics_s, vals_s = zip(*top_topics)
            axes[i].bar(topics_s, vals_s, color="skyblue")
            axes[i].set_xticklabels(topics_s, rotation=45, ha="right")
            axes[i].set_title(f"Label {lbl}")
            axes[i].set_ylabel("Count")

        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"{args.lang}_{src}.pdf"), dpi=300, bbox_inches="tight")
        plt.show()
    # ==================================================================

if __name__ == "__main__":
    main()