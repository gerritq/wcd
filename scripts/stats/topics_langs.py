import os
import json
import numpy as np
from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.spatial.distance import jensenshannon
from sklearn.manifold import MDS

BASE_DIR = os.getenv("BASE_WCD")
IN_DIR = os.path.join(BASE_DIR, "data/sents")
OUT_DIR = os.path.join(BASE_DIR, "data/out/tables")
os.makedirs(OUT_DIR, exist_ok=True)


def load_data(lang: str):
    path = os.path.join(IN_DIR, f"{lang}_sents.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for x in data:
        if x["source"] == "good":
            x["source"] = "fa"
    return data


def clean_topic(x):
    if not x:
        return None
    return x.split(".")[1].replace("*", "").strip().lower()


def build_topic_matrix(lang_data, langs):
    topic_set = set()
    per_lang_topics = {}

    for lang in langs:
        data = lang_data[lang]
        topics = [
            clean_topic(x["topic"])
            for x in data
            if x.get("topic") and x.get("source") == "fa"
        ]
        topics = [t for t in topics if t is not None]
        c = Counter(topics)
        per_lang_topics[lang] = c
        topic_set.update(c.keys())

    all_topics = sorted(topic_set)
    M = np.zeros((len(langs), len(all_topics)), float)

    for i, lang in enumerate(langs):
        c = per_lang_topics[lang]
        v = np.array([c.get(t, 0) for t in all_topics], float)
        if v.sum() > 0:
            v = v / v.sum()
        M[i] = v

    return M, all_topics


def js_distance_matrix(P):
    n = len(P)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = jensenshannon(P[i], P[j], base=2.0)
            D[i, j] = D[j, i] = d
    return D


def cosine_distance_matrix(P):
    norms = np.linalg.norm(P, axis=1) + 1e-12
    Pn = P / norms[:, None]
    sim = Pn @ Pn.T
    D = 1 - sim
    np.fill_diagonal(D, 0)
    return D


def hellinger_distance_matrix(P):
    n = len(P)
    sqrtP = np.sqrt(P)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(sqrtP[i] - sqrtP[j]) / np.sqrt(2)
            D[i, j] = D[j, i] = d
    return D


def plot_heatmap(D, langs, name):
    plt.figure(figsize=(8, 6))
    sns.heatmap(D, xticklabels=langs, yticklabels=langs,
                cmap="viridis", annot=True, fmt=".3f")
    plt.title(f"{name} Distance Between Languages")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{name.lower()}_heatmap.png"), dpi=300)
    plt.close()


def plot_embedding(D, langs, name):
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=0)
    coords = mds.fit_transform(D)

    plt.figure(figsize=(6, 5))
    plt.scatter(coords[:, 0], coords[:, 1])
    for (x, y), lang in zip(coords, langs):
        plt.text(x + 0.01, y + 0.01, lang, fontsize=10)
    plt.title(f"{name} Topic Embedding")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{name.lower()}_embedding.png"), dpi=300)
    plt.close()


def main():
    langs = ["en", "nl", "no", "it", "pt", "ro", "ru", "uk", "bg", "vi", "id", "tr"]
    lang_data = {lang: load_data(lang) for lang in langs}

    topic_matrix, _ = build_topic_matrix(lang_data, langs)

    metrics = [
        ("Jensen–Shannon", js_distance_matrix(topic_matrix)),
        ("Cosine", cosine_distance_matrix(topic_matrix)),
        ("Hellinger", hellinger_distance_matrix(topic_matrix)),
    ]

    for name, D in metrics:
        plot_heatmap(D, langs, name)
        plot_embedding(D, langs, name)


if __name__ == "__main__":
    main()