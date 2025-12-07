import os
import json
import argparse
import numpy as np
from collections import Counter
from scipy.spatial.distance import jensenshannon
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

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


def clean_topic(x: str | None) -> str | None:
    if not x:
        return None
    return x.split(".")[1].replace("*", "").strip().lower()


def js_distance_fa_between_labels(data_lang1: list[dict], data_lang2: list[dict]) -> float:
    topics1 = Counter(
        clean_topic(x["topic"])
        for x in data_lang1
        if x.get("topic") and x.get("source") == "fa"
    )
    topics2 = Counter(
        clean_topic(x["topic"])
        for x in data_lang2
        if x.get("topic") and x.get("source") == "fa"
    )

    all_topics = sorted(set(topics1) | set(topics2))
    p = np.array([topics1.get(t, 0) for t in all_topics], dtype=float)
    q = np.array([topics2.get(t, 0) for t in all_topics], dtype=float)

    eps = 1e-10
    p = (p + eps) / (p.sum() + eps * len(all_topics))
    q = (q + eps) / (q.sum() + eps * len(all_topics))

    return jensenshannon(p, q, base=2.0)


def compute_js_distance_for_all_languages(langs: list[str]) -> dict[tuple[str, str], float]:
    lang_data = {lang: load_data(lang) for lang in langs}
    js_distances = {}
    for i, lang1 in enumerate(langs):
        for j, lang2 in enumerate(langs):
            if i < j:
                dist = js_distance_fa_between_labels(lang_data[lang1], lang_data[lang2])
                js_distances[(lang1, lang2)] = dist
    return js_distances


def create_matrix_table(js_distances: dict[tuple[str, str], float], langs: list[str]) -> str:
    header = "\t" + "\t".join(langs)
    rows = [header]
    for lang1 in langs:
        row = [lang1]
        for lang2 in langs:
            if lang1 == lang2:
                row.append("0.000")
            else:
                key = (lang1, lang2) if (lang1, lang2) in js_distances else (lang2, lang1)
                dist = js_distances[key]
                row.append(f"{dist:.3f}")
        rows.append("\t".join(row))
    return "\n".join(rows)


def create_js_heatmap(js_distances: dict[tuple[str, str], float], langs: list[str]) -> np.ndarray:
    size = len(langs)
    matrix = np.zeros((size, size), dtype=float)
    for i, lang1 in enumerate(langs):
        for j, lang2 in enumerate(langs):
            if lang1 == lang2:
                matrix[i, j] = 0.0
            else:
                key = (lang1, lang2) if (lang1, lang2) in js_distances else (lang2, lang1)
                matrix[i, j] = js_distances[key]

    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, xticklabels=langs, yticklabels=langs, cmap="viridis", annot=True, fmt=".3f")
    plt.title("Jensen–Shannon Distance Between Languages")
    plt.xlabel("Language")
    plt.ylabel("Language")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "cm_topics.png"), dpi=300)
    plt.close()

    return matrix


def build_topic_matrix(lang_data: dict[str, list[dict]], langs: list[str]) -> tuple[np.ndarray, list[str]]:
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
    matrix = np.zeros((len(langs), len(all_topics)), dtype=float)

    for i, lang in enumerate(langs):
        c = per_lang_topics[lang]
        row = np.array([c.get(t, 0) for t in all_topics], dtype=float)
        if row.sum() > 0:
            row = row / row.sum()
        matrix[i] = row

    return matrix, all_topics


def plot_language_embedding(lang_matrix: np.ndarray, langs: list[str]):
    pca = PCA(n_components=2, random_state=0)
    coords = pca.fit_transform(lang_matrix)

    plt.figure(figsize=(6, 5))
    for (x, y), lang in zip(coords, langs):
        plt.scatter(x, y)
        plt.text(x + 0.01, y + 0.01, lang, fontsize=10)

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Language Embedding Based on Topic Distributions")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "language_topic_pca.png"), dpi=300)
    plt.close()


def main():
    langs = ['en', 'nl', 'no', 'it', 'pt', 'ro', 'ru', 'uk', 'bg', 'vi', 'id', 'tr']

    # js_distances = compute_js_distance_for_all_languages(langs)
    # table_str = create_matrix_table(js_distances, langs)
    # print(table_str)

    # matrix = create_js_heatmap(js_distances, langs)
    # np.savetxt(os.path.join(OUT_DIR, "cm_topics.txt"), matrix, fmt="%.6f", delimiter="\t")

    lang_data = {lang: load_data(lang) for lang in langs}
    topic_matrix, _ = build_topic_matrix(lang_data, langs)
    plot_language_embedding(topic_matrix, langs)


if __name__ == "__main__":
    main()