import os
import json
from collections import Counter

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import lang2vec.lang2vec.lang2vec as l2v

plt.rcParams.update({"font.size": 22})
sns.set(style="white")

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
BASE_DIR = os.getenv("BASE_WCD")
DATA_DIR = os.path.join(BASE_DIR, "data/languages")
IN_DIR = os.path.join(BASE_DIR, "data/sents")
OUT_DIR = os.path.join(BASE_DIR, "scripts/tables/plots")
os.makedirs(DATA_DIR, exist_ok=True)

# ---------------------------------------------------------------------
# Languages
# ---------------------------------------------------------------------
# Full language list (ISO 639-3 codes)
languages = [
    "eng",  # English
    "por",  # Portuguese
    "deu",  # German
    "rus",  # Russian
    "ita",  # Italian
    "vie",  # Vietnamese
    "tur",  # Turkish
    "nld",  # Dutch
    "ukr",  # Ukrainian
    "ron",  # Romanian
    "ind",  # Indonesian
    "bul",  # Bulgarian
    "uzb",  # Uzbek
    "nob",  # Norwegian Bokmål
    "aze",  # Azerbaijani
    "mkd",  # Macedonian
    "hye",  # Armenian
    "sqi",  # Albanian
]

# Display / ISO 639-1 codes
languages_display = [
    "en",
    "pt",
    "de",
    "ru",
    "it",
    "vi",
    "tr",
    "nl",
    "uk",
    "ro",
    "id",
    "bg",
    "uz",
    "no",
    "az",
    "mk",
    "hy",
    "sq",
]
# languages = [
#     "eng",  # English
#     "nld",  # Dutch
#     "nob",  # Norwegian Bokmål
#     "ita",  # Italian
#     "por",  # Portuguese
#     "ron",  # Romanian
#     "rus",  # Russian
#     "ukr",  # Ukrainian
#     "bul",  # Bulgarian
#     "ind",  # Indonesian
#     "vie",  # Vietnamese
#     "tur",  # Turkish
# ]

# languages_display = [
#     "en",
#     "nl",
#     "no",
#     "it",
#     "pt",
#     "ro",
#     "ru",
#     "uk",
#     "bg",
#     "id",
#     "vi",
#     "tr",
# ]

# Use the 2-letter codes for all topic-based stuff
topic_langs = languages_display

topics_display = {"regions": "Regions",
                  "media": "Media",
                  "stem": "STEM",
                  "military_and_warfare": "Military/Warfare",
                  "biography": "Biography",
                  "sports": "Sports",
                  "history": "History",
                  "philosophy_and_religion": "Philosophy/Religion",
                  "visual_arts": "Visual Arts",
                  "transportation": "Transportation"}

# ---------------------------------------------------------------------
# E-linguistics and lang2vec distances
# ---------------------------------------------------------------------
# def get_l2v_distances(distance_type: str) -> pd.DataFrame:
#     data = l2v.distance(distance_type, languages)
#     df = pd.DataFrame(data, index=languages_display, columns=languages_display)
#     df = df * 100.0
#     return df


def load_elinguistics() -> pd.DataFrame:
    df = pd.read_excel(os.path.join(DATA_DIR, "elinguistics.xlsx"), index_col=0)
    # Reorder rows/cols to match our language order
    df = df.loc[languages_display, languages_display]
    return df


# ---------------------------------------------------------------------
# Topic data and similarity
# ---------------------------------------------------------------------
def load_topic_data(lang: str):
    path = os.path.join(IN_DIR, f"{lang}_sents.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for x in data:
        if x.get("source") == "good":
            x["source"] = "fa"
    return data


def clean_topic(x):
    if not x:
        return None
    parts = x.split(".")
    if len(parts) > 1:
        x = parts[1]
    return x.replace("*", "").strip().lower()


def build_topic_matrix(lang_data, langs):
    topic_set = set()
    per_lang_topics = {}

    for lang in langs:
        data = lang_data[lang]
        topics = [
            clean_topic(x.get("topic"))
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

    return M, all_topics, per_lang_topics


def cosine_similarity_matrix(P: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(P, axis=1, keepdims=True) + 1e-12
    Pn = P / norms
    sim = Pn @ Pn.T
    np.fill_diagonal(sim, 1.0)
    return sim


# ---------------------------------------------------------------------
# Main figure with 4 subplots
# ---------------------------------------------------------------------
def plot_combined(
    df_eling: pd.DataFrame,
    df_l2v: pd.DataFrame,
    topic_counts: Counter,
    cos_sim: np.ndarray,
    langs: list[str],
):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), dpi=200)

    # # 1) E-linguistics heatmap (lower triangle)
    # mask_eling = np.triu(np.ones(df_eling.shape, dtype=bool), k=1)
    # sns.heatmap(
    #     df_eling,
    #     mask=mask_eling,
    #     ax=axes[0],
    #     square=False,
    #     vmin=0,
    #     vmax=100,
    #     annot=True, 
    #     annot_kws={"size": 12},
    #     cbar=False,
    # )
    # axes[0].set_title("(a) eLinguistics.net (Genetic Distance [0,1])")
    # axes[0].set_xticklabels(langs)
    # axes[0].set_yticklabels(langs, rotation=0)

    # 2) lang2vec heatmap (lower triangle)
    mask_l2v = np.triu(np.ones(df_l2v.shape, dtype=bool), k=1)
    sns.heatmap(
        df_l2v,
        mask=mask_l2v,
        ax=axes[0],
        square=False,
        cbar=False,
        vmin=0,
        vmax=100,
        annot=True, 
        annot_kws={"size": 12}
    )
    axes[0].set_title("(b) lang2vec (Syntactic Distance [0,1])")
    axes[0].set_xticklabels(langs)
    axes[0].set_yticklabels(langs, rotation=0)

    # 3) Cosine similarity matrix
    mask_cos = np.triu(np.ones(cos_sim.shape, dtype=bool), k=1)

    # Replace your cosine similarity heatmap call with this:
    sns.heatmap(
        cos_sim,
        mask=mask_cos,
        ax=axes[1],
        square=False,
        cbar=False,
        vmin=0.0,
        vmax=1.0,
        cmap="viridis_r",
        annot=True, 
        annot_kws={"size": 8}
    )
    axes[1].set_title("(c) Topic Similarity (Cosine Similarity [-1,1])")
    axes[1].set_xticklabels(langs)
    axes[1].set_yticklabels(langs, rotation=0)

    # 4) Top-10 topics across languages
    top_topics = topic_counts.most_common(10)
    topics, counts = zip(*top_topics)
    axes[32].barh(range(len(topics)), counts)
    axes[2].set_yticks(range(len(topics)))
    axes[2].set_yticklabels(topics)
    axes[2].invert_yaxis()
    axes[2].set_title("(d) Top-10 Topics (All Languages)")
    axes[2].set_xlabel("Frequency")
    axes[2].set_yticklabels([topics_display.get(t[0], t) for t in top_topics])



    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "desc_plot.pdf")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved combined figure to {out_path}")


def main():
    # # 1) Distances from external resources
    # df_l2v = get_l2v_distances("syntactic")
    df_eling = load_elinguistics()

    # 2) Topic-based data
    lang_data = {lang: load_topic_data(lang) for lang in topic_langs}
    topic_matrix, all_topics, per_lang_topics = build_topic_matrix(lang_data, topic_langs)

    # Global top-10 topics
    global_counts = Counter()
    for c in per_lang_topics.values():
        global_counts.update(c)

    # Cosine similarity matrix across languages
    cos_sim = cosine_similarity_matrix(topic_matrix)

    # 3) Final combined plot
    plot_combined(
        df_eling=df_eling,
        df_l2v=None,
        topic_counts=global_counts,
        cos_sim=cos_sim,
        langs=topic_langs,
    )


if __name__ == "__main__":
    main()