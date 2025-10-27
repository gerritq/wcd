import lang2vec.lang2vec.lang2vec as l2v
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.getenv("BASE_WCD")
DATA_DIR = os.path.join(BASE_DIR, "data/languages")

languages = [
    "eng",  # English
    "nld",  # Dutch
    "nob",  # Norwegian Bokmål
    "ita",  # Italian
    "por",  # Portuguese
    "ron",  # Romanian
    "rus",  # Russian
    "ukr",  # Ukrainian
    "bul",  # Bulgarian
    "ind",  # Indonesian
    "vie",  # Vietnamese
    "tur",  # Turkish
]

languages_display  = [
    "en",  # English
    "nl",  # Dutch
    "no",  # Norwegian
    "it",  # Italian
    "pt",  # Portuguese
    "ro",  # Romanian
    "ru",  # Russian
    "uk",  # Ukrainian
    "bg",  # Bulgarian
    "id",
    "vi",
    "tr"
]

# print(l2v.available_languages())

def get_distances(distance_type, languages):
    data = l2v.distance(distance_type, languages)
    df = pd.DataFrame(data, columns=languages_display, index=languages_display)
    df = df*100
    return df


def eLinguistic():
    df = pd.read_excel(os.path.join(DATA_DIR, "elinguistics.xlsx"), index_col=0)
    return df


def create_plots(elinguistics_data, l2v_data):
    """Plot two lower-triangle heatmaps side by side for comparison."""

    mask_elinguistics  = np.triu(np.ones(elinguistics_data.shape, dtype=bool), k=1)
    mask_l2v = np.triu(np.ones(l2v_data.shape, dtype=bool), k=1)

    plt.figure(figsize=(8,6))
    sns.heatmap(
        l2v_data,
        mask=mask_l2v,
        annot=True,
        cmap="viridis",
        square=True,
        cbar=True,
        vmin=0, 
        vmax=100,
        annot_kws={"size": 12}
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tick_params(axis='both', labelsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "l2v.pdf"), dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8,6))
    sns.heatmap(
        elinguistics_data,
        mask=mask_elinguistics,
        annot=True,
        cmap="viridis",
        square=True,
        cbar=False,
        annot_kws={"size": 12}
    )
    
    plt.tick_params(axis='both', labelsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "elinguistics.pdf"), dpi=300, bbox_inches="tight")
    plt.close()


def main():
    df_l2v = get_distances("syntactic", languages)
    df_elinguistics = eLinguistic()

    create_plots(df_elinguistics, df_l2v)

if __name__ == "__main__":
    main()
