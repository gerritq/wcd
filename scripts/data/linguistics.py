import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import lang2vec.lang2vec as l2v

BASE_DIR = os.getenv("BASE_WCD")
DATA_DIR = os.path.join(BASE_DIR, "data/languages")


def eLinguistic():

    df = pd.read_excel(os.path.join(DATA_DIR, "elinguistics.xlsx"), index_col=0)
    mask = np.triu(np.ones_like(df, dtype=bool))
    sns.heatmap(df, mask=mask, annot=True, cmap="viridis", square=True)
    plt.figure(figsize=(8,6))
    sns.heatmap(
        df,
        annot=True,
        cmap="rocket_r",
        annot_kws={"size": 14},
        square=True, 
        cbar=True
    )
    plt.tick_params(axis="x", labelsize=14)
    plt.tick_params(axis="y", labelsize=14)
    plt.savefig(os.path.join(DATA_DIR, "elinguistics.pdf"), dpi=300, bbox_inches="tight")
    plt.close()

def run_lang2_vev():
    langs = [
    "eng",  # English
    "nld",  # Dutch
    "nor",  # Norwegian
    "ita",  # Italian
    "por",  # Portuguese
    "ron",  # Romanian
    "rus",  # Russian
    "ukr",  # Ukrainian
    "bul",  # Bulgarian
    "zho",  # Chinese
    "ara",  # Arabic
    "ind"   # Indonesian
    ]
    print(l2v)
    print(l2v.distance(['syntactic','geographic'], langs))

def main():
    # eLinguistic()
    run_lang2_vev()

if __name__ == "__main__":
    main()