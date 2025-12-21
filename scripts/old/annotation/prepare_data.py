import os
import pandas as pd
import requests
from tqdm import tqdm
from datasets import Dataset, DatasetDict, load_from_disk
from typing import Tuple, List

BASE_DIR = os.getenv("BASE_WCD")
SETS_DIR = os.path.join(BASE_DIR, "data/sets/main")
ANNOTATION_DIR = os.path.join(BASE_DIR, "data/annotation")
os.makedirs(ANNOTATION_DIR, exist_ok=True)

def load_data(lang: str) -> DatasetDict:
    """
    Load data and assign a varibal of the respective set.
    """
    path = os.path.join(SETS_DIR, lang)
    return load_from_disk(path)

def prepare_df(df: pd.DataFrame, lang: str) -> pd.DataFrame:
    """
    Construct final df: add wikipedia link, rename, and add metadata.
    """
    df = df[['title', 'section', 'label', 'claim']].copy()
    df['wikipedia_link'] = f"https://{lang}.wikipedia.org/wiki/" + df['title'].str.replace(" ", "_")

    df = df.rename(columns={'title': 'wikipedia_article',
                            'section': 'wikipedia_section'})
    return df[['wikipedia_article', 'wikipedia_section', 'wikipedia_link', 'label', 'claim']]

def pilot_data(lang: str) -> None:
    """
    Generates a pilot data df and saves it.
    """

    ds = load_data(lang)
    df = pd.DataFrame(ds['train'])
    
    # 50 instances each
    df_1 = df[df["label"] == 1].head(25)
    df_0 = df[df["label"] == 0].head(25)

    df = pd.concat([df_1, df_0]).reset_index(drop=True)
    df = prepare_df(df, lang)
    out_path = os.path.join(ANNOTATION_DIR, f"{lang}_pilot.csv")
    df.to_csv(out_path, index=False, encoding="utf-8")

def main_data(lang: str) -> None:
    
    ds = load_data(lang)
    dfs = []
    for split in ["train", "dev", "test"]:
        if split in ds:
            df = pd.DataFrame(ds[split])
            df["split"] = split
            dfs.append(df)
    df_all = pd.concat(dfs, ignore_index=True)
    df = prepare_df(df, lang)
    out_path = os.path.join(ANNOTATION_DIR, f"{lang}.csv")
    df_all.to_csv(out_path, index=False, encoding="utf-8")
        
if __name__ == "__main__":
    languages = ["en"]
    for lang in languages:
        pilot_data(lang)
        main_data(lang)