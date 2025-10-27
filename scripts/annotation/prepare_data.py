import os
import pandas as pd
from datasets import load_from_disk

BASE_DIR = os.getenv("BASE_WCD")
SETS_DIR = os.path.join(BASE_DIR, "data/sets/main")
ANNOTATION_DIR = os.path.join(BASE_DIR, "data/annotation/raw")
os.makedirs(ANNOTATION_DIR, exist_ok=True)

languages = ["en", "pt"]

def load_data(lang: str) -> list:
    path = os.path.join(SETS_DIR, f"{lang}")
    return load_from_disk(path)

def pilot_data():

    for lang in languages:
        ds = load_data(lang)
        dfs = []
        for split in ["train", "validation", "test"]:
            if split in ds:
                df = pd.DataFrame(ds[split])
                df["split"] = split
                dfs.append(df)
        df_all = pd.concat(dfs, ignore_index=True)
        
        # 50 instances each
        df_1 = df[df["label"] == 1].head(50)
        df_0 = df[df["label"] == 0].head(50)

        df_sampled = pd.concat([df_1, df_0]).sample(frac=1, random_state=42).reset_index(drop=True)
        out_path = os.path.join(ANNOTATION_DIR, f"{lang}.csv")
        df_all.to_csv(out_path, index=False, encoding="utf-8")

def main_data():
    for lang in languages:
        ds = load_data(lang)
        dfs = []
        for split in ["train", "validation", "test"]:
            if split in ds:
                df = pd.DataFrame(ds[split])
                df["split"] = split
                dfs.append(df)
        df_all = pd.concat(dfs, ignore_index=True)
        out_path = os.path.join(ANNOTATION_DIR, f"{lang}.csv")
        df_all.to_csv(out_path, index=False, encoding="utf-8")

if __name__ == "__main__":
    main()