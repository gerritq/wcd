import os
import csv
from datasets import Dataset, DatasetDict
import re

BASE_DIR = os.getenv("BASE_WCD")
DATA_DIR = os.path.join(BASE_DIR, "data/raw/check_that")
OUT_DIR  = os.path.join(BASE_DIR, "data/sets")

os.makedirs(OUT_DIR, exist_ok=True)

def clean_tweet(text: str) -> str:
    # remove any http/https links
    return re.sub(r"http\S+", "", text).strip()

def load_tsv(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            data.append({
                "claim": clean_tweet(row["tweet_text"]),
                "label": int(row["class_label"])
            })
    return data

def main():
    languages = ["bulgarian", "english", "dutch"]
    split_map = {"train": "train", "dev": "validation", "dev_test": "test"}

    for language in languages:
        split_data = {}
        for src_split, hf_split in split_map.items():
            tsv_path = os.path.join(DATA_DIR, f"CT22_{language}_1B_claim_{src_split}.tsv")
            if not os.path.exists(tsv_path):
                print(f"[WARN] Missing file: {tsv_path} — skipping this split")
                continue
            rows = load_tsv(tsv_path)
            split_data[hf_split] = Dataset.from_list(rows)
            print(f"[{language}] {hf_split}: {len(rows)} examples")

        if not split_data:
            print(f"[WARN] No splits found for {language}, skipping save")
            continue

        ds = DatasetDict(split_data)
        out_path = os.path.join(OUT_DIR, f"ct_{language}")
        ds.save_to_disk(out_path)
        print(f"[OK] Saved {language} dataset to {out_path}")

if __name__ == "__main__":
    main()