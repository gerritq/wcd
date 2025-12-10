import os
from datasets import load_from_disk

import os

DATASET_DIR = "/scratch/prj/inf_nlg_ai_detection/wcd/data/sets/main"

all_langs = [
    d for d in os.listdir(DATASET_DIR)
    if os.path.isdir(os.path.join(DATASET_DIR, d))
]

print(all_langs)
for lang in all_langs:
    print(f"\nLanguage: {lang}")
    lang_path = os.path.join(DATASET_DIR, lang)

    if not os.path.exists(lang_path):
        print(f"Skipping {lang} — folder does not exist.")
        continue

    # load the dataset for this language
    ds = load_from_disk(lang_path)

    # check each split
    for split in ["train", "dev", "test"]:
        if split not in ds:
            continue

        split_ds = ds[split]

        unique_titles = set()
        for x in split_ds:
            unique_titles.add(x["title"])

        print(f"  {split}: {len(unique_titles)} unique titles")