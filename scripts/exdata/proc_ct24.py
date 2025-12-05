import os
import csv
import json
from collections import defaultdict

BASE_DIR = "/scratch/prj/inf_nlg_ai_detection/wcd/data/raw/ct24"

languages = ["arabic", "dutch", "english"]

LANG_MAP = {"arabic": "ar",
            "dutch": "nl",
            "english": "en"
            }

split_map = {
    "train": "train",
    "dev": "dev",
    "test_gold": "test"
}

def read_tsv(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))

def main():
    for lang in languages:
        text_key = "tweet_text" if lang != "english" else "Text"
        counts = defaultdict(lambda: defaultdict(int))

        output_path = os.path.join(BASE_DIR, f"ct24_{LANG_MAP[lang]}.jsonl")

        with open(output_path, "w", encoding="utf-8") as out:
            for suf, split_name in split_map.items():
                tsv_path = os.path.join(BASE_DIR, f"CT24_checkworthy_{lang}_{suf}.tsv")
                print(tsv_path)
                if not os.path.exists(tsv_path):
                    print(f"[WARN] Missing file: {tsv_path}")
                    continue

                rows = read_tsv(tsv_path)

                for row in rows:
                    if row['class_label'] == "No":
                        counts[split_name][0] += 1
                    if row['class_label'] == "Yes":
                        counts[split_name][1] += 1  
                    obj = {
                        "claim": row[text_key],
                        "label": 1 if row['class_label'].lower() == "yes" else 0,
                        "split": split_name,
                        "lang": LANG_MAP[lang]
                    }
                    out.write(json.dumps(obj, ensure_ascii=False) + "\n")

        print("Lang", lang)
        print(counts)

if __name__ == "__main__":
    main()