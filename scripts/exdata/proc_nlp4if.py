import os
import csv
import json
from collections import defaultdict

BASE_DIR = "/scratch/prj/inf_nlg_ai_detection/wcd/data/raw/nlp4if"

languages = ["arabic", "bulgarian", "english"]

LANG_MAP = {"arabic": "ar",
            "bulgarian": "bg",
            "english": "en"
            }

split_map = {
    "train": "train",
    "dev": "dev",
    "test_gold": "test"
}

HEADERS = ["tweet_no", "tweet_text", "q1_label", "q2_label", "q3_label", "q4_label", "q5_label", "q6_label", "q7_label"]

def read_tsv(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t", fieldnames=HEADERS))

def main():
    for lang in languages:

        counts = defaultdict(lambda: defaultdict(int))
        output_path = os.path.join(BASE_DIR, f"nlp4if_{LANG_MAP[lang]}.jsonl")

        with open(output_path, "w", encoding="utf-8") as out:
            for suf, split_name in split_map.items():
                tsv_path = os.path.join(BASE_DIR, f"covid19_disinfo_binary_{lang}_{suf}.tsv")
                
                print(tsv_path)
                if not os.path.exists(tsv_path):
                    print(f"[WARN] Missing file: {tsv_path}")
                    continue

                rows = read_tsv(tsv_path)

                for row in rows[1:]: # skip header
                    if row['q1_label'].lower() == "nan":
                        continue
                    if row['q1_label'].lower() == "no":
                        counts[split_name][0] += 1
                    if row['q1_label'].lower() == "yes":
                        counts[split_name][1] += 1  
                    label = 1 if row['q1_label'].lower() == "yes" else 0
                    obj = {
                        "claim": row["tweet_text"],
                        "label": int(label),
                        "split": split_name,
                        "lang": LANG_MAP[lang]
                    }
                    out.write(json.dumps(obj, ensure_ascii=False) + "\n")

        print("Lang", lang)
        print(counts)

if __name__ == "__main__":
    main()