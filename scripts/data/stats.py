import os
import json
from collections import defaultdict

BASE_DIR = os.getenv("BASE_WCD")
INPUT_DIR = os.path.join(BASE_DIR, "data/raw")

def main():
    languages  = [
        # "en",  # English
        "nl",  # Dutch
        "no",  # Norwegian (Bokmål is 'nb', Nynorsk is 'nn', 'no' redirects to Bokmål)
        "it",  # Italian
        "pt",  # Portuguese
        "ro",  # Romanian
        "ru",  # Russian
        "uk",  # Ukrainian
        "bg",  # Bulgarian
        "zh",  # Chinese
        "ar",  # Arabic
        "id"   # Indonesian
    ]

    for lang in languages:
        count = defaultdict(int)
        file_path = os.path.join(INPUT_DIR, f"{lang}_all.jsonl")

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                count[item['source']] += 1
        
        sorted_count = dict(sorted(count.items()))
        print(f"========== {lang} ==========")
        print(sorted_count, "\n")

if __name__ == "__main__":
    main()


# for fname in sorted(os.listdir(BASE_DIR)):
#     fpath = os.path.join(BASE_DIR, fname)
#     if os.path.isfile(fpath) and fname.endswith(".jsonl"):
#         counts = Counter()
#         with open(fpath, "r", encoding="utf-8") as f:
#             for line in f:
#                 try:
#                     obj = json.loads(line)
#                     label = obj.get("label", obj.get("label_2", None))
#                     if label is not None:
#                         counts[label] += 1
#                 except json.JSONDecodeError:
#                     continue
#         print(f"{fname}: {dict(counts)}")