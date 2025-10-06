import os
import json
from collections import defaultdict

"""
Notes
- views number between og and after aggregation the same, as we filter for colon when getting views data


"""
BASE_DIR = os.getenv("BASE_WCD")
INPUT_DIR = os.path.join(BASE_DIR, "data/raw/api")

def count_items(source: str, lang: str):
    try:
        path = os.path.join(INPUT_DIR, f"{lang}_{source}.jsonl")
        n = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                n += 1
        return n
    except:
        return 0
    
def main():
    languages  = [
        "en",  # English
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
        count_colon = 0
        file_path = os.path.join(INPUT_DIR, f"{lang}_all.jsonl")

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                count[item['source']] += 1
                if ":" in item['title']:
                    count_colon+=1
        
        for x in ['views', 'good', 'fa']:
            count[f'{x}_og'] = count_items(x, lang)
        print("Colon in title:", count_colon)
        
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