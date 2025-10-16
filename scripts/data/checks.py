import os
import json 
import random
random.seed(42)

BASE_DIR = os.getenv("BASE_WCD", ".")
INPUT_DIR = os.path.join(BASE_DIR, f"data/sents") 

def ends_with_colon(pos, neg) -> None:

    pos_n = 0
    for x in pos:
        if x['claim'].endswith(";") or x['claim'].endswith(":"):
            pos_n +=1
    print("\tPositive share ending with colon or semi-colon", round(pos_n / len(pos), 2))

    neg_n = 0
    for x in neg:
        if x['claim'].endswith(";") or x['claim'].endswith(":"):
            neg_n +=1
    print("\tNegative share ending with colon or semi-colon", round(neg_n / len(neg), 2))

def main():
    
    lang='pt'
    n=20

    print("="*10)
    print(f"Running {lang} ...", flush=True)
    print("="*10)

    INPUT_FILE = os.path.join(INPUT_DIR, f"{lang}_sents.json")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f) 
        random.shuffle(data)
        pos = [x for x in data if x['label'] == 1]
        neg = [x for x in data if x['label'] == 0]
        checks = pos[:n//2] + neg[:n//2]

        ends_with_colon(pos, neg)
    for i, item in enumerate(checks):
        print(f"\nItem {i}")
        print(item)

if __name__ == "__main__":
    main()