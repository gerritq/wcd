import os
import json 
import random
random.seed(42)

BASE_DIR = os.getenv("BASE_WCD", ".")
INPUT_DIR = os.path.join(BASE_DIR, f"data/sents") 

def main():
    
    lang='it'
    n=20

    print("="*10)
    print(f"Running {lang} ...", flush=True)
    print("="*10)

    INPUT_FILE = os.path.join(INPUT_DIR, f"{lang}_sents.json")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f) 
        random.shuffle(data)
        pos = [x for x in data if x['label'] == 1][:n//2]
        neg = [x for x in data if x['label'] == 0][:n//2]
        checks = pos + neg
    for i, item in enumerate(checks):
        print(f"\nItem {i}")
        print(item)

if __name__ == "__main__":
    main()