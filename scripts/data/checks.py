import os
import json 
import random
random.seed(42)

BASE_DIR = os.getenv("BASE_WCD", ".")
INPUT_DIR = os.path.join(BASE_DIR, f"data/sents") 

def main():
    
    lang='en'
    n=20

    print("="*10)
    print(f"Running {lang} ...", flush=True)
    print("="*10)

    INPUT_FILE = os.path.join(INPUT_DIR, f"{lang}_sents.json")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f) 
    random.shuffle(data)
    for i, item in enumerate(data[:n]):
        print(f"\nItem {i}")
        for key, value in item.items():
            print(key, ":", value)

if __name__ == "__main__":
    main()