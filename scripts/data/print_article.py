import json 
import os

BASE_DIR = os.getenv("BASE_WCD", ".")
INPUT_DIR = os.path.join(BASE_DIR, f"data/raw/parsed") 


def printer(item: dict):
    print(f"\n=============== {item['title']} ===============")
    print(f"{item['source']}")
    texts = item['text']
    for e in texts:
        print(f"\nSection {e['header']}")
        for i, p in enumerate(e['paragraphs']):
            print(f"Paragraph {i}\n", p, "\n")

def main():
    lang='id'
    INPUT_FILE = os.path.join(INPUT_DIR, f"{lang}_parsed.jsonl")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    item = data[3]
    printer(item)

if __name__ == "__main__":
    main()