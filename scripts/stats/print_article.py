import json 
import os
from collections import defaultdict

BASE_DIR = os.getenv("BASE_WCD", ".")
INPUT_DIR = os.path.join(BASE_DIR, f"data/sents") 


def printer(item: dict):
    print(f"\n=============== {item['title']} ===============")
    print(f"{item['source']}")
    texts = item['text']
    for e in texts:
        print(f"\nSection {e['header']}")
        for i, p in enumerate(e['paragraphs']):
            print(f"Paragraph {i}\n", p, "\n")

def main():
    lang='en'
    INPUT_FILE = os.path.join(INPUT_DIR, f"{lang}_sents.json")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    titles = list(set([x['title'] for x in data]))
    for t in titles[:1]:
        all_sents = [x for x in data if x['title'] == t]

        print(f"\n=============== {t} ===============")
        sections = defaultdict(list)
        for s in all_sents:
            sections[s['section']].append(s)

        for k, v in sections.items(): 
            print(f"\n\n====== Section: {k} ======")
            for s in v:
                print(f"\tSentence ({s['label']}): <START>{s['sentence']}<END>")
    
if __name__ == "__main__":
    main()