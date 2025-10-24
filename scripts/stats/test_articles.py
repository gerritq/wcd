import json 
import os
from collections import defaultdict
import random

random.seed(42)

BASE_DIR = os.getenv("BASE_WCD", ".")
INPUT_DIR = os.path.join(BASE_DIR, f"data/sents") 
OUT_PATH = os.path.join(BASE_DIR, "data/out/article_tests.txt")

def printer(item: dict):
    print(f"\n=============== {item['title']} ===============")
    print(f"{item['source']}")
    texts = item['text']
    for e in texts:
        print(f"\nSection {e['header']}")
        for i, p in enumerate(e['paragraphs']):
            print(f"Paragraph {i}\n", p, "\n")

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
        "id",
        "vi",
        "tr"
    ]
    
    txt =""
    for lang in languages:
        INPUT_FILE = os.path.join(INPUT_DIR, f"{lang}_sents.json")
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        titles = sorted(set(x['title'] for x in data))
        random.shuffle(titles)

        txt += f"\n\n#========== LANGUAGE {lang} ==========\n\n"
        for i, t in enumerate(titles[:20]):
            all_sents = [x for x in data if x['title'] == t]

            txt += f"\n\n## ========== Article ({lang}) {i}: {t} =========="
            sections = defaultdict(list)
            for s in all_sents:
                sections[s['section']].append(s)

            for k, v in sections.items(): 
                txt += f"\n\n### ========== Section: {k} =========="
                for s in v:
                    txt += f"\n\tSentence ({s['label']}): <START>{s['sentence']}<END>\n"
                    txt += f"\n\tClaim ({s['label']}): <START>{s['claim']}<END>\n"


    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(txt)
    
if __name__ == "__main__":
    main()