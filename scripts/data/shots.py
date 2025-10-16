import os
import json
from datasets import load_from_disk

BASE_DIR = os.getenv("BASE_WCD", ".")
IN_DIR = os.path.join(BASE_DIR, "data/sets")
OUT_DIR = os.path.join(BASE_DIR, "data/sents/shots")

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
    # "zh",  # Chinese
    # "ar",  # Arabic
    "id"   # Indonesian
    ]   

    all_shots = {}
    for lang in languages:
        data = load_from_disk(os.path.join(IN_DIR, lang))['dev']

        pos = [x for x in data if x['label'] == 1][:2]
        neg = [x for x in data if x['label'] == 0][:2]

        shots = pos + neg
        shots = [{"section": x['section'], "context": x['context'], "claim": x['claim'], "label": x['label']} for x in shots]
        all_shots[lang] = shots

    shots_path = os.path.join(OUT_DIR, f"shots.json")
    with open(shots_path, "w", encoding="utf-8") as f:
        json.dump(all_shots, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()




