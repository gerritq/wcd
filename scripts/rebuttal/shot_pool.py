import os
import json
from datasets import load_from_disk

BASE_DIR = os.getenv("BASE_WCD", ".")
IN_DIR = os.path.join(BASE_DIR, "data/sets/main")
OUT_DIR = os.path.join(BASE_DIR, "scripts/rebuttal")
os.makedirs(OUT_DIR, exist_ok=True)

def main():

    languages = ["id", "az"]

    all_shots = {}
    for lang in languages:
        data = load_from_disk(os.path.join(IN_DIR, lang))['dev']

        pos = [x for x in data if x['label'] == 1][:30]
        neg = [x for x in data if x['label'] == 0][:30]

        shots = pos + neg
        shots = [{"section": x['section'],
                  "previous_sentence": x['previous_sentence'], 
                  "claim": x['claim'],
                  "subsequent_sentence": x['subsequent_sentence'], 
                  "claim": x['claim'],
                  "label": x['label']} for x in shots]
        all_shots[lang] = shots

    shots_path = os.path.join(OUT_DIR, f"all_shots.json")
    with open(shots_path, "w", encoding="utf-8") as f:
        json.dump(all_shots, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()




