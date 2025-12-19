import os 
import json
import random
import pandas as pd

# ----------------------------------------------------------------
# configs
# ----------------------------------------------------------------

BASE_DIR = os.getenv("BASE_WCD", ".")
IN_DIR = os.path.join(BASE_DIR, "data/sents")
OUT_DIR = os.path.join(BASE_DIR, "data/out")

os.makedirs(OUT_DIR, exist_ok=True)

random.seed(42)    

N = 30

LANGS = {"high": ["en", "pt", "de", "ru", "it", "vi", "tr", "nl"],
         "medium": ["uk", "ro", "id", "bg", "uz"],
         "low": ["no", "az", "mk", "hy", "sq"],
         }


# ----------------------------------------------------------------
# fucntions
# ----------------------------------------------------------------
def get_random_sample(language: str) -> list[dict]:
                             
    """Data loader for the monolinugual setting.""" 
    data_dir = os.path.join(IN_DIR, f"{language}_sents.json")
    with open(data_dir, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    random.shuffle(data)

    pos = [x for x in data if x["label"] == 1][:N//2]
    neg = [x for x in data if x["label"] == 0][:N//2]

    sampled_data = pos + neg
    random.shuffle(sampled_data)
    return sampled_data

def main():

    records = []
    for resource, langs in LANGS.items():
        for lang in langs:
            print("="*10)
            print(f"Sampling {lang}")
            print("="*10)
            samples = get_random_sample(lang)
            for sample in samples:
                link = f"https://{lang}.wikipedia.org/wiki/{sample['title']}"
                record = {
                    "language": lang,
                    "resource": resource,
                    "link": link,
                    "claim": sample["claim"],
                    "prev": sample["previous_sentence"],
                    "sub": sample["subsequent_sentence"],
                    "section": sample["section"],
                    "label": sample["label"],
                    "label_correct": "",
                    "claim_correct": "",
                    "context_correct": "",
                }
                records.append(record)

    df = pd.DataFrame.from_records(records)
    out_path = os.path.join(OUT_DIR, "mwcd_quality.xlsx")
    # create df
    df.to_excel(out_path, index=False)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()