import os
import json
import random
from typing import List, Dict, Any
from datasets import Dataset, DatasetDict, load_from_disk
from collections import defaultdict

SEED = 42
random.seed(SEED)

LANG_MAP = {'english': 'en',
            'dutch': 'nl',
            'bulgarian': 'nl',
            'arabic': 'ar'}

BASE_DIR = os.getenv("BASE_WCD", ".")
IN_DIR = os.path.join(BASE_DIR, "data/raw")
OUT_DIR = os.path.join(BASE_DIR, "data/sets")

def load_data(data_name: str, lang: str) -> list:

    # here load jsonl instead of json
    path = os.path.join(IN_DIR, data_name, f"{data_name}_{lang}.jsonl")
    with open(path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    return data


def prepare_data(data_name: str, lang: str) -> list:
    
    # load data
    data = load_data(data_name=data_name, lang=lang)

    out = defaultdict(list)
    for x in data:
        out[x['split']].append(x)
    
    for k, v in out.items():
        random.shuffle(v)

    ds = DatasetDict({
        "train": Dataset.from_list(out['train']),
        "dev": Dataset.from_list(out['dev']),
        "test": Dataset.from_list(out['test']),
    })

    out_dir = os.path.join(OUT_DIR, "ex", f"{LANG_MAP[lang]}_{data_name}")
    ds.save_to_disk(out_dir)

def main():

    data = {"ct24": ["english", 'dutch', "arabic"],
            "nlp4if": ["english", 'bulgarian', "arabic"]
            }
    
    for data_name, languages in data.items():
        for lang in languages:
            print(f"\nRUNNING {data_name} {lang} ...", flush=True)        
            prepare_data(data_name=data_name, lang=lang)
        

if __name__ == "__main__":
    main()