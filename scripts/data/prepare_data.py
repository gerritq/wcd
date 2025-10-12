import os
import json
import random
from typing import List, Dict, Any
from datasets import Dataset, DatasetDict, load_from_disk
from collections import defaultdict

SEED = 42
random.seed(SEED)

BASE_DIR = os.getenv("BASE_WCD", ".")
IN_DIR = os.path.join(BASE_DIR, "data/sents")
OUT_DIR = os.path.join(BASE_DIR, "data/sets")

def prepare_data(lang: str, total_n: int) -> list:
    
    def load_data(lang: str) -> list:
        path = os.path.join(IN_DIR, f"{lang}_sents.json")
        with open(path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except Exception:
                data = []
        return data

    # load data
    data = load_data(lang)

    # balance data by label, picking high-quality subsets first
    n_per_label = total_n // 2  

    sorted_pos = []
    sorted_neg = []
    for subset in ['fa', 'good', "views"]:
        temp = [x for x in data if x['source'] == subset]
        
        temp_pos = [x for x in temp if x['label'] == 1]
        temp_neg = [x for x in temp if x['label'] == 0]
        
        sorted_pos.extend(temp_pos)
        sorted_neg.extend(temp_neg)

    final_data = sorted_pos[:n_per_label] + sorted_neg[:n_per_label]

    assert len(final_data) == total_n, "Data error."

    source_label_dist = defaultdict(lambda: defaultdict(int))
    for x in final_data:
        source_label_dist[x["source"]][x["label"]] += 1
    print("Source × Label distribution:")
    for src, lbls in source_label_dist.items():
        print(f"\t{src}: {dict(lbls)}")

    return final_data

def build_monolingual_dataset(lang: str, total_n: int, out_dir: str) -> None:
    # load and select data
    data = load_data(lang, total_n)
    items=[]
    for x in data:
        # change label_conservative here to test
        items.append({"claim": x['claim'], 
                      "label": int(x["label"])})

    # split data
    split_1 = int(0.8 * len(items))
    split_2 = int(0.9 * len(items))
    split_1_half = split_1 // 2
    split_2_half = split_2 // 2
    
    pos = random.shuffle([x for x in items if x["label"] == 1])
    neg = random.shuffle([x for x in items if x["label"] == 0])

    train = pos[:split_1_half] + neg[:split_1_half]
    dev = pos[split_1_half:split_2_half] + neg[split_1_half:split_2_half]
    test = pos[split_2_half:] + neg[split_2_half:]

    # small check
    for set_, name in zip([train, dev, test], ['train', 'dev', 'test']):
        label_dist = defaultdict(int)
        for x in set_:
            label_dist[x["label"]] += 1
        print(f"Distribuion set {name}: {dict(label_dist)}")

    ds = DatasetDict({
        "train": Dataset.from_list(train),
        "dev": Dataset.from_list(dev),
        "test": Dataset.from_list(test),
    })
    ds.save_to_disk(out_dir)

def build_multilingual_training_data(languages: List[str], total_n: int, out_dir: str) -> None:
    """Takes data from the monolingual datasets"""
    training_n = int(.8 * total_n)
    val_n = int(.1 * total_n)
    n_languages = len(languages)
    
    # train split
    train_n_per_language = training_n // n_languages
    train_n_per_language_per_label = train_n_per_language // 2

    # dev split
    val_n_per_language = val_n // n_languages
    val_n_per_language_per_label = val_n_per_language // 2

    train = []
    dev = []
    for lang in languages: 
        # load data 
        in_dir = os.path.join(OUT_DIR, lang)
        temp = load_from_disk(in_dir)
        
        # train
        pos = [x for x in temp['train'] if x['label'] == 1]
        neg = [x for x in temp['train'] if x['label'] == 0]

        train.extend(pos[:train_n_per_language_per_label])
        train.extend(neg[:train_n_per_language_per_label])

        # dev
        pos = [x for x in temp['dev'] if x['label'] == 1]
        neg = [x for x in temp['dev'] if x['label'] == 0]

        dev.extend(pos[:val_n_per_language_per_label])
        dev.extend(neg[:val_n_per_language_per_label])

        # add language labels
        for x in train:
            x['lang'] = lang
        for x in dev:
            x['lang'] = lang

    # small check
    for set_, name in zip([train, dev], ['train', 'dev']):
        label_dist = defaultdict(lambda: defaultdict(int))
        for x in set_:
            label_dist[x["lang"]][x['label']] += 1

        label_dist = {k: dict(v) for k, v in label_dist.items()}
        print(f"Distribuion set {name}: {dict(label_dist)}")

    ds = DatasetDict({
        "train": Dataset.from_list(train),
        "dev": Dataset.from_list(val),
        })
    ds.save_to_disk(out_dir)

def main():

    languages  = [
        #  "en",  # English
        # "nl",  # Dutch
        # "no",  # Norwegian (Bokmål is 'nb', Nynorsk is 'nn', 'no' redirects to Bokmål)
        # "it",  # Italian
        # "pt",  # Portuguese
        # "ro",  # Romanian
        # "ru",  # Russian
        # "uk",  # Ukrainian
        # "bg",  # Bulgarian
        # "zh",  # Chinese
        # "ar",  # Arabic
        # "id"   # Indonesian
    ]
    
    # set n
    training_n = 5000
    total_n = 5000 / .8 # assuming .1 dev and test

    for lang in languages:
        print(f"\nRUNNING {lang} ...", flush=True)
        # mono
        out_dir = os.path.join(OUT_DIR, f"{lang}")
        ds = build_monolingual_dataset(lang, total_n, out_dir)

    # multilingual
    out_dir = os.path.join(OUT_DIR, f"multi")
    build_multilingual_training_data(languages, total_n, out_dir)

if __name__ == "__main__":
    main()