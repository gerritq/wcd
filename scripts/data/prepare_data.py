import os
import json
import random
from typing import List, Dict, Any
from datasets import Dataset, DatasetDict, Features, ClassLabel, Value
from collections import defaultdict

SEED = 42
random.seed(SEED)

BASE_DIR = os.getenv("BASE_WCD", ".")
IN_DIR = os.path.join(BASE_DIR, "data/sents")
OUT_DIR = os.path.join(BASE_DIR, "data/sets")

def load_data(lang: str, n: int) -> list:
    target_per_label = n // 2  

    # load data
    path = os.path.join(IN_DIR, f"{lang}_sents.json")
    with open(path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except Exception:
            data = []

    # bucket by source and label
    data_by_source_label = defaultdict(list)
    for item in data:
        key = f"{item['source']}_{item['label']}"
        data_by_source_label[key].append(item)

    data_out = []
    for label in [0, 1]:
        collected = []
        remaining = target_per_label
        for source in ["fa", "good", "views"]:  # high to low quality
            key = f"{source}_{label}"
            if key not in data_by_source_label:
                continue
            random.shuffle(data_by_source_label[key])
            take = min(remaining, len(data_by_source_label[key]))
            collected.extend(data_by_source_label[key][:take])
            remaining -= take
            if remaining == 0:
                break
        data_out.extend(collected)

    random.shuffle(data_out)
    label_dist = defaultdict(int)
    source_label_dist = defaultdict(lambda: defaultdict(int))
    for x in data_out:
        label_dist[x["label"]] += 1
        source_label_dist[x["source"]][x["label"]] += 1

    print(f"Len total data {len(data_out)}")
    print(f"Label distribution: {dict(label_dist)}")
    print("Source × Label distribution:")
    for src, lbls in source_label_dist.items():
        print(f"  {src}: {dict(lbls)}")
    return data_out

def build_monolingual_dataset(lang: str, n: int) -> DatasetDict:
    data = load_data(lang, n)
    random.shuffle(data)
    assert len(data) == n, "Data length error."

    k = n // 2
    pos = [x for x in data if x["label"] == 1][:k]
    neg = [x for x in data if x["label"] == 0][:k]
    assert len(pos) == k, "Positive data does not meet min length."
    assert len(neg) == k, "Negative data does not meet min length."
    data = pos + neg
    random.shuffle(data)

    items = []
    for x in data:
        # change label_conservative here to test
        items.append({"claim": x['claim'], 
                      "label": int(x["label"])})
    ds = Dataset.from_list(items)
    ds = ds.cast_column("label", ClassLabel(num_classes=2, names=["0", "1"])) # needed to use stratify below
    split = ds.train_test_split(test_size=0.1, seed=SEED, stratify_by_column="label")
    
    label_dist = defaultdict(int)
    for x in split['test']:
        label_dist[x["label"]] += 1
    print(f"Test set label balance {dict(label_dist)}")
    return DatasetDict({"train": split["train"], "test": split["test"]})

def build_multilingual_dataset(langs: List[str], n: int, test_share: float = .1) -> DatasetDict:
    # is here the full training data
    n_langs = len(langs)

    per_lang_total = n // n_langs
    per_lang_train = int(round(per_lang_total * (1 - test_share)))
    per_lang_test  = int(round(n * test_share))
    
    train = []
    test = []
    for lang in langs:
        data = load_data(lang)
        random.shuffle(data)

        pos = []
        neg = []
        for x in data:
            item = {'lang': lang, 'claim': x['claim'], 'label': x['label']}
            if x['label'] == 1:
                pos.append(item)
            else:
                neg.append(item)

        tr_pos_n = min(len(pos), per_lang_train // 2)
        tr_neg_n = min(len(neg), per_lang_train - tr_pos_n)

        te_pos_n = min(len(pos) - tr_pos_n, per_lang_test // 2)
        te_neg_n = min(len(neg) - tr_neg_n, per_lang_test - te_pos_n)

        train.extend(pos[:tr_pos_n] + neg[:tr_neg_n])
        test.extend(pos[tr_pos_n:tr_pos_n + te_pos_n] + neg[tr_neg_n:tr_neg_n + te_neg_n])

    random.shuffle(train)
    random.shuffle(test)

    # check 
    def counts(items):
        d = defaultdict(lambda: defaultdict(int))
        for x in items:
            d[x["lang"]][x["label"]] += 1
        return {k: dict(v) for k, v in d.items()}

    print("Training counts:", counts(train))
    print("Test counts:", counts(test))

    return DatasetDict({
        "train": Dataset.from_list(train),
        "test": Dataset.from_list(test)
    })

def save_dataset(ds: DatasetDict, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    ds.save_to_disk(out_dir)

def main():

    languages  = [
        # "en",  # English
        "nl",  # Dutch
        "no",  # Norwegian (Bokmål is 'nb', Nynorsk is 'nn', 'no' redirects to Bokmål)
        # "it",  # Italian
        # "pt",  # Portuguese
        "ro",  # Romanian
        # "ru",  # Russian
        # "uk",  # Ukrainian
        # "bg",  # Bulgarian
        # "zh",  # Chinese
        # "ar",  # Arabic
        # "id"   # Indonesian
    ]
    
    n = 5000

    for lang in languages:
        print(f"\nRUNNING {lang} ...", flush=True)
        # mono
        ds = build_monolingual_dataset(lang, n)
        out_dir = os.path.join(OUT_DIR, f"{lang}")
        save_dataset(ds, out_dir)

    # multilingual
    # ds = build_multilingual_dataset(langs, n)
    # out_dir = os.path.join(OUT_DIR, f"multi_{n}")
    # save_dataset(ds, out_dir)

if __name__ == "__main__":
    main()