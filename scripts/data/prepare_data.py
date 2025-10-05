import os
import json
import random
from typing import List, Dict, Any
from datasets import Dataset, DatasetDict
from collections import defaultdict

SEED = 42
random.seed(SEED)

BASE_DIR = os.getenv("BASE_WCD", ".")
OUT_DIR = os.path.join(BASE_DIR, "data/sets")

def load_data(lang: str) -> List[Dict[str, Any]]:
    path = os.path.join(BASE_DIR, f"data/proc/{lang}_sents.jsonl")
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                data.append(obj)
            except Exception:
                continue
    return data

def build_monolingual_dataset(lang: str, n: int) -> DatasetDict:
    data = load_data(lang)
    random.shuffle(data)
    k = n // 2
    pos = [x for x in data if x["label"] == 1][:k]
    neg = [x for x in data if x["label"] == 0][:k]
    data = pos + neg
    random.shuffle(data)

    items = []
    for x in data:
        items.append({"claim": x['claim'], "label": int(x["label"])})
    ds = Dataset.from_list(items)
    split = ds.train_test_split(test_size=0.1, seed=SEED)
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

    langs = ['en', 'pt', 'hu', 'pl']
    n = 5000

    for lang in langs:
        # mono
        ds = build_monolingual_dataset(lang, n)
        out_dir = os.path.join(OUT_DIR, f"{lang}_{n}")
        save_dataset(ds, out_dir)

    # multilingual
    ds = build_multilingual_dataset(langs, n)
    out_dir = os.path.join(OUT_DIR, f"multi_{n}")
    save_dataset(ds, out_dir)

if __name__ == "__main__":
    main()