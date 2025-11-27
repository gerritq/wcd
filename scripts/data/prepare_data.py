import os
import json
import random
from typing import List, Dict, Any
from datasets import Dataset, DatasetDict, load_from_disk
from collections import defaultdict

SEED = 42
random.seed(SEED)

BASE_DIR = os.getenv("BASE_WCD", ".")
IN_DIR = os.path.join(BASE_DIR, "data/sents_new")
OUT_DIR = os.path.join(BASE_DIR, "data/sets_new")

def load_data(lang: str) -> list:
    path = os.path.join(IN_DIR, f"{lang}_sents.json")
    with open(path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except Exception:
            data = []
    return data

def prepare_data(lang: str, total_n: int) -> list:
    
    # load data
    data = load_data(lang)

    # balance data by label, picking high-quality subsets first
    n_per_label = int(total_n // 2 ) 

    pos = []
    neg = []
    for subset in ['fa', 'good']: #  'good', "views"
        temp = [x for x in data if x['source'] == subset]
        
        temp_pos = [x for x in temp if x['label'] == 1]
        temp_neg = [x for x in temp if x['label'] == 0]
        
        pos.extend(temp_pos)
        neg.extend(temp_neg)

    random.shuffle(pos)
    random.shuffle(neg)

    final_data = pos[:n_per_label] + neg[:n_per_label]

    assert len(final_data) == total_n, "Data error."

    source_label_dist = defaultdict(lambda: defaultdict(int))
    for x in final_data:
        source_label_dist[x["source"]][x["label"]] += 1
    print("\tSource × Label distribution:")
    for src, lbls in source_label_dist.items():
        print(f"\t\t{src}: {dict(lbls)}")
    print(" ")
    return final_data

def build_monolingual_dataset(lang: str, total_n: int) -> None:
    # load and select data
    all_data = prepare_data(lang, total_n)
    data=[]
    for x in all_data:
        # change label_conservative here to test
        data.append({"claim": x['claim'], 
                     "label": int(x["label"]),
                     "title": x['title'],
                     "section": x['section'],
                     "previous_sentence": x['previous_sentence'],
                     "subsequent_sentence": x['subsequent_sentence'],
                     "source": x['source'],
                     "lang": lang })

    # split data
    split_1 = int(0.8 * len(data))
    split_2 = int(0.9 * len(data))
    split_1_half = int(split_1 // 2)
    split_2_half = int(split_2 // 2)
    
    pos = [x for x in data if x["label"] == 1]
    neg = [x for x in data if x["label"] == 0]
    random.shuffle(pos)
    random.shuffle(neg)

    train = pos[:split_1_half] + neg[:split_1_half]
    dev = pos[split_1_half:split_2_half] + neg[split_1_half:split_2_half]
    test = pos[split_2_half:] + neg[split_2_half:]

    random.shuffle(train)
    random.shuffle(dev)
    random.shuffle(test)

    # small check
    print("\tFinal distribution")
    for set_, name in zip([train, dev, test], ['train', 'dev', 'test']):
        label_dist = defaultdict(int)
        for x in set_:
            label_dist[x["label"]] += 1
        print(f"\tDistribuion set {name}: {dict(label_dist)}")

    ds = DatasetDict({
        "train": Dataset.from_list(train),
        "dev": Dataset.from_list(dev),
        "test": Dataset.from_list(test),
    })

    out_dir = os.path.join(OUT_DIR, "main", f"{lang}")
    ds.save_to_disk(out_dir)

# def build_multilingual_training_data(languages: List[str], total_n: int, out_dir: str) -> None:
#     """Takes data from the monolingual datasets"""
#     training_n = int(.8 * total_n)
#     dev_n = int(.1 * total_n)
#     n_languages = len(languages)
    
#     # train split
#     train_n_per_language = training_n // n_languages
#     train_n_per_language_per_label = train_n_per_language // 2

#     # dev split
#     dev_n_per_language = dev_n // n_languages
#     dev_n_per_language_per_label = dev_n_per_language // 2

#     train = []
#     dev = []
#     for lang in languages: 
#         # load data 
#         in_dir = os.path.join(OUT_DIR, lang)
#         temp = load_from_disk(in_dir)
        
#         # for x in temp['train']:
#         #     x['lang'] = lang
#         # for x in temp['dev']:
#         #     x['lang'] = lang

#         # train
#         pos = [x for x in temp['train'] if x['label'] == 1]
#         neg = [x for x in temp['train'] if x['label'] == 0]

#         train.extend(pos[:train_n_per_language_per_label])
#         train.extend(neg[:train_n_per_language_per_label])

#         # dev
#         pos = [x for x in temp['dev'] if x['label'] == 1]
#         neg = [x for x in temp['dev'] if x['label'] == 0]

#         dev.extend(pos[:dev_n_per_language_per_label])
#         dev.extend(neg[:dev_n_per_language_per_label])

#         # # add language labels
#         # for x in train:
#         #     x['lang'] = lang
#         # for x in dev:
#         #     x['lang'] = lang

#     # small check
#     for set_, name in zip([train, dev], ['train', 'dev']):
#         label_dist = defaultdict(lambda: defaultdict(int))
#         for x in set_:
#             label_dist[x["lang"]][x['label']] += 1

#         label_dist = {k: dict(v) for k, v in label_dist.items()}
#         print(f"Distribuion set {name}: {dict(label_dist)}")

#     ds = DatasetDict({
#         "train": Dataset.from_list(train),
#         "dev": Dataset.from_list(dev),
#         })
#     ds.save_to_disk(out_dir)

def build_random_test_set(lang: str, total_n: int):
    data = load_data(lang)
    data = [x for x in data if x['source'] == "random"]

    n_per_label = int(total_n // 2 ) 

    pos = [x for x in data if x['label'] == 1]
    neg = [x for x in data if x['label'] == 0]
    random.shuffle(pos)
    random.shuffle(neg)

    assert len(pos) > n_per_label and len(neg) > n_per_label,  f"Too few data for {lang}"

    final_data = pos[:n_per_label] + neg[:n_per_label]

    # if we do 625 we get 624 which is ok
    # assert len(final_data) == total_n, f"Data error. Len data{len(final_data)}"

    source_label_dist = defaultdict(lambda: defaultdict(int))
    for x in final_data:
        source_label_dist[x["source"]][x["label"]] += 1
    print("\tSource × Label distribution:")
    for src, lbls in source_label_dist.items():
        print(f"\t\t{src}: {dict(lbls)}")
    print(" ")

    ds = DatasetDict({
        "test": Dataset.from_list(final_data)
        })

    out_dir = os.path.join(OUT_DIR, "random", f"{lang}")
    ds.save_to_disk(out_dir)


def main():

    languages  = [
        "en",  # English
        "nl",  # Dutch
        # "no",  # Norwegian (Bokmål is 'nb', Nynorsk is 'nn', 'no' redirects to Bokmål)
        # "it",  # Italian
        # "pt",  # Portuguese
        # "ro",  # Romanian
        # "ru",  # Russian
        # "uk",  # Ukrainian
        # "bg",  # Bulgarian
        # "id",   # Indonesian
        # "vi",
        # "tr"
    ]
    
    # set n
    training_n = 5000
    total_n = training_n / .8 # assuming .1 dev and test
    random_n = (training_n / .8) * .1

    for lang in languages:
        print(f"\nRUNNING {lang} ...", flush=True)

        
        # mono main set
        build_monolingual_dataset(lang, total_n)
        # build_random_test_set(lang, random_n)



    # multilingual
    # out_dir = os.path.join(OUT_DIR, f"multi")
    # build_multilingual_training_data(languages, total_n, out_dir)

if __name__ == "__main__":
    main()