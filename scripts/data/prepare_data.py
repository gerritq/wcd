import os
import json
import random
from datasets import Dataset, DatasetDict, load_from_disk
from collections import defaultdict, Counter

SEED = 42
random.seed(SEED)

BASE_DIR = os.getenv("BASE_WCD", ".")
IN_DIR = os.path.join(BASE_DIR, "data/sents")
OUT_DIR = os.path.join(BASE_DIR, "data/sets")
CL_DIR = os.path.join(BASE_DIR, "data/sets/cl")

def load_data(lang: str) -> list:
    """Load claims into a list of dicts"""
    path = os.path.join(IN_DIR, f"{lang}_sents.json")
    with open(path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except Exception:
            data = []
    
    out_data = []
    for x in data:
        if x['source'] in ['fa', 'good']:
            out_data.append(x)
    
    unique_titles = set([x['title'] for x in out_data])
    print("Unique titles:", len(unique_titles))
    return out_data

def split_claims_into_sets_by_article(lang: str, 
                             train_frac=0.6, 
                             dev_frac=0.2,
                             test_frac=0.2) -> dict:

    # load data
    claims = load_data(lang)

    # group claims by article
    articles_dict = defaultdict(list)
    for claim in claims:
        articles_dict[claim['title']].append(claim)
    
    # get all article titles
    articles = list(articles_dict.keys())
    # shuffle articles and assign each article into train/dev/test
    random.shuffle(articles)
    n = len(articles)
    train_end = int(n * train_frac)
    dev_end = train_end + int(n * dev_frac)
    test_end = dev_end + int(n * test_frac)

    # assign articles to sets
    train_articles, dev_articles, test_articles = articles[:train_end], articles[train_end:dev_end], articles[dev_end:]
        
    # now assign claims to sets avoiding article leakage
    train = []
    for title in train_articles:
        train.extend(articles_dict[title])

    dev = []
    for title in dev_articles:
        dev.extend(articles_dict[title])

    test = []
    for title in test_articles:
        test.extend(articles_dict[title])
        
    return {
        "train": train,
        "dev": dev,
        "test": test
    }   


def build_monolingual_dataset(configs: dict,
                              lang: str) -> None:

    # get split articles
    split_claims = split_claims_into_sets_by_article(lang)

    # build balance datasets
    full_data = {}
    for set_name, claims in split_claims.items():
        pos, neg = [], []

        if set_name == 'train':
            desired_n = configs['train_n']
        elif set_name == 'dev':
            desired_n = configs['dev_n']
        else:
            desired_n = configs['test_n']

        pos_temp, neg_temp = [], []
        for claim in claims:
            # print(claim)
            if claim['label'] == 1:
                pos_temp.append({
                    "claim": claim['claim'],
                    "label": 1,
                    "title": claim['title'],
                    "section": claim['section'],
                    "previous_sentence": claim.get('previous_sentence', ''),
                    "subsequent_sentence": claim.get('subsequent_sentence', ''),
                    "source": claim['source'],
                    "lang": lang
                })
            else:
                neg_temp.append({
                    "claim": claim['claim'],
                    "label": 0,
                    "title": claim['title'],
                    "section": claim['section'],
                    "previous_sentence": claim.get('previous_sentence', ''),
                    "subsequent_sentence": claim.get('subsequent_sentence', ''),
                    "source": claim['source'],
                    "lang": lang
                })
        random.shuffle(pos_temp)
        random.shuffle(neg_temp)
        pos.extend(pos_temp[:desired_n//2])
        neg.extend(neg_temp[:desired_n//2])

        full_data[set_name] = pos + neg
        random.shuffle(full_data[set_name])
    
    # small check
    for set_name, set_data in full_data.items():
        label_dist = defaultdict(int)
        unique_sources = set()
        unique_titles = set()
        claims_per_article = Counter()
        for x in set_data:
            label_dist[x["label"]] += 1
            unique_sources.add(x["source"])
            unique_titles.add(x["title"])
            claims_per_article[x["title"]] += 1
        
        print(f"\nSet: {set_name}")
        print("Topi 5 articles by number of claims:")
        for title, count in claims_per_article.most_common(5):
            print(f"\t{title}: {count}")
        print(f"\tUnique titles in {set_name}: {len(unique_titles)}")
        print(f"\tUnique sources in {set_name}: {unique_sources}")
        # pt case has 4945 which is ok
        if not sum(label_dist.values()) >= (configs[f"{set_name}_n"]):
            print("\nWARNING")
            print(f"Data size error: {sum(label_dist.values())}")

        if not label_dist[0] == label_dist[1]:
            print("\nWARNING")
            print(f"Label balance error {sum(label_dist.values())}")
        print(f"\tDistribuion set {set_name}: {dict(label_dist)}")


    ds = DatasetDict({
        "train": Dataset.from_list(full_data["train"]),
        "dev": Dataset.from_list(full_data["dev"]),
        "test": Dataset.from_list(full_data["test"]),
    })
    out_dir = os.path.join(OUT_DIR, "main", f"{lang}")
    ds.save_to_disk(out_dir)
    


# def prepare_data(lang: str, total_n: int) -> list:
    
#     # load data
#     data = load_data(lang)

#     # balance data by label, picking high-quality subsets first
#     n_per_label = int(total_n // 2 ) 

#     pos = []
#     neg = []
#     for subset in ['fa', 'good']: #  'good', "views"
#         temp = [x for x in data if x['source'] == subset]
        
#         temp_pos = [x for x in temp if x['label'] == 1]
#         temp_neg = [x for x in temp if x['label'] == 0]
        
#         pos.extend(temp_pos)
#         neg.extend(temp_neg)

#     random.shuffle(pos)
#     random.shuffle(neg)

#     final_data = pos[:n_per_label] + neg[:n_per_label]

#     assert len(final_data) == total_n, "Data error."

#     source_label_dist = defaultdict(lambda: defaultdict(int))
#     for x in final_data:
#         source_label_dist[x["source"]][x["label"]] += 1
#     print("\tSource × Label distribution:")
#     for src, lbls in source_label_dist.items():
#         print(f"\t\t{src}: {dict(lbls)}")
#     print(" ")
#     return final_data

# def build_monolingual_dataset(lang: str, total_n: int) -> None:
#     # load and select data
#     all_data = prepare_data(lang, total_n)
#     data=[]
#     for x in all_data:
#         # change label_conservative here to test
#         data.append({"claim": x['claim'], 
#                      "label": int(x["label"]),
#                      "title": x['title'],
#                      "section": x['section'],
#                      "previous_sentence": x['previous_sentence'],
#                      "subsequent_sentence": x['subsequent_sentence'],
#                      "source": x['source'],
#                      "lang": lang })

#     # split data
#     split_1 = int(0.8 * len(data))
#     split_2 = int(0.9 * len(data))
#     split_1_half = int(split_1 // 2)
#     split_2_half = int(split_2 // 2)
    
#     pos = [x for x in data if x["label"] == 1]
#     neg = [x for x in data if x["label"] == 0]
#     random.shuffle(pos)
#     random.shuffle(neg)

#     train = pos[:split_1_half] + neg[:split_1_half]
#     dev = pos[split_1_half:split_2_half] + neg[split_1_half:split_2_half]
#     test = pos[split_2_half:] + neg[split_2_half:]

#     random.shuffle(train)
#     random.shuffle(dev)
#     random.shuffle(test)

#     # small check
#     print("\tFinal distribution")
#     for set_, name in zip([train, dev, test], ['train', 'dev', 'test']):
#         label_dist = defaultdict(int)
#         for x in set_:
#             label_dist[x["label"]] += 1
#         print(f"\tDistribuion set {name}: {dict(label_dist)}")

#     ds = DatasetDict({
#         "train": Dataset.from_list(train),
#         "dev": Dataset.from_list(dev),
#         "test": Dataset.from_list(test),
#     })

#     out_dir = os.path.join(OUT_DIR, "main", f"{lang}")
#     ds.save_to_disk(out_dir)


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
        "id",   # Indonesian
        "vi",
        "tr"
    ]
    
    configs = {
                "train_n": 4000,
                "dev_n": 400,
                "test_n": 400
        }

    for lang in languages:
        print(f"\nRUNNING {lang} ...", flush=True)

        
        # mono main set
        build_monolingual_dataset(configs=configs, lang=lang)

    # cross-lingual data
    # training_languages = ['en', 'it', 'ru']
    # test_languages = ['no', 'ro', 'bg']
    # build_cross_lingual_training_data(training_languages=training_languages, 
    #                                   test_languages=test_languages,
    #                                   in_dir=OUT_DIR,
    #                                   out_dir=CL_DIR)

    # multilingual
    # out_dir = os.path.join(OUT_DIR, f"multi")
    # build_multilingual_training_data(languages, total_n, out_dir)

if __name__ == "__main__":
    main()