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
                             train_frac=0.7, 
                             dev_frac=0.15,
                             test_frac=0.15) -> dict:

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

    random.shuffle(train)
    random.shuffle(dev)
    random.shuffle(test)
        
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
        desired_n = configs[f"{set_name}_n"]

        for claim in claims:
            # print(claim)
            if claim['label'] == 1:
                pos.append({
                    "claim": claim['claim'],
                    "label": 1,
                    "title": claim['title'],
                    "section": claim['section'],
                    "previous_sentence": claim['previous_sentence'],
                    "subsequent_sentence": claim['subsequent_sentence'],
                    "source": claim['source'],
                    "lang": lang
                })
            else:
                neg.append({
                    "claim": claim['claim'],
                    "label": 0,
                    "title": claim['title'],
                    "section": claim['section'],
                    "previous_sentence": claim['previous_sentence'],
                    "subsequent_sentence": claim['subsequent_sentence'],
                    "source": claim['source'],
                    "lang": lang
                })
        
        random.shuffle(pos)
        random.shuffle(neg)
        pos = pos[:desired_n // 2]
        neg = neg[:desired_n // 2]

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
        
        print("")
        print(f"\nSET: {set_name}")
        print(f"Unique titles in {set_name}: {len(unique_titles)}")
        print(f"Unique sources in {set_name}: {unique_sources}")

        # pt case has 4945 which is ok
        if not sum(label_dist.values()) >= (configs[f"{set_name}_n"]):
            print("\nWARNING")
            print(f"Data size error: {sum(label_dist.values())}")

        if not label_dist[0] == label_dist[1]:
            print("\nWARNING")
            print(f"Label balance error {sum(label_dist.values())}")
        print(f"\tDistribuion set {set_name}: {dict(label_dist)}")


        print("Top 5 articles by number of claims:")
        for title, count in claims_per_article.most_common(5):
            print(f"\t{title}: {count}")
        
        print(" ")

    ds = DatasetDict({
        "train": Dataset.from_list(full_data["train"]),
        "dev": Dataset.from_list(full_data["dev"]),
        "test": Dataset.from_list(full_data["test"]),
    })
    out_dir = os.path.join(OUT_DIR, "main", f"{lang}")
    ds.save_to_disk(out_dir)
    
def main():

    languages  = [
        # "en",  # English
        # "nl",  # Dutch
        # "no",  # Norwegian (Bokmål is 'nb', Nynorsk is 'nn', 'no' redirects to Bokmål)
        # "it",  # Italian
        # "pt",  # Portuguese
        # "ro",  # Romanian
        # "ru",  # Russian
        # "uk",  # Ukrainian
        # "sr",  # Serbian
        # "bg",  # Bulgarian
        # "id",   # Indonesian
        # "vi",
        # "tr",
        # "sq",
        # "mk",
        # "hy",
        # "az",
        "de",
        "uz"
    ]
    
    configs = {
                "train_n": 8000,
                "dev_n": 500,
                "test_n": 500
        }

    for lang in languages:
        print("="*20, flush=True)
        print(f"LANGUAGE {lang} ...", flush=True)
        print("="*20, flush=True)

        
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