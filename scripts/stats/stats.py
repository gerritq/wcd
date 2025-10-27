import os
import json
from collections import defaultdict
from datasets import Dataset, DatasetDict, load_from_disk

"""
Notes
- views number between og and after aggregation the same, as we filter for colon when getting views data
"""
BASE_DIR = os.getenv("BASE_WCD")
API_DIR = os.path.join(BASE_DIR, "data/raw/api")
SENTS_DIR = os.path.join(BASE_DIR, "data/sents")
SETS_DIR = os.path.join(BASE_DIR, "data/sets")
PARSE_DIR = os.path.join(BASE_DIR, "data/raw/parse")

def final_sets_examples():
    pass

def final_sets_distributions():

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
        "id",
        "vi",
        "tr"
    ]

    for lang in languages:
        print("="*5, lang, "="*5)
        
        for subdir in ['main', 'random']:
            print("="*3, subdir, "="*3)
            in_dir = os.path.join(SETS_DIR, subdir, lang)
            data = load_from_disk(in_dir)
            for set_name in data:
                source_label_dist = defaultdict(lambda: defaultdict(int))
                for x in data[set_name]:
                    source_label_dist[x["source"]][x["label"]] += 1
                print(f"{set_name.upper()} Source × Label distribution:")
                for src, lbls in source_label_dist.items():
                    print(f"\t\t{src}: {dict(lbls)}")
                print("")

def count_articles(source: str, lang: str):
    """count articles at different processing stages - tbd"""
    try:
        path = os.path.join(API_DIR, f"{lang}_{source}.jsonl")
        n = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                n += 1
        return n
    except:
        return 0

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
        "id",
        "vi",
        "tr"
    ]
    
def final_article_distribution():
    def count_sents(lang: str):
        count = defaultdict(int)
        count_source = defaultdict(lambda: defaultdict(int))
        try:
            path = os.path.join(SENTS_DIR, f"{lang}_sents.json")
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            print("\tTotal N sentences", len(data))
            for item in data:
                label = item['label']
                source = item['source']
                count[label] += 1
                count_source[source][label] +=1
        except Exception as e:
            print(f"Issue counting sents {e}")
        
        sorted_count = dict(sorted(count.items()))
        print("\t Total N labels:", sorted_count, "\n")

        sorted_count_source = {
            source: dict(sorted(label_counts.items()))
            for source, label_counts in count_source.items()
        }
        print("\tLabels by source:", sorted_count_source, "\n")
        
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
        "id",
        "vi",
        "tr"
    ]

    for lang in languages:
        count = defaultdict(int)
        all_count = defaultdict(int)
        count_colon = 0
        file_path = os.path.join(INPUT_DIR, f"{lang}_all.jsonl")

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                count[item['source']] += 1
                # if ":" in item['title']:
                #     count_colon+=1
        
        # this adds the oriingal counts of the sources
        # for x in ['views', 'good', 'fa']:
        #     count[f'{x}_og'] = count_articles(x, lang)

        # print("Colon in title:", count_colon)
        
        sorted_count = dict(sorted(count.items()))
        print(f"========== {lang} ==========")
        print("Final distribution of articles")
        print(sorted_count, "\n")

        # print("Distribution of sentences")
        # count_final_ds(lang)

        print("Distribution of sentences")
        count_sents(lang)

if __name__ == "__main__":
    # final_article_distribution()

    final_sets_distributions()

# for fname in sorted(os.listdir(BASE_DIR)):
#     fpath = os.path.join(BASE_DIR, fname)
#     if os.path.isfile(fpath) and fname.endswith(".jsonl"):
#         counts = Counter()
#         with open(fpath, "r", encoding="utf-8") as f:
#             for line in f:
#                 try:
#                     obj = json.loads(line)
#                     label = obj.get("label", obj.get("label_2", None))
#                     if label is not None:
#                         counts[label] += 1
#                 except json.JSONDecodeError:
#                     continue
#         print(f"{fname}: {dict(counts)}")