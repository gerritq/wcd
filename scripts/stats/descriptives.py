import os
import json
import argparse
import numpy as np
from collections import defaultdict
import stanza
import random
from scipy.spatial.distance import jensenshannon
from collections import Counter
import sys

# DIRs
BASE_DIR = os.getenv("BASE_WCD")
IN_DIR = os.path.join(BASE_DIR, "data/sents")
OUT_DIR = os.path.join(BASE_DIR, "data/out/tables")
os.makedirs(OUT_DIR, exist_ok=True)

metric = "cosine"

"""
- add script, language family and group
"""
# LANG_ARICLES = {"en": 7100146,
#                 "nl": 2204186,
#                 "no": 661961,
#                 "it": 1946643,
#                 "pt": 1161324,
#                 "ro": 518582,
#                 "ru": 2074183,
#                 "uk": 1398784,
#                 "bg": 306755,
#                 "vi": 1296489,
#                 "id": 756871,
#                 "tr": 654170}

LANG_METADATA = {
    # High resource
    "en": {
        "name": "English",
        "script": "Latin",
        "family": "Indo-European",
        "group": "Germanic",
        "resource": "high"
    },
    # "zh": {
    #     "name": "Chinese",
    #     "script": "Han",
    #     "family": "Sino-Tibetan",
    #     "group": "Sinitic",
    #     "resource": "high"
    # },
    "pt": {
        "name": "Portuguese",
        "script": "Latin",
        "family": "Indo-European",
        "group": "Romance",
        "resource": "high"
    },
    "de": {
        "name": "German",
        "script": "Latin",
        "family": "Indo-European",
        "group": "Germanic",
        "resource": "high"
    },
    "ru": {
        "name": "Russian",
        "script": "Cyrillic",
        "family": "Indo-European",
        "group": "Slavic",
        "resource": "high"
    },
    "it": {
        "name": "Italian",
        "script": "Latin",
        "family": "Indo-European",
        "group": "Romance",
        "resource": "high"
    },
    "vi": {
        "name": "Vietnamese",
        "script": "Latin",
        "family": "Austroasiatic",
        "group": "Mon-Khmer",
        "resource": "high"
    },
    "tr": {
        "name": "Turkish",
        "script": "Latin",
        "family": "Turkic",
        "group": "Oghuz",
        "resource": "high"
    },
    "nl": {
        "name": "Dutch",
        "script": "Latin",
        "family": "Indo-European",
        "group": "Germanic",
        "resource": "high"
    },

    # Mid resource
    # "th": {
    #     "name": "Thai",
    #     "script": "Thai",
    #     "family": "Kra-Dai",
    #     "group": "Tai",
    #     "resource": "mid"
    # },
    "uk": {
        "name": "Ukrainian",
        "script": "Cyrillic",
        "family": "Indo-European",
        "group": "Slavic",
        "resource": "mid"
    },
    "ro": {
        "name": "Romanian",
        "script": "Latin",
        "family": "Indo-European",
        "group": "Romance",
        "resource": "mid"
    },
    "id": {
        "name": "Indonesian",
        "script": "Latin",
        "family": "Austronesian",
        "group": "Malayo-Polynesian",
        "resource": "mid"
    },
    "bg": {
        "name": "Bulgarian",
        "script": "Cyrillic",
        "family": "Indo-European",
        "group": "Slavic",
        "resource": "mid"
    },
    "uz": {
        "name": "Uzbek",
        "script": "Latin",
        "family": "Turkic",
        "group": "Karluk",
        "resource": "mid"
    },
    # Low resource
    "no": {
        "name": "Norwegian",
        "script": "Latin",
        "family": "Indo-European",
        "group": "Germanic",
        "resource": "low"
    },
    "az": {
        "name": "Azerbaijani",
        "script": "Latin",
        "family": "Turkic",
        "group": "Oghuz",
        "resource": "low"
    },
    "mk": {
        "name": "Macedonian",
        "script": "Cyrillic",
        "family": "Indo-European",
        "group": "Slavic",
        "resource": "low"
    },
    "hy": {
        "name": "Armenian",
        "script": "Armenian",
        "family": "Indo-European",
        "group": "Armenian",
        "resource": "low"
    },
    "sq": {
        "name": "Albanian",
        "script": "Latin",
        "family": "Indo-European",
        "group": "Albanian",
        "resource": "low"
    },
}

def load_data(lang: str):
    path = os.path.join(IN_DIR, f"{lang}_sents.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # we do a small adjustment by coutning good articesl as fa articles
    # this is only for id and tr, and we will put an asteriks in the table to indicate that
    for x in data:
        if x['source'] == "good":
            x['source'] = "fa"

    return data

def count_unique_titles(data: list[dict]) -> int:
    unique = set()
    for item in data:
        unique.add(item["title"])
    return len(unique)
        
def cosine_similarity_fa_between_labels(data: list[dict]) -> float:
    """Cosine similarity between topic distributions of label=1 vs label=0."""
    def clean_topic(x):
        if not x:
            return None
        return x.split(".")[-1].replace("*", "").strip().lower()

    pos = Counter(
        clean_topic(x["topic"])
        for x in data
        if x["topic"] and x["source"] == "fa" and int(x["label"]) == 1
    )
    neg = Counter(
        clean_topic(x["topic"])
        for x in data
        if x["topic"] and x["source"] == "fa" and int(x["label"]) == 0
    )

    all_topics = sorted(set(pos + neg))

    pos_p = np.array([pos.get(t, 0) for t in all_topics], dtype=float)
    neg_q = np.array([neg.get(t, 0) for t in all_topics], dtype=float)

    if pos_p.sum() == 0 or neg_q.sum() == 0:
        return 0.0

    # normalise to distributions (not strictly necessary for cosine, but fine)
    pos_p = pos_p / pos_p.sum()
    neg_q = neg_q / neg_q.sum()

    num = float(np.dot(pos_p, neg_q))
    den = float(np.linalg.norm(pos_p) * np.linalg.norm(neg_q)) + 1e-12
    return num / den

def contain_number(x):
    return int(any(ch.isdigit() for ch in str(x)))

def compute_stats(data: list) -> dict:
    
    stats = defaultdict(dict)
    
    for label in [0, 1]:
        subset = [x for x in data if x['source'] == "fa" and int(x['label']) == label]
        
        random.shuffle(subset)
        nlp_subset = subset[:300]
        if not subset:
            continue
        
        total = len(subset)
        avg_len = np.mean([len(str(x['claim']).split()) for x in subset])
        avg_contain_number = np.mean([contain_number(x['claim']) for x in subset])
        stats["fa"][label] = {
            "total": total,
            "avg_len": round(float(avg_len), 1),
            "contain_num": round(float(avg_contain_number) * 100, 1),
        }
    return stats

def get_cell(sdict, source, label):
    """Return a 3-tuple (#, len, num%) with safe defaults."""
    d = sdict.get(source, {}).get(label, None)
    if d is None:
        return (0, 0.0, 0.0)
    return (d["total"], d["avg_len"], d["contain_num"])

def main():

    langs = LANG_METADATA.keys() # order by resource level
    # gather stats
    lang_stats = {}
    topic_stats = {}
    for lang in langs:

        print("="*20)
        print("Language:", lang)
        print("="*20)
        
        data = load_data(lang)
        unique_titles = count_unique_titles(data)
        
        lang_stats[lang] = compute_stats(data)
        topic_stats[lang] = cosine_similarity_fa_between_labels(data)
        
    table = []
    table.append("\\begin{tabular}{l cccc ccc ccc c}")
    
    table.append(" & \\multicolumn{4}{c}{\\textbf{Language Statistics}} & \\multicolumn{3}{c}{\\textbf{No Citation Needed (0)}} & \\multicolumn{3}{c}{\\textbf{Citation Needed (1)}} \\\\")
    table.append("\\cmidrule(lr){2-5}\\cmidrule(l){6-8}\\cmidrule(l){9-11}")
    table.append(" & \\textbf{Resource} & \\textbf{\\#Featured Articles} & \\textbf{Language Family/Group} & \\textbf{Script} & \\textbf{\\#Claims} & \\textbf{Avg Len} & \\textbf{#Numeric (\\%)} & \\textbf{\\#Claims} & \\textbf{Avg Len} & \\textbf{#Numeric (\\%)} &  \\textbf{Topic Sim.} \\\\")    
    table.append("\\hline")

    total_no_citation_needed = 0
    total_citation_needed = 0
    
    prev_resource_level = ""
    for lang in langs:
        sdict = lang_stats.get(lang, {})
        
        resource_level = LANG_METADATA[lang]["resource"]
        unique_titles = count_unique_titles(load_data(lang))
        fa_articles = unique_titles
        lang_family, lang_group = LANG_METADATA[lang]["family"], LANG_METADATA[lang]["group"]
        lang_family_group = f"{lang_family}/{lang_group}"
        script = LANG_METADATA[lang]["script"]
        
        fa0 = get_cell(sdict, "fa", 0)
        fa1 = get_cell(sdict, "fa", 1)
        total_no_citation_needed += fa0[0]
        total_citation_needed += fa1[0]
        topic_sim = topic_stats[lang]

        if resource_level != prev_resource_level and lang != "en":
            if prev_resource_level != "":
                table.append("\\\hdashline")
            prev_resource_level = resource_level

        row = (
            f"{LANG_METADATA[lang]['name']} ({lang}) & "
            f"{resource_level} & {fa_articles:,} & {lang_family_group} & {script} & "
            f"{fa0[0]:,} & {fa0[1]:.1f} & {fa0[2]:.1f} & "
            f"{fa1[0]:,} & {fa1[1]:.1f} & {fa1[2]:.1f} & {topic_sim:.2f} \\\\ "
        )
        table.append(row)

    table.append("\\midrule")
    table.append(
        f"\\textbf{{Total}} &  &  &  &  & "
        f"{total_no_citation_needed:,} &  & & "
        f"{total_citation_needed:,} &  & & \\\\ "
    )
    table.append("\\end{tabular}")

    latex_table = "\n".join(table)
    print("\n\n")
    print(latex_table)
    print("\n\n")

    # optionally save to file
    # out_path = os.path.join(OUT_DIR, "desc_new.tex")
    # with open(out_path, "w", encoding="utf-8") as f:
    #     f.write(latex_table)

if __name__ == "__main__":
    main()