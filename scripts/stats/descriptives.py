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


"""
- add script, language family and group
"""
LANG_ARICLES = {"en": 7100146,
                "nl": 2204186,
                "no": 661961,
                "it": 1946643,
                "pt": 1161324,
                "ro": 518582,
                "ru": 2074183,
                "uk": 1398784,
                "bg": 306755,
                "vi": 1296489,
                "id": 756871,
                "tr": 654170}

LANG_FA_ARTICLES = {"en": 6822,
                    "nl":  380,
                    "no": 351,
                    "it": 592,
                    "pt": 1535,
                    "ro": 200,
                    "ru": 2091,
                    "uk": 249,
                    "bg": 148,
                    "vi": 478,
                    "id": 422,
                    "tr": 229}

LANG_SCRIPTS = {
    "en": "Latin",
    "nl": "Latin",
    "no": "Latin",
    "it": "Latin",
    "pt": "Latin",
    "ro": "Latin",
    "ru": "Cyrillic",
    "uk": "Cyrillic",
    "bg": "Cyrillic",
    "vi": "Latin",
    "id": "Latin",
    "tr": "Latin"
}

LANG_GROUPS = {
    "en": "Germanic",
    "nl": "Germanic",
    "no": "Germanic",
    "it": "Romance",
    "pt": "Romance",
    "ro": "Romance",
    "ru": "Slavic",
    "uk": "Slavic",
    "bg": "Slavic",
    "vi": "Austroasiatic",
    "id": "Austronesian",
    "tr": "Turkic"
}

LANG_MAP = {
    "en": "en", 
    "nl": "nl", 
    "no": "nb", 
    "it": "it", 
    "pt": "pt", 
    "ro": "ro",
    "ru": "ru", 
    "uk": "uk", 
    "bg": "bg", 
    "ar": "ar", 
    "id": "id",
    "tr": "tr",
    "vi": "vi"
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
        

def js_distance_fa_between_labels(data: list[dict]) -> float:
    """gets the js distance between fa and random topics
    to do: may use different topic levels. Currently we are on the lowest with 64 unique topics
    """

    def clean_topic(x):
        if not x:
            return None
        return x.split(".")[-1].replace("*", "").strip().lower()

    
    pos = Counter([clean_topic(x['topic']) for x in data if x['topic'] and x['source'] == "fa" and int(x['label']) == 1])
    neg = Counter([clean_topic(x['topic']) for x in data if x['topic'] and x['source'] == "fa" and int(x['label']) == 0])
    
    all_topics = sorted(set(pos + neg))
    

    # get the distributions in the right order
    pos_p = np.array([pos.get(t, 0) for t in all_topics], dtype=float)
    neg_q = np.array([neg.get(t, 0) for t in all_topics], dtype=float)

    assert len(pos_p) == len(neg_q), "Mismatch in topic length."

    # make probabilirt distributions
    eps = 1e-10
    pos_p = (pos_p + eps) / (pos_p.sum() + eps * len(all_topics))
    neg_q = (neg_q + eps) / (neg_q.sum() + eps * len(all_topics))

    # assert fa_p.sum() == 1 and random_q.sum() == 1, f"{fa_p.sum()} or {random_q.sum()}"

    js_dist = jensenshannon(pos_p, neg_q, base=2.0)
    return js_dist

def contain_number(x):
    return int(any(ch.isdigit() for ch in str(x)))

def contains_propn(x: str, tag: str, nlp) -> int:
    doc = nlp(x)
    for sent in doc.sentences:
        for word in sent.words:
            if word.upos == tag:
                return 1
    return 0

def compute_stats(data: list, nlp, nlp_bool: bool) -> dict:
    
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
        if nlp_bool:
            avg_contain_propn = np.mean([contains_propn(x['claim'], "PROPN", nlp) for x in nlp_subset])
        else:
            avg_contain_propn = 1
        stats["fa"][label] = {
            "total": total,
            "avg_len": round(float(avg_len), 1),
            "contain_num": round(float(avg_contain_number) * 100, 1),
            "contain_propn": round(float(avg_contain_propn) * 100, 1),
        }
    return stats

def get_cell(sdict, source, label):
    """Return a 3-tuple (#, len, num%) with safe defaults."""
    d = sdict.get(source, {}).get(label, None)
    if d is None:
        return (0, 0.0, 0.0, 0.0)
    return (d["total"], d["avg_len"], d["contain_num"], d["contain_propn"])

def main():

    nlp_bool = bool(int(sys.argv[1]))
    langs = ['en', 'nl', 'no', 'it', 'pt', 'ro', 'ru', 'uk', 'bg', "vi", "id", "tr"]
    # gather stats
    lang_stats = {}
    topic_stats = {}
    for lang in langs:
        print(f"\tRunning {lang}")
        data = load_data(lang)
        if nlp_bool:
            nlp = stanza.Pipeline(lang=LANG_MAP[lang], 
                                  processors='tokenize,mwt,pos', 
                                  model_dir=os.getenv('HF_HOME'),
                                  dir=os.getenv('HF_HOME'))
        else:
            nlp = None
        lang_stats[lang] = compute_stats(data, nlp, nlp_bool)
        topic_stats[lang] = js_distance_fa_between_labels(data)

    table = []
    table.append("\\begin{tabular}{l ccccc cccc cccc c}")
    
    table.append(" & \\multicolumn{4}{c}{\\textbf{Language Statistics}} & \\multicolumn{4}{c}{\\textbf{Citation Needed (1)}} & \\multicolumn{4}{c}{\\textbf{No Citation Needed (0)}} \\\\")
    table.append("\\cmidrule(lr){2-5}\\cmidrule(l){6-9}\\cmidrule(l){10-13}")
    table.append(" & \\textbf{\\#Articles} & \\textbf{\\#FA Articles} & \\textbf{Language Group} & \\textbf{Script} & \\textbf{\\#Claims} & \\textbf{Avg Len} & \\textbf{Numeric (\\%)} & \\textbf{PROPN (\\%)} & \\textbf{\\#Claims} & \\textbf{Avg Len} & \\textbf{Numeric (\\%)} & \\textbf{PROPN (\\%)} & \\textbf{Topic Sim.} \\\\")    
    table.append("\\midrule")

    for lang in langs:
        sdict = lang_stats.get(lang, {})
        n_articles = LANG_ARICLES[lang]
        fa_articles = LANG_FA_ARTICLES[lang]
        lang_group = LANG_GROUPS[lang]
        script = LANG_SCRIPTS[lang]
        fa0 = get_cell(sdict, "fa", 0)
        fa1 = get_cell(sdict, "fa", 1)
        js_distance = topic_stats[lang]

        row = (
            f"{lang} & "
            f"\\multicolumn{{1}}{{|c}}{{{n_articles:,}}} & {fa_articles:,} & {lang_group} & {script} & "
            f"\\multicolumn{{1}}{{|c}}{{{fa0[0]:,}}} & {fa0[1]:.1f} & {fa0[2]:.1f} & \\multicolumn{{1}}{{c|}}{{{fa0[3]:.1f}}} & "
            f"{fa1[0]:,} & {fa1[1]:.1f} & {fa1[2]:.1f} & \\multicolumn{{1}}{{c|}}{{{fa1[3]:.1f}}} & \\multicolumn{{1}}{{c|}}{{{js_distance:.2f}}} \\\\ "
        )
        table.append(row)

    table.append("\\bottomrule")
    table.append("\\end{tabular}")

    latex_table = "\n".join(table)
    print(latex_table)

    # optionally save to file
    out_path = os.path.join(OUT_DIR, "desc_new.tex")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(latex_table)

if __name__ == "__main__":
    main()