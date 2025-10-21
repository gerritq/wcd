import os
import json
import argparse
import numpy as np
from collections import defaultdict
import stanza
import random

# DIRs
BASE_DIR = os.getenv("BASE_WCD")
IN_DIR = os.path.join(BASE_DIR, "data/sents")
OUT_DIR = os.path.join(BASE_DIR, "data/out")
os.makedirs(OUT_DIR, exist_ok=True)

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
    "zh": "zh-hans", 
    "ar": "ar", 
    "id": "id",
}


def load_data(lang: str):
    path = os.path.join(IN_DIR, f"{lang}_sents.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def contain_number(x):
    return int(any(ch.isdigit() for ch in str(x)))

def contains_propn(x: str, tag: str, nlp) -> int:
    doc = nlp(x)
    for sent in doc.sentences:
        for word in sent.words:
            if word.upos == tag:
                return 1
    return 0

def compute_stats(data, nlp):
    stats = defaultdict(dict)
    for source in ['fa', 'random']:
        for label in [0, 1]:
            subset = [x for x in data if x.get('source') == source and int(x.get('label', 0)) == label]
            random.shuffle(subset)
            nlp_subset = subset[:300]
            if not subset:
                continue
            total = len(subset)
            avg_len = np.mean([len(str(x['claim']).split()) for x in subset])
            avg_contain_number = np.mean([contain_number(x['claim']) for x in subset])
            avg_contain_propn = np.mean([contains_propn(x['claim'], "PROPN", nlp) for x in nlp_subset])
            # avg_contain_propn = 1
            stats[source][label] = {
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


    # gather stats
    lang_stats = {}
    for lang in ['en', 'nl', 'no', 'it', 'pt', 'ro', 'ru', 'uk', 'bg']:
        print(f"\tRunning {lang}")
        data = load_data(lang)
        nlp = stanza.Pipeline(lang=LANG_MAP[lang], 
                              processors='tokenize,mwt,pos', 
                              model_dir=os.getenv('HF_HOME'),
                              dir=os.getenv('HF_HOME'))
        # nlp = None
        lang_stats[lang] = compute_stats(data, nlp)

    table = []
    table.append("\\begin{tabular}{l cccc cccc cccc cccc}")
    table.append(" & \\multicolumn{8}{c}{\\textbf{Featured Articles}} & \\multicolumn{8}{c}{\\textbf{Random Articles}}\\\\")
    table.append("\\cmidrule(lr){2-9}\\cmidrule(l){10-17}")
    table.append(" & \\multicolumn{4}{c}{\\textbf{No Citation Needed (0)}} & \\multicolumn{4}{c}{\\textbf{Citation Needed (1)}} & \\multicolumn{4}{c}{\\textbf{No Citation Needed (0)}} & \\multicolumn{4}{c}{\\textbf{Citation Needed (1)}}\\\\")
    table.append("\\cmidrule(lr){2-5}\\cmidrule(l){6-9}\\cmidrule(l){10-13}\\cmidrule(l){14-17}")
    table.append(" & \\textbf{\\#Claims} & \\textbf{Avg Len} & \\textbf{Numeric (\\%)} & \\textbf{PROPN (\\%)} & \\textbf{\\#Claims} & \\textbf{Avg Len} & \\textbf{Numeric (\\%)} & \\textbf{PROPN (\\%)} & \\textbf{\\#Claims} & \\textbf{Avg Len} & \\textbf{Numeric (\\%)} & \\textbf{PROPN (\\%)} & \\textbf{\\#Claims} & \\textbf{Avg Len} & \\textbf{Num (\\%)} & \\textbf{PROPN (\\%)}\\\\")
    table.append("\\midrule")

    for lang in ['en', 'nl', 'no', 'it', 'pt', 'ro', 'ru', 'uk', 'bg']:
        sdict = lang_stats.get(lang, {})
        fa0 = get_cell(sdict, "fa", 0)
        fa1 = get_cell(sdict, "fa", 1)
        rd0 = get_cell(sdict, "random", 0)
        rd1 = get_cell(sdict, "random", 1)

        row = (
            f"{lang} & "
            f"\\multicolumn{{1}}{{|c}}{{{fa0[0]}}} & {fa0[1]:.1f} & {fa0[2]:.1f} & \\multicolumn{{1}}{{c|}}{{{fa0[3]:.1f}}} & "
            f"{fa1[0]} & {fa1[1]:.1f} & {fa1[2]:.1f} & \\multicolumn{{1}}{{c|}}{{{fa1[3]:.1f}}} & "
            f"{rd0[0]} & {rd0[1]:.1f} & {rd0[2]:.1f} & \\multicolumn{{1}}{{c|}}{{{rd0[3]:.1f}}} & "
            f"{rd1[0]} & {rd1[1]:.1f} & {rd1[2]:.1f} & \\multicolumn{{1}}{{c|}}{{{rd1[3]:.1f}}}\\\\"
        )
        table.append(row)

    table.append("\\bottomrule")
    table.append("\\end{tabular}")

    latex_table = "\n".join(table)
    print(latex_table)

    # optionally save to file
    out_path = os.path.join(OUT_DIR, "descriptives.tex")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(latex_table)

if __name__ == "__main__":
    main()