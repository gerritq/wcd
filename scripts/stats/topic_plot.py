import os
import json
import argparse
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from matplotlib.backends.backend_pdf import PdfPages

# DIRs
BASE_DIR = os.getenv("BASE_WCD")
IN_DIR = os.path.join(BASE_DIR, "data/sents")
OUT_DIR = os.path.join(BASE_DIR, "data/out/topics")
os.makedirs(OUT_DIR, exist_ok=True)

def load_data(lang: str):
    path = os.path.join(IN_DIR, f"{lang}_sents.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def clean_topic(x):
    if not x:
        return None
    return x.split(".")[-1].replace("*", "").strip().lower()

def fa_vs_random(languages):

    figs = []

    for lang in tqdm(languages):
        rows = load_data(lang)

        fa = Counter([clean_topic(x['topic']) for x in rows if x['topic'] and x['source'] == "fa"])
        randoms = Counter([clean_topic(x['topic']) for x in rows if x['topic'] and x['source'] == "random"])

        all_topics = sorted(set(fa.keys()) | set(randoms.keys()))

        fa_p = np.array([fa.get(t, 0) for t in all_topics], dtype=float)
        random_q = np.array([randoms.get(t, 0) for t in all_topics], dtype=float)

        # convert to prob distributions
        eps = 1e-10
        fa_p = (fa_p + eps) / (fa_p.sum() + eps * len(all_topics))
        random_q = (random_q + eps) / (random_q.sum() + eps * len(all_topics))

        figs.append((lang, all_topics, fa_p, random_q))


    with PdfPages(os.path.join(OUT_DIR, "fa_vs_random.pdf")) as pdf:

        plt.figure() 
        plt.axis('off')
        plt.text(0.5,0.5,"Topic distributions FA vs random articles for all languages",ha='center',va='center')
        pdf.savefig()
        plt.close() 
        for lang, all_topics, fa_p, random_q in tqdm(figs):
            x = np.arange(len(all_topics))
            width = 0.35

            plt.figure(figsize=(10, 25))
            plt.barh(x - width / 2, fa_p, height=width, label="FA", color="skyblue")
            plt.barh(x + width / 2, random_q, height=width, label="Random", color="lightcoral")

            plt.yticks(x, all_topics)
            plt.gca().invert_yaxis()
            plt.ylim(-0.5, len(all_topics) - 0.5)
            plt.title(f"{lang.upper()}")
            plt.xlabel("Probability")
            plt.legend()
            plt.tight_layout()
            
            pdf.savefig()
            plt.close() 

def within_reference(languages):

    for source in ['fa', 'random']:
        figs = []
        for lang in tqdm(languages):
            rows = load_data(lang)

            pos = Counter([clean_topic(x['topic']) for x in rows if x['topic'] and x['label'] == 1 and x['source'] == source])
            neg = Counter([clean_topic(x['topic']) for x in rows if x['topic'] and x['label'] == 0 and x['source'] == source])
        
            all_topics = sorted(set(pos.keys()) | set(neg.keys()))

            pos_p = np.array([pos.get(t, 0) for t in all_topics], dtype=float)
            neg_q = np.array([neg.get(t, 0) for t in all_topics], dtype=float)

            # convert to prob distributions
            eps = 1e-10
            pos_p = (pos_p + eps) / (pos_p.sum() + eps * len(all_topics))
            neg_q = (neg_q + eps) / (neg_q.sum() + eps * len(all_topics))

            figs.append((lang, all_topics, pos_p, neg_q))


        with PdfPages(os.path.join(OUT_DIR, f"{source}.pdf")) as pdf:

            plt.figure() 
            plt.axis('off')
            plt.text(0.5,0.5,f"Topic distributions for {source} articles by label for all languages",ha='center',va='center')
            pdf.savefig()
            plt.close() 
            for lang, all_topics, pos_p, neg_q in tqdm(figs):
                x = np.arange(len(all_topics))
                width = 0.35

                plt.figure(figsize=(10, 25))
                plt.barh(x - width / 2, pos_p, height=width, label="Citation Needed", color="skyblue")
                plt.barh(x + width / 2, neg_q, height=width, label="No Citation Needed", color="lightcoral")

                plt.yticks(x, all_topics)
                plt.gca().invert_yaxis()
                plt.ylim(-0.5, len(all_topics) - 0.5)
                plt.title(f"{lang.upper()}")
                plt.xlabel("Probability")
                plt.legend()
                plt.tight_layout()
                
                pdf.savefig()
                plt.close() 

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

    # fa_vs_random(languages)
    within_reference(languages)


if __name__ == "__main__":
    main()

