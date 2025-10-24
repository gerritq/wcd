import stanza
import os
import json
import numpy as np
from scipy.spatial.distance import jensenshannon
from collections import Counter

##########################################################################################
# STANZA test
##########################################################################################

def js_distance_fa_random(data):

    def clean_topic(x):
        if not x:
            return None
        return x.split(".")[-1].replace("*", "").strip().lower()

    
    fa = Counter([clean_topic(x['topic']) for x in data if x['topic'] and x['source'] == "fa"])
    randoms = Counter([clean_topic(x['topic']) for x in data if x['topic'] and x['source'] == "random"])

    all_topics = sorted(set(fa + randoms))
    print(all_topics)

    # get the distributions in the right order
    fa_p = np.array([fa.get(t, 0) for t in all_topics], dtype=float)
    random_q = np.array([randoms.get(t, 0) for t in all_topics], dtype=float)

    assert len(fa_p) == len(random_q), "Mismatch in topic length."

    # make probabilirt distributions
    eps = 1e-10
    fa_p = (fa_p + eps) / (fa_p.sum() + eps * len(all_topics))
    random_q = (random_q + eps) / (random_q.sum() + eps * len(all_topics))

    # assert fa_p.sum() == 1 and random_q.sum() == 1, f"{fa_p.sum()} or {random_q.sum()}"

    js_dist = jensenshannon(fa_p, random_q, base=2.0)
    print(js_dist)
    return js_dist

if __name__ == "__main__":
    with open("/scratch/prj/inf_nlg_ai_detection/wcd/data/sents/id_sents.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    js_distance_fa_random(data)


##########################################################################################
# STANZA test
##########################################################################################


# def contains_propn(x: str, tag: str, nlp) -> int:
#     doc = nlp(x)
#     for sent in doc.sentences:
#         for word in sent.words:
#             if word.upos == tag:
#                 return 1
#     return 0

# LANG_MAP = {
#     "en": "en", 
#     "nl": "nl", 
#     "no": "nb", 
#     "it": "it", 
#     "pt": "pt", 
#     "ro": "ro",
#     "ru": "ru", 
#     "uk": "uk", 
#     "bg": "bg", 
#     "zh": "zh-hans", 
#     "ar": "ar", 
#     "id": "id",
# }

# lang = "en"
# nlp = stanza.Pipeline(lang=LANG_MAP[lang], processors='tokenize,pos', model_dir=os.getenv('HF_HOME'))
# print(contains_propn("Obama is the president.", "PROPN", nlp))
# print(contains_propn("This has no real noun.", "PROPN", nlp))