import os
import json
import re
import stanza
from tqdm import tqdm
from stanza.pipeline.core import Pipeline

"""
General
- treat ar and zh separately

Corrections
- en:  -Beyoncé reflecting on 4 to GQ in 2024[7]
-- rm div class templatequotecite
DONE

- en: She intended 4 to help change that status, commenting, \"Figuring out a way to get R&B back on the radio is challenging ..
-- avoid spliiting ... ?

- it: Area direttiva
-- can rm text in bold, though it is in a paragraph; but this is also dropped bc of length

- it: Nota: in corsivo i calciatori che hanno lasciato la società a stagione in corso.
-- rm small

- label
-- label 99 is a claim in a paragaph with citation, but has no direct citation
-- we could train a classifier on those and check whether they make a difference?
-- question is do we inlcude them to 1 or do we drop them?

- en: Buses in Chennai were branded with the promotional slogan "Namma Chess, Namma Pride" (trans.
-- add custom words

- cleaning
-- no open parantheses

- checks
-- add a check that no brackets are in the cleaned sentences!!

- drop sections in parsing
-- any(section in item['section'].lower() for section in DROP_SECTIONS_LANG) or
-- add to this also bold etc
"""


BASE_DIR = os.getenv("BASE_WCD", ".")
INPUT_DIR = os.path.join(BASE_DIR, f"data/raw/parsed") 
OUTPUT_DIR = os.path.join(BASE_DIR, f"data/sents")

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

# for code in set(LANG_MAP.values()):
#     print(f"Downloading {code}")
#     stanza.download(code, processors="tokenize", verbose=False)

def clean_citation(x: str) -> str:
    # rm white space around references: eg [ 19] [ citation needed]
    x = re.sub(r'\[(\s*[\w\s]+\s*)\]', lambda m: f"[{m.group(1).strip()}]", x)
    return x

def ends_with_citation(x):
    """is this d or w in the regex?"""
    # accounts for both references after and before end token
    x = x.strip()
    return (bool(re.search(r'(?:\[\d+\])$', x)) or  # matches .[90]
            bool(re.search(r'(?:\[\d+\])+[\'"”]?[.?!][\'"”]?$', x)) 
            )# this matches...[90]. or [90]? or ...

def has_citation(text):
    return bool(re.search(r'\[\d+\]', text))

def remove_citations(x):
    x = re.sub(r'\[[\w\s]+\]', '', x) # rm [19], and also notes and others
    x = x.strip()
    return x # also rm notes, and others

def split_sentences(text: str, lang: str, nlp: Pipeline):
    doc = nlp(text)
    return [s.text.strip() for s in doc.sentences]

def drop_sentence(x: str) -> bool:
    '''takes clean sent and outputs whether it is clean or should be dropped du to parsing errors'''
    x = x.strip()
    
    if len(x.split()) < 4:
        return True

    if len(x) < 15:
        return True

    # sentence does not begin with an uppercase, when it starts with a letter
    if x[0].isalpha() and not x[0].isupper():
        return True

    if not bool(re.search(r'[.!?"\'”;:]$' , x)):
        return True

    # cleaning issues
    if x.count("(") != x.count(")"):
        return True # parantheses do not match
    return False

def proc_article(article: dict, lang: str, nlp: Pipeline):
    sents = []
    for section in article['text']:
        if section['header'] == "Lead":
            continue
        for paragraph in section['paragraphs']:
            # clean citations
            paragraph = clean_citation(paragraph)
            
            # paragraph citation indicators
            p_ends_with_citation = ends_with_citation(paragraph)
            p_any_citation = has_citation(paragraph)
            
            # get sentences
            extracted_sents = split_sentences(paragraph, lang, nlp)
            for s in extracted_sents:
                sents.append({'title': article['title'],
                              'section': section['header'],
                              'p_ends_with_citation': p_ends_with_citation,
                              'p_any_citation': p_any_citation,
                              'sentence': s})
    return sents


def proc_sentence(item):
    sentence = item['sentence'].strip()
    citation = has_citation(sentence) # has_citation(sentence)

    sentence_clean = remove_citations(sentence)

    if drop_sentence(sentence_clean):
        return None
    
    # citation label 
    label_conservative=None
    if not citation and not item['p_any_citation']:
        label = 0 # not citation and p has no citation => no citation needed
        label_conservative = 0
    else:
        label = 1 # has citation or is in a p with citation => citation needed

    if citation:
        label_conservative = 1
    
    # else:
    #     label = 99

    item.update({"has_citation": citation,
                 "claim": sentence_clean,
                 'label': label,
                 'label_conservative': label})
    return item

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
        "zh",  # Chinese
        "ar",  # Arabic
        "id"   # Indonesian
    ]
    for lang in languages:
        print("="*10)
        print(f"Running {lang} ...", flush=True)
        print("="*10)
        
        INPUT_FILE = os.path.join(INPUT_DIR, f"{lang}_parsed.jsonl")
        OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"{lang}_sents.json")

        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
        data = data[:50]

        # define parser
        nlp = stanza.Pipeline(lang='en', processors="tokenize", tokenize_pretokenized=False, use_gpu=True)

        all_sents = []
        for article in tqdm(data):
            all_sents.extend(proc_article(article, lang, nlp))

        out_sents = []
        for sent in all_sents:
            item = proc_sentence(sent)
            if item:
                out_sents.append(item)

        with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
            json.dump(all_sents, out_f, ensure_ascii=False, indent=2)

    
if __name__ == "__main__":
    main()
        

# PIPELINES = {
#     k: stanza.Pipeline(LANG_MAP[k], processors="tokenize", tokenize_pretokenized=False, use_gpu=False)
#     for k in LANG_MAP
# }

# # --- Citation patterns ---
# # Brackets across scripts: ASCII [] () and full-width Chinese/Japanese 【】（）
# OPEN = r"\[\(\（【"
# CLOSE = r"\]\)\）】"
# # A single citation like [12], [citation needed], (1), 【参见】 — avoid line breaks inside
# CIT = rf"{OPEN}[^{OPEN}{CLOSE}\n]{{1,120}}{CLOSE}"
# # One or more citations, possibly with spaces between them
# CIT_CLUSTER = rf"(?:\s*{CIT})+"

# # Sentence-ending punctuation incl. CJK and ellipsis
# SENT_PUNCT = r"[.!?。！？…]"

# # Compiled regexes
# LEADING_CIT_RE = re.compile(rf"^\s*({CIT_CLUSTER})")
# TRAILING_CIT_RE = re.compile(rf"({CIT_CLUSTER})\s*$")
# END_PUNCT_THEN_CIT_RE = re.compile(rf"({SENT_PUNCT})(\s*{CIT_CLUSTER})\s*$")
# ANY_CIT_RE = re.compile(CIT)



# def normalize_citations(sentences):
#     """
#     Post-process sentences:
#     - Move any leading citation cluster to the previous sentence
#     - Keep trailing citations with the sentence
#     Returns list of dicts: {'text': str, 'citations': [str]}
#     """
#     sents = sentences[:]  # copy
#     # 1) Move leading citations to previous sentence
#     out = []
#     for i, s in enumerate(sents):
#         m = LEADING_CIT_RE.match(s)
#         if m and i > 0:
#             leading = m.group(1).strip()
#             # attach to previous
#             out[-1] = out[-1] + (" " if not out[-1].endswith(" ") else "") + leading
#             s = s[m.end():].lstrip()
#         out.append(s)

#     # 2) Extract citations per sentence & optionally normalise punctuation-before/after citations
#     results = []
#     for s in out:
#         # If sentence ends with punctuation followed by citations, keep as-is (they belong here)
#         # If citations are inside the sentence already, also fine

#         # Collect citations
#         citations = [m.group(0) for m in ANY_CIT_RE.finditer(s)]

#         # Optionally also provide a clean sentence without citations
#         clean = ANY_CIT_RE.sub("", s).strip()
#         # Remove double spaces created by removal
#         clean = re.sub(r"\s{2,}", " ", clean)

#         results.append({"text": s, "clean_text": clean, "citations": citations})
#     return results

# def split_with_citations(text: str, lang: str):
#     """High-level helper: returns list of {'text','clean_text','citations'}."""
#     sents = stanza_sent_split(text, lang)
#     return normalize_citations(sents)

# # ---------------------------
# # Example
# # ---------------------------
# if __name__ == "__main__":
#     example = (
#         "This is a sentence [1]. Next sentence starts with a citation. [2] It continues.\n"
#         "Chinese style example：这是一个句子[3]。 接下来是另一个句子。[4] 继续。\n"
#         "Arabic example: هذه جملة[5]. [6] ثم جملة أخرى"
#     )
#     for lg in ["en", "zh", "ar"]:
#         print(f"\n=== {lg} ===")
#         for i, s in enumerate(split_with_citations(example, lg), 1):
#             print(i, s)