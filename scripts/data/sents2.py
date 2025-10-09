import os
import json
import re
import stanza
from tqdm import tqdm
from stanza.pipeline.core import Pipeline
import torch
"""
General
- treat ar and zh separately
- drop multipel white space!!!!
DONE

General segmenation check:
- single quote needsd to be appended to previous, if sentence starts with reference, same; 
DONE
- check that a sentence does not start with citations
!!!
- accoutn for noise in the reference splitting case
!!

- manual splitting of those that did not work: <START>A observação da Guerra das Malvinas, travada em 1982 entre a Argentina e o Reino Unido, fez a Marinha do Brasil perceber sua fraqueza num hipotético conflito no Atlântico Sul.[1] Aeronaves argentinas afundaram ou danificaram vários navios britânicos com mísseis antinavio e bombas, e só não fizeram mais dano devido às baixas pesadas que sofreram para as aeronaves britânicas com mísseis ar-ar.<END>
!!

- what the heck: Na ausência do 1.º GAE, o Minas Gerais foi reduzido ao papel de porta-helicópteros.[1]:20
-- rm double brackets, and also add that to the cleaning
- do the next sentence begins with citations ...

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


- pt
-- cleaning of cap or things that are behind citations ...
-- see here: https://pt.wikipedia.org/wiki/1.%C2%BA_Esquadr%C3%A3o_de_Avi%C3%B5es_de_Intercepta%C3%A7%C3%A3o_e_Ataque
- -177 as well!!!!!
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


def clean(x: str):
    """various operations to clean the text"""

    # basic cleaning
    x = re.sub(r"\s+", " ", x).strip() # rm whitespace

    # special cleaning operations from screening the data
    # saw in pt but not in all articles, eg https://pt.wikipedia.org/wiki/1.%C2%BA_Esquadr%C3%A3o_de_Avi%C3%B5es_de_Intercepta%C3%A7%C3%A3o_e_Ataque
    x = re.sub(r"(\]):\s?([\d-]+|cap\.?\s?\d+)", r"\1", x) # first group: [19]:12-13, second group: [19]:cap. 3


    return x.strip()

def clean_citation(x: str) -> str:
    # rm white space around references: eg [ 19] [ citation needed]
    x = re.sub(r'\[(\s*[\w\s]+\s*)\]', lambda m: f"[{m.group(1).strip()}]", x)
    # clean white space between references
    x = re.sub(r'\]\s+\[', '][', x)
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

# def split_sentences(text: str, lang: str, nlp: Pipeline):
#     doc = nlp(text)
#     return [s.text.strip() for s in doc.sentences]


def split_sentences2(text: str, lang: str, nlp: Pipeline):
    """takes a clean text!"""
    doc = nlp(text)
    sentences = [s.text.strip() for s in doc.sentences]

    # corrections
    corrected = []
    for s in sentences:
        if bool(re.search(r'^(\[\d+\])+$', s)): # sentece is one or more references, eg [2] or [2][3]
            if corrected:
                corrected[-1] = corrected[-1].rstrip() + s
        else:
            corrected.append(s)


    return corrected


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
            # cleaning -> order important!!
            paragraph = clean(paragraph) # general cleaning
            paragraph = clean_citation(paragraph) # citations
            
            print(f"\n\n{paragraph}")
            # paragraph citation indicators
            p_ends_with_citation = ends_with_citation(paragraph)
            p_any_citation = has_citation(paragraph)
            
            # get sentences
            extracted_sents = split_sentences2(paragraph, lang, nlp)
            for s in extracted_sents:
                print(f"<START>{s}<END>")
                sents.append({'title': article['title'],
                              'section': section['header'],
                              'source': article['source'],
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
                 'label_conservative': label_conservative})
    return item

def main():
    languages  = [
        # "en",  # English
        # "nl",  # Dutch
        # "no",  # Norwegian (Bokmål is 'nb', Nynorsk is 'nn', 'no' redirects to Bokmål)
        # "it",  # Italian
        "pt",  # Portuguese
        # "ro",  # Romanian
        # "ru",  # Russian
        # "uk",  # Ukrainian
        # "bg",  # Bulgarian
        # # "zh",  # Chinese
        # # "ar",  # Arabic
        #  "id"   # Indonesian
    ]
    for lang in languages:
        print("="*10)
        print(f"Running {lang} ...", flush=True)
        print("="*10)
        
        INPUT_FILE = os.path.join(INPUT_DIR, f"{lang}_parsed.jsonl")
        OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"{lang}_sents.json")

        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
            # data = [x for x in data if x['source'] not in "views"]
        data = [x for x in data if x['title'] == "1.º Esquadrão de Aviões de Interceptação e Ataque"]
        print(data)
        # data = data[:50]

        # define parser
        nlp = stanza.Pipeline(lang=LANG_MAP[lang], processors="tokenize", tokenize_pretokenized=False, use_gpu=torch.cuda.is_available())

        all_sents = []
        for article in tqdm(data):
            all_sents.extend(proc_article(article, lang, nlp))

        out_sents = []
        for sent in all_sents:
            item = proc_sentence(sent)
            if item:
                out_sents.append(item)

        with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
            json.dump(out_sents, out_f, ensure_ascii=False, indent=2)

    
if __name__ == "__main__":
    main()