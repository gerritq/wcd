import os
import json
import re
import stanza
from tqdm import tqdm
from stanza.pipeline.core import Pipeline
import torch

"""
[senza fonte]
[citation needed]

2.
  {
    "title": "2С3",
    "section": "История создания",
    "source": "fa",
    "context": "Основным применением таких САУ было непосредственное сопровождение пехоты и танков и стрельба по вражеским целям прямой наводкой.",
    "p_ends_with_citation": true,
    "p_any_citation": true,
    "sentence": "В то же время в западных странах и США имелись САУ, предназначенные для ведения огня с закрытых позиций[прим.",
    "has_citation": false,
    "claim": "В то же время в западных странах и США имелись САУ, предназначенные для ведения огня с закрытых позиций[прим.",
    "label": 1,
    "label_conservative": null
  },

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

# VARs
QUOTES = "\"'‘’“”«»‹›„‚"
QUOTES = re.escape(QUOTES)
END_PUNCT = ".!?"
END_PUNCT = re.escape(END_PUNCT)
SENT_STARTS_WITH_QUOTE=0

# for code in set(LANG_MAP.values()):
#     print(f"Downloading {code}")
#     stanza.download(code, processors="tokenize", verbose=False)


def clean(x: str):
    """various operations to clean the text"""

    # basic cleaning
    x = re.sub(r"\s+", " ", x).strip() # rm whitespace

    # special cleaning operations from screening the data
    # saw in pt but not in all articles, eg https://pt.wikipedia.org/wiki/1.%C2%BA_Esquadr%C3%A3o_de_Avi%C3%B5es_de_Intercepta%C3%A7%C3%A3o_e_Ataque
    x = re.sub(r"(\]):\s?([\d\-–]+|cap\.?\s?\d+)", r"\1", x) # first group: [19]:12-13, second group: [19]:cap. 3

    # eg en https://en.wikipedia.org/wiki/1964_Illinois_House_of_Representatives_election

    # no space between quotes and punction, eg nl https://nl.wikipedia.org/wiki/Sacharias_Jansen
    x = re.sub( rf"([{QUOTES}])\s*([?.!])", r"\1\2", x)

    return x.strip()

def clean_citation(x: str) -> str:
    # rm white space around references: eg [ 19] [ citation needed]
    x = re.sub(r'\[(\s*[\w\s]+\s*)\]', lambda m: f"[{m.group(1).strip()}]", x)
    # clean white space between references
    x = re.sub(r'\]\s+\[', '][', x)
    # clean whitespace between punction and citation
    x = re.sub(r'([.?!"])\s+\[', r'\1[', x)
    # clean whitespace between citation and punction
    x = re.sub(r'\]\s+([^\w\s])', r']\1', x)
    # insert whitespace betwee reference]-text, , eg nl https://nl.wikipedia.org/wiki/Sacharias_Jansen
    x = re.sub(r"(\])(\w)", r"\1 \2", x)

    # do we want to reverse??
    x = re.sub(r'([.?!]["\'”’]?)(\s*(?:\[[\w\s\?]+\])+)', r'\2\1', x) # ? in the second group as for some language citation needed has a question mark inlcuded, eg pt, nl
    return x

def ends_with_citation(x):
    """is this d or w in the regex?"""
    # accounts for both references after and before end token
    x = x.strip()
    # return (bool(re.search(r'(?:\[\d+\])$', x)) or  # matches .[90]
    #         bool(re.search(r'(?:\[\d+\])+[\'"”]?[.?!][\'"”]?$', x)) 
    #         )# this matches...[90]. or [90]? or ...

    return bool(re.search(rf'(?:\[(\d+|\w\s\d+)\])+(\[[\w\s]+\])*[.?!:;{QUOTES}]*$', x))
    

def has_citation(text):
    # account for citations like [r 1], eg en https://en.wikipedia.org/wiki/1964_Illinois_House_of_Representatives_election
    return bool(re.search(r'\[(\d+|\w\s\d+)\]', text))

def remove_citations(x):
    if not x:
        return None
    x = re.sub(r'\[[\w\s]+\]', '', x) # rm [19], and also notes and others
    x = x.strip()
    return x # also rm notes, and others

# def split_sentences(text: str, lang: str, nlp: Pipeline):
#     doc = nlp(text)
#     return [s.text.strip() for s in doc.sentences]


def split_sentences2(text: str, lang: str, nlp: Pipeline):
    """takes a clean text!"""
    global SENT_STARTS_WITH_QUOTE
    doc = nlp(text)
    sentences = [s.text.strip() for s in doc.sentences]

    # for n turns, check whether there are cases where the split was not succefull.abs
    # failed_split_pattern

    # corrections
    corrected = []
    for s in sentences:
        if bool(re.search(rf'^(\[\d+\])+[{END_PUNCT}]?$', s)): # sentece is one or more references, eg [2] or [2][3]
            if corrected:
                corrected[-1] = corrected[-1].rstrip() + s
        else:
            corrected.append(s)


    # some checks
    for c in corrected:
        if c.strip().startswith("["):
            SENT_STARTS_WITH_QUOTE +=1
            print("Sentence starts with a citation.")
        

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

    if not bool(re.search(rf'[{QUOTES}.!?;:]$' , x)):
        return True

    # cleaning issues
    if x.count("(") != x.count(")"):
        return True # parantheses do not match

    if x.count("[") != x.count("]"):
        return True # brackets do not match
    return False

def proc_article(article: dict, lang: str, nlp: Pipeline):
    sents = []
    for section in article['text']:
        if section['header'] == "Lead":
            continue
        for paragraph in section['paragraphs']:
            # cleaning -> order important!!
            print("\n\n+++++++++++++++\n\n", paragraph)
            paragraph = clean(paragraph) # general cleaning
            paragraph = clean_citation(paragraph) # citations
            print("\n\n\CLEAN+++++++++++++++\n\n", paragraph)
            # print(f"\n\n{paragraph}")
            # paragraph citation indicators
            p_ends_with_citation = ends_with_citation(paragraph)
            p_any_citation = has_citation(paragraph)
            
            # get sentences
            extracted_sents = split_sentences2(paragraph, lang, nlp)
            
            prev_sentence = None
            for s in extracted_sents:
                print(f"<START>{s}<END>")
                sents.append({'title': article['title'],
                              'section': section['header'],
                              'source': article['source'],
                              'context':  prev_sentence,
                              'p_ends_with_citation': p_ends_with_citation,
                              'p_any_citation': p_any_citation,
                              'sentence': s})
                prev_sentence = s
    return sents


def proc_sentence(item):
    sentence = item['sentence'].strip()
    context = item['context']
    citation = has_citation(sentence) # has_citation(sentence)

    sentence_clean = remove_citations(sentence)
    context_clean = remove_citations(context)

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
                 "context": context_clean,
                 "claim": sentence_clean,
                 'label': label,
                 'label_conservative': label_conservative})
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
        # # "zh",  # Chinese
        # "ar",  # Arabic
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
            # data = [x for x in data if x['source'] not in "views"]
        # data = [x for x in data if x['title'] == "1964 Illinois House of Representatives election"]
        print(data)
        # data = [data[10]]
        # print(data)
        data = data[:100]
        # print(data)

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

        print("Sentence dropped bc of failed splitting", SENT_STARTS_WITH_QUOTE)

    
if __name__ == "__main__":
    main()