import os
import json
import re
import stanza
from tqdm import tqdm
from stanza.pipeline.core import Pipeline
import torch
import argparse

BASE_DIR = os.getenv("BASE_WCD", ".")
INPUT_DIR = os.path.join(BASE_DIR, f"data/raw/parsed") 
OUTPUT_DIR = os.path.join(BASE_DIR, f"data/sents")
INFO_DIR = os.path.join(BASE_DIR, f"data/info")

parser = argparse.ArgumentParser()
parser.add_argument("--languages", nargs="+", required=True)
args = parser.parse_args()

# For stanza
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
    "id": "id",
    "vi": "vi",
    "tr": "tr",
}

# VARs
QUOTES = "\"'‘’“”«»‹›„‚"
QUOTES = re.escape(QUOTES)
END_PUNCT = ".!?"
END_PUNCT = re.escape(END_PUNCT)
SENT_STARTS_WITH_QUOTE=0


def clean_paragraph(x: str) -> str:
    """various operations to clean the text"""

    # rm whitespace
    x = re.sub(r"\s+", " ", x).strip() # rm whitespace

    # special cleaning operations from screening the data
    # PT example: https://pt.wikipedia.org/wiki/1.%C2%BA_Esquadr%C3%A3o_de_Avi%C3%B5es_de_Intercepta%C3%A7%C3%A3o_e_Ataque
    # EN exmaple: https://en.wikipedia.org/wiki/1964_Illinois_House_of_Representatives_election
    # RegexL first group: [19]:12-13, second group: [19]:cap. 3
    x = re.sub(r"(\]):\s?([\d\-–]+|cap\.?\s?\d+)", r"\1", x) 

    # no space between quotes and punction, and punctuation and quotes
    # NL example https://nl.wikipedia.org/wiki/Sacharias_Jansen
    x = re.sub( rf"([{QUOTES}])\s*([?.!])", r"\1\2", x)
    x = re.sub( rf"([?.!])\s*([{QUOTES}])", r"\1\2", x)

    return x.strip()

def clean_citation(x: str) -> str:
    # rm white space around references: eg [ 19] [ citation needed]
    x = re.sub(r'\[(\s*[\w\s]+\s*)\]', lambda m: f"[{m.group(1).strip()}]", x)
    # clean white space between references
    x = re.sub(r'\]\s+\[', '][', x)
    # clean whitespace between punction and citation
    x = re.sub(r'([^\w\s])\s+\[', r'\1[', x)
    # clean whitespace between citation and punction
    x = re.sub(r'\]\s+([^\w\s])', r']\1', x)
    # insert whitespace betwee reference]-text
    # Example NL https://nl.wikipedia.org/wiki/Sacharias_Jansen
    x = re.sub(r"(\])(\w)", r"\1 \2", x)
    x = re.sub(r"(\w)(\[)", r"\1 \2", x)

    # Ensure that we always have citations __within__ the sentence
    # Eg ... [19]. and not ... .[19]
    # "?"" in the second group as for some language citation needed has a question mark inlcuded, eg pt, nl
    x = re.sub(rf'([{END_PUNCT};:][{QUOTES}]?)(\s*(?:\[[\w\s\?]+\])+)', r'\2\1', x)
    return x.strip()

def ends_with_citation(x):
    """checks wheter a text ends with a citation"""
    x = x.strip()
    
    # we had this before
    # bool(re.search(rf'(?:\[(\d+|\w\s\d+)\])+(\[[\w\s]+\])*[.?!:;{QUOTES}]*$', x))
    return bool(re.search(rf'(\[[\w\d\s]+\])[^\w\d]*$', x))
    

def has_citation(text):
    """checks whether there is any citation in the text"""
    # We consider numeric citations only, but account for citations like [r 1]
    # Example en https://en.wikipedia.org/wiki/1964_Illinois_House_of_Representatives_election
    return bool(re.search(r'\[(\d+|\w+\s\d+)\]', text))

def remove_citations(x):
    """rm every brackets"""
    if not x:
        return None
    # rm [19], and also notes and others
    x = re.sub(r'\[.*?\]', '', x)
    x = x.strip()
    return x

def split_sentences(text: str, nlp: Pipeline):
    """modified splitter based on stanza
    Stanza performs pretty well when the citation is inside the end punctuation.
    Hence, we perform the cleaning and reverse punctuation-citation instances.

    to-do: the correction logic can be improved
    
    @text: this has to be clean text, ie, applying the cleaning functions above first
    @nlp: stanza pipeline
    """
    global SENT_STARTS_WITH_QUOTE
    
    # 1. get all the sentences
    # we do not observe that there are sentences that need further splitting
    doc = nlp(text)
    sentences = [s.text.strip() for s in doc.sentences]

    # 2. Correct if splitting failed, i.e., if a sentence starts with a citation.
    corrected = []
    for s in sentences:
        if bool(re.search(rf'^(\[\d+\])+[{END_PUNCT};:]?$', s)): # sentece is one or more references, eg [2] or [2][3]
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

    if not bool(re.search(rf'[^\w\s]$' , x)):
        return True

    # non-matching brackets, quotes etxc.
    if x.count("(") != x.count(")"):
        return True

    if x.count("[") != x.count("]"):
        return True

    if x.count("{") != x.count("}"):
        return True
    
    if x.count("\'") != x.count("\'"):
        return True

    if x.count("\"") != x.count("\""):
        return True

    if x.count("\"") != x.count("\""):
        return True

    return False

def proc_article(article: dict, nlp: Pipeline):
    """process an article. Critical to clean the paragraph before splitting."""
    
    sents = []
    for section in article['text']:
        # we do not consider the lead section
        if section['header'] == "Lead":
            continue
        for paragraph in section['paragraphs']:
            # cleaning order important!
            # print("\n\n+++++++++++++++\n\n", paragraph)
            paragraph = clean_paragraph(paragraph) # general cleaning
            paragraph = clean_citation(paragraph) # citations
            # print("\n\n\CLEAN+++++++++++++++\n\n", paragraph)

            p_ends_with_citation = ends_with_citation(paragraph)
            p_any_citation = has_citation(paragraph)
            
            # get sentences
            extracted_sents = split_sentences(paragraph, nlp)
            
            prev_sentence = None
            for s in extracted_sents:
                # print(f"<START>{s}<END>")
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
    context = item['context'].strip() if item['context'] else item['context']
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

    for lang in args.languages:
        print("="*10)
        print(f"Running {lang} ...", flush=True)
        print("="*10)
        
        INPUT_FILE = os.path.join(INPUT_DIR, f"{lang}_parsed.jsonl")
        OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"{lang}_sents.json")

        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f][:10]

        # Stanza parser -  run on gpu otherwise this takes years
        nlp = stanza.Pipeline(lang=LANG_MAP[lang], 
                              processors="tokenize", 
                              tokenize_pretokenized=False, 
                              use_gpu=torch.cuda.is_available())

        all_sents = []
        for article in tqdm(data):
            all_sents.extend(proc_article(article, nlp))

        out_sents = []
        for sent in all_sents:
            item = proc_sentence(sent)
            if item:
                out_sents.append(item)

        with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
            json.dump(out_sents, out_f, ensure_ascii=False, indent=2)

        # save number of failed splits
        with open(os.path.join(INFO_DIR, f"sent_{lang}_stats.json"), "w", encoding="utf-8") as f:
            json.dump({"sent_starts_with_quote": SENT_STARTS_WITH_QUOTE}, f, ensure_ascii=False, indent=2)

    
if __name__ == "__main__":
    main()