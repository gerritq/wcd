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

    # special cleaning operations from screening the data
    # PT example: https://pt.wikipedia.org/wiki/1.%C2%BA_Esquadr%C3%A3o_de_Avi%C3%B5es_de_Intercepta%C3%A7%C3%A3o_e_Ataque
    # EN exmaple: https://en.wikipedia.org/wiki/1964_Illinois_House_of_Representatives_election
    # RegexL first group: [19]:12-13, second group: [19]:cap. 3
    x = re.sub(r"(\]):\s?([\d\-–]+|cap\.?\s?\d+)", r"\1", x) 

    # no space between quotes and punction, and punctuation and quotes
    # NL example https://nl.wikipedia.org/wiki/Sacharias_Jansen
    x = re.sub( rf"([{QUOTES}])\s*([?.!])", r"\1\2", x)
    x = re.sub( rf"([?.!])\s*([{QUOTES}])", r"\1\2", x)

    # no space between word and punct
    x = re.sub( rf"(\w)\s+([{END_PUNCT}{QUOTES};:,\)])", r"\1\2", x)

    # add a space between a punctuation and word
    # x = re.sub(rf"([{END_PUNCT}{QUOTES};:,\)])(\w{4,})", r"\1 \2", x)
    x = re.sub(rf"(\.)(\w{4,}?)", r"\1 \2", x)

    # rm templates. There should be none but we found rare instances
    # Eg Romanian article "Doggerbank "
    x = re.sub(r"\(?\{{2}[\w\s\=\|\:]+\}{2}\)?", "", x)

    # rm tags and links; very rare but exist
    x = re.sub(r"\<.*?\>", "", x)
    x = re.sub(r"http\S+", "", x)

    # rm whitespace
    x = re.sub(r"\s+", " ", x).strip() # rm whitespace

    return x.strip()

def clean_citation(x: str) -> str:
    # rm white space around references: eg [ 19] [ citation needed]
    x = re.sub(r'\[(\s*[\w\s]+\s*)\]', lambda m: f"[{m.group(1).strip()}]", x)
    # clean white space and other punct between references
    # Example for non-white space: https://ro.wikipedia.org/wiki/Soporul_de_C%C3%A2mpie,_Cluj
    # x = re.sub(r'\]\s+\[', '][', x) # this is white space only
    x = re.sub(r'\][^\w\d]+\[', '][', x)
    # clean whitespace between punction and citation
    x = re.sub(r'([^\w\s])\s+\[', r'\1[', x)
    # clean whitespace between citation and punction
    x = re.sub(r'\]\s+([^\w\s])', r']\1', x)
    # insert whitespace betwee reference]-text
    # Example NL https://nl.wikipedia.org/wiki/Sacharias_Jansen
    x = re.sub(r"([\]\)])(\w)", r"\1 \2", x)

    # Ensure that we always have citations __within__ the sentence
    # Eg ... [19]. and not ... .[19]
    # "?"" in the second group as for some language citation needed has a question mark inlcuded, eg pt, nl
    x = re.sub(rf'([{END_PUNCT};:][{QUOTES}]?)(\s*(?:\[[\w\s\?]+\])+)', r'\2\1', x)
    return x.strip()

def final_clean(x):

    # no space between word and punct
    x = re.sub( rf"(\w)\s+([{END_PUNCT}{QUOTES};:,\)])", r"\1\2", x)

    # rm multiple dots at the end
    x = re.sub(r"\.{2,}$", r".", x)
    
    # rm whitespace
    x = re.sub(r"\s+", " ", x).strip() # rm whitespace

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
    """rm every brackets. comes at the cost of also rm brackets in text, but we want to be sure everything is rm."""
    if not x:
        return None
    
    placeholder = "___ELLIPSIS_PLACEHOLDER___"
    x = x.replace("[...]", placeholder)
    
    # rm [19], and also notes and others
    x = re.sub(r'\[.*?\]', '', x)
    
    x = x.replace(placeholder, "[...]")
    
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
    i = 0
    n = len(sentences)

    while i < n:
        s = sentences[i].strip()

        # sentence is only references like [2] or [2][3]
        if re.search(rf'^(\[\d+\])+[\s{END_PUNCT};:]?$', s):
            if corrected:
                corrected[-1] = corrected[-1].rstrip() + s
            i += 1
            continue

        # here we try to merge when sentence which is not last end with a colon ":"
        if s.strip().endswith(":") and i < n - 1:
            merged = s.rstrip() + " " + sentences[i + 1].lstrip()
            corrected.append(merged)
            i += 2
            continue

        # try to correct for cases where the paranthesis got split
        # if this leads to odd str, we use heuristics to drop them later
        if s.strip().startswith("(") and i < n - 1:
            if corrected:
                corrected[-1] = corrected[-1].rstrip() + " " + s.strip()
            i += 1
            continue


        corrected.append(s)
        i += 1

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

    # this happens sometimes when paranthesis are not split correctly
    if x.startswith("("):
        return True

    # non-matching brackets, quotes etxc.
    if x.count("(") != x.count(")"):
        return True

    if x.count("[") != x.count("]"):
        return True

    if x.count("{") != x.count("}"):
        return True

    # if x.count("\"") % 2 != 0:
    #     return True

    if x.count("„") != x.count("“"):
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
                              "topic": article['topic'],
                              'section': section['header'],
                              'source': article['source'],
                              'context':  prev_sentence,
                              'p_ends_with_citation': p_ends_with_citation,
                              'p_any_citation': p_any_citation,
                              'sentence': s,
                              })
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

    # final clean
    sentence_clean = final_clean(sentence_clean)
    if context_clean:
        context_clean = final_clean(context_clean)
    
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
            data = [json.loads(line) for line in f]
            # data = [x for x in data if x['title'] == "Hobbitul"]

        # Stanza parser -  run on gpu otherwise this takes years
        nlp = stanza.Pipeline(lang=LANG_MAP[lang], 
                              processors="tokenize", 
                              tokenize_pretokenized=False, 
                              use_gpu=torch.cuda.is_available(),
                              dir=os.getenv('HF_HOME'),
                              model_dir=os.getenv('HF_HOME'))

        all_sents = []
        for article in tqdm(data):
            all_sents.extend(proc_article(article, nlp))

        out_sents = []
        seen_claims = set()
        for sent in all_sents:
            item = proc_sentence(sent)
            if item:
                claim = item['claim']
                if claim not in seen_claims:
                    out_sents.append(item)
                    seen_claims.add(claim)

        with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
            json.dump(out_sents, out_f, ensure_ascii=False, indent=2)

        # save number of failed splits
        with open(os.path.join(INFO_DIR, f"sent_{lang}_stats.json"), "w", encoding="utf-8") as f:
            json.dump({"sent_starts_with_quote": SENT_STARTS_WITH_QUOTE}, f, ensure_ascii=False, indent=2)

    
if __name__ == "__main__":
    main()