import os
import re
import json 
from nltk.tokenize import sent_tokenize
import sys

import nltk
nltk.download('punkt_tab')

LANG=sys.argv[1]
BASE_DIR = os.getenv("BASE_WCD", ".")
INPUT_PATH = os.path.join(BASE_DIR, f"data/raw/{LANG}_txt.jsonl") 
OUTPUT_PATH = os.path.join(BASE_DIR, f"data/proc/{LANG}_sents.jsonl")

DROP_SECTIONS={'en': ['references', 'list', 'see also', 'notes', 'bibliography', 'further reading', 'external links', 'discography', 'filmography'],
               'pt': ["referências", "lista", "ver também", "notas", "bibliografia", "leitura adicional", "ligações externas", "discografia", "filmografia", "videografia"],
               'hu': ['hivatkozások','jegyzetek','lásd még','bibliográfia','további irodalom','külső hivatkozások','discográfia','filmográfia'],
               'pl': ['przypisy', 'uwagi', 'zobacz też', 'bibliografia', 'dalsza lektura', 'linki zewnętrzne', 'dyskografia', 'filmografia']}
DROP_SECTIONS = DROP_SECTIONS[LANG]

def txt_ends_with_citation(txt):
    """is this d or w in the regex?"""
    return bool(re.search(r'(?:\[\w+\])+$', txt.rstrip())) or bool(re.search(r'(?:\[\w+\])+[.?!]"?$', txt.rstrip()))

def has_citation(text):
    return bool(re.search(r'\[\d+\]', text))

def remove_citations(text):
    return re.sub(r'\[\w+\]', '', text) # also rm notes, and others

def modify_sent_tokenise(text):
    sentences = sent_tokenize(text)

    for i in range(1,len(sentences)):
        sent = sentences[i].strip()
        citations = re.match(r'^((?:\[\w+\])+)', sent)
        if citations:
            citations = citations.group(1)
            sentences[i-1] = sentences[i-1] + citations
            sentences[i] = sent[len(citations):].lstrip()
        
    return [x for x in sentences if x]
    
def proc_article(article):
    sents = []
    for section in article['text']:
        if section['header'] == "Lead":
            continue
        for paragraph in section['paragraphs']:
            p_ends_with_citation = txt_ends_with_citation(paragraph)
            p_any_citation = has_citation(paragraph)
            extracted_sents = modify_sent_tokenise(paragraph)
            for s in extracted_sents:
                sents.append({'title': article['title'],
                              'section': section['header'],
                              'p_ends_with_citation': p_ends_with_citation,
                              'p_any_citation': p_any_citation,
                              'sentence': s})

    return sents

def proc_sentence(item):
    sentence = item['sentence'].strip()
    citation = txt_ends_with_citation(sentence) # has_citation(sentence)

    sentence_clean = remove_citations(sentence)


    if (
        any(section in item['section'].lower() for section in DROP_SECTIONS) or
        (len(sentence_clean.split()) < 4) or
        (len(sentence_clean) < 15) or
        (sentence_clean[0].isalpha() and not sentence_clean[0].isupper()) or
        (not bool(re.search(r'[.!?]"?$' , sentence_clean)))
        
    ):
        return None
    
    # strict label
    if not citation and not item['p_any_citation']:
        label = 0
    elif citation:
        label = 1
    else:
        return None

    item.update({"has_citation": citation,
                 "claim": sentence_clean,
                 'label': label})
    return item

def main():

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    all_sents = []
    for article in data:
        all_sents.extend(proc_article(article))

    out_sents = []
    for sent in all_sents:
        item = proc_sentence(sent)
        if item:
            out_sents.append(item)


    
    with open(OUTPUT_PATH, "w", encoding="utf-8") as out_f:
        for x in out_sents:
            out_f.write(json.dumps(x, ensure_ascii=False) + "\n")

    # x = data[0]
    # p = x['text'][3]['paragraphs'][1]
    
    # print(p)
    # print(paragraph_ends_with_citation(p))
    # # print(sent_tokenize(p))
    # print(modify_sent_tokenise(p))
    
    
if __name__ == "__main__":
    main()
        