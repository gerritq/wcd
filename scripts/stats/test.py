import stanza
import os

def contains_propn(x: str, tag: str, nlp) -> int:
    doc = nlp(x)
    for sent in doc.sentences:
        for word in sent.words:
            if word.upos == tag:
                return 1
    return 0

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

lang = "en"
nlp = stanza.Pipeline(lang=LANG_MAP[lang], processors='tokenize,pos', model_dir=os.getenv('HF_HOME'))
print(contains_propn("Obama is the president.", "PROPN", nlp))
print(contains_propn("This has no real noun.", "PROPN", nlp))