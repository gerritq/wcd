import os
import json
import argparse
from bs4 import BeautifulSoup
from tqdm import tqdm
from skip_sections import DROP_SECTIONS
import unicodedata

BASE_DIR = os.getenv("BASE_WCD")
INPUT_PATH = os.path.join(BASE_DIR, "data/raw/htmls")
OUTPUT_PATH = os.path.join(BASE_DIR, "data/raw/parsed")
INFO_PATH = os.path.join(BASE_DIR, "data/info/parsing_stats.json")

parser = argparse.ArgumentParser()
parser.add_argument("--languages", nargs="+", required=True)
args = parser.parse_args()


def clean_txt(paragraphs):
    
    out = []
    for x in paragraphs:
        x = unicodedata.normalize("NFKC", x)
        x = x.strip()
        x = x.replace("\n", "")
        if not x:
            continue
        out.append(x)

    return out

def parse_html(html: str, DROP_SECTIONS_LANG: list, CITATION_NEEDED_LANG: str):
    soup = BeautifulSoup(html, 'lxml')
    
    sections = []
    current_section = {"header": "Lead", "paragraphs": []}

    for tag in soup.find_all(['h2', 'h3', 'h4', 'h5', 'h6','p']):
        if tag.name in ['h2', 'h3', 'h4', 'h5', 'h6']:
            # begin a new sec
            if current_section["paragraphs"]:
                sections.append(current_section)
            current_section = {"header": tag.get_text(), "paragraphs": []}
        
        # rm block quote
        elif tag.name == 'p':
            # drop if blockquote
            if tag.find_parent("blockquote"):
                continue  
            # drop if other quote
            if tag.find_parent("div", class_="templatequotecite"):
                continue
            if tag.find_parent("div", class_="poem"): # for real?
                continue
            if tag.find_parent("div", class_="quote"): # for real?
                continue
            # drop if small
            for small in tag.find_all("small"):
                small.decompose()
            # this drops latex math notation
            for ann in tag.find_all("annotation"):
                if "\\displaystyle" in ann.get_text():
                    ann.decompose()

            # get txt
            text = tag.get_text()  
            if text:
                current_section["paragraphs"].append(text)

    if current_section["paragraphs"]:
        sections.append(current_section)

    sections_out = []
    for s in sections:
        # clean header
        s['header'] = unicodedata.normalize("NFKC", s['header'])
        if s['header'].lower().strip() in DROP_SECTIONS_LANG:
            continue
        s['paragraphs'] = clean_txt(s['paragraphs'])

        # We drop paragraphs in which there is a [citation needed] statement
        if (CITATION_NEEDED_LANG and CITATION_NEEDED_LANG.lower() in " ".join(s['paragraphs']).lower()):
            continue

        sections_out.append(s)

    return sections_out


def main():

    # do lower case
    CITATION_NEEDED = {"en": "[citation needed]",
                        "nl": "[Bron?]",
                        "no": "[trenger referanse]",
                        "it": "[senza fonte]",
                        "pt": "[carece de fontes]",
                        "ro": "[necesită citare]",
                        "ru": "[источник?]",
                        "uk": None,
                        "bg": None,
                        "sr": "[Činjenica-lat]",
                        "id": "[butuh rujukan]",
                        "vi": "[cần dẫn nguồn]",
                        "tr": "[kaynak belirtilmeli]",
                        "sq": "[nevojitet citimi]",
                        "mk": None,
                        "hy": None,
                        "az": None,
                        "de": None,
                        "zh": "[来源请求]",
                        "uz": "[manba kerak]",
                        "th": "[ต้องการอ้างอิง]",
                        }
    stats = {}

    for lang in args.languages:
        print(f"Running {lang} ...", flush=True)

        INPUT_FILE = os.path.join(INPUT_PATH, f"{lang}_htmls.jsonl")
        OUTPUT_FILE = os.path.join(OUTPUT_PATH, f"{lang}_parsed.jsonl")
        
        DROP_SECTIONS_LANG = DROP_SECTIONS[lang]
        CITATION_NEEDED_LANG = CITATION_NEEDED[lang]

        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
            # data = data[:100]
            # data = [x for x in data if x['title'] == "Teoria relativității generale"]

        dropped = 0
        parsed = 0

        with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
            for x in tqdm(data):
                parsed_text = parse_html(x['raw'], DROP_SECTIONS_LANG, CITATION_NEEDED_LANG)
                if not parsed_text:
                    dropped += 1
                    continue
                else:
                    parsed += 1
                    x['text'] = parsed_text
                    del x['raw']
                    out_f.write(json.dumps(x) + "\n")

        stats[lang] = {"parsed": parsed, "dropped": dropped}

    # save stats
    with open(INFO_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
if __name__ == "__main__":
    main()
        