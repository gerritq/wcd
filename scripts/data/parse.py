import os
import json
from bs4 import BeautifulSoup
from tqdm import tqdm
from skip_sections import DROP_SECTIONS

BASE_DIR = os.getenv("BASE_WCD")
INPUT_PATH = os.path.join(BASE_DIR, "data/raw/htmls")
OUTPUT_PATH = os.path.join(BASE_DIR, "data/raw/parsed")

def clean_paragraphs(paragraphs):
    
    out = []
    for x in paragraphs:
        x = x.strip()
        x = x.replace("\n", "")
        if not x:
            continue
        out.append(x)

    return out

def parse_html(html: str, DROP_SECTIONS_LANG: dict):
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
            # drop if small
            if any(child.name == "small" for child in tag.children if child.name):
                continue

            # get txt
            text = tag.get_text()
            if text:
                current_section["paragraphs"].append(text)

    if current_section["paragraphs"]:
        sections.append(current_section)

    sections_out = []
    for s in sections:
        if s['header'].lower().strip() in DROP_SECTIONS_LANG:
            continue
        s['paragraphs'] = clean_paragraphs(s['paragraphs'])
        sections_out.append(s)

    return sections_out


def main():

    languages  = [
        # "en",  # English
        # "nl",  # Dutch
        # "no",  # Norwegian (Bokmål is 'nb', Nynorsk is 'nn', 'no' redirects to Bokmål)
        # "it",  # Italian
        # "pt",  # Portuguese
        # "ro",  # Romanian
        # "ru",  # Russian
        "uk",  # Ukrainian
        "bg",  # Bulgarian
        # "zh",  # Chinese
        # "ar",  # Arabic
        "id"   # Indonesian
    ]

    for lang in languages:
        print(f"Running {lang} ...", flush=True)

        INPUT_FILE = os.path.join(INPUT_PATH, f"{lang}_htmls.jsonl")
        OUTPUT_FILE = os.path.join(OUTPUT_PATH, f"{lang}_parsed.jsonl")
        
        DROP_SECTIONS_LANG = DROP_SECTIONS[lang]

        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
            # data = data[:100]

        with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
            for x in tqdm(data):
                x['text'] = parse_html(x['raw'], DROP_SECTIONS_LANG)
                del x['raw']
                out_f.write(json.dumps(x) + "\n")
    
if __name__ == "__main__":
    main()
        