import os
import json
from bs4 import BeautifulSoup
from tqdm import tqdm
from skip_sections import DROP_SECTIONS

BASE_DIR = os.getenv("BASE_WCD")
INPUT_PATH = os.path.join(BASE_DIR, "data/raw/htmls")
OUTPUT_PATH = os.path.join(BASE_DIR, "data/raw/parsed")
INFO_PATH = os.path.join(BASE_DIR, "data/info/parsing_stats.json")

def clean_paragraphs(paragraphs):
    
    out = []
    for x in paragraphs:
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
        if (CITATION_NEEDED_LANG and CITATION_NEEDED_LANG in " ".join(s['paragraphs'])):
            return None

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
                        "id": None,
                        }

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
        # "zh",  # Chinese
        # "ar",  # Arabic
        "id"   # Indonesian
    ]

    stats = {}

    for lang in languages:
        print(f"Running {lang} ...", flush=True)

        INPUT_FILE = os.path.join(INPUT_PATH, f"{lang}_htmls.jsonl")
        OUTPUT_FILE = os.path.join(OUTPUT_PATH, f"{lang}_parsed.jsonl")
        
        DROP_SECTIONS_LANG = DROP_SECTIONS[lang]
        CITATION_NEEDED_LANG = CITATION_NEEDED[lang]

        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
            # data = data[:100]

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
        