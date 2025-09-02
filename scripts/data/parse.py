import json
from bs4 import BeautifulSoup
from tqdm import tqdm
import sys

LANG=sys.argv[1]
INPUT_PATH = f"../../data/raw/{LANG}_raw.jsonl"
OUTPUT_PATH = f"../../data/raw/{LANG}_txt.jsonl"

def clean_paragraphs(paragraphs):
    
    out = []
    for x in paragraphs:
        x = x.strip()
        x = x.replace("\n", "")
        if not x:
            continue
        out.append(x)

    return out


def parse_html(html):
    soup = BeautifulSoup(html, 'html.parser')
    
    sections = []
    current_section = {"header": "Lead", "paragraphs": []}

    for tag in soup.find_all(['h2', 'h3', 'h4', 'h5', 'h6','p']):
        if tag.name in ['h2', 'h3', 'h4', 'h5', 'h6']:
            # begin a new sec
            if current_section["paragraphs"]:
                sections.append(current_section)
            current_section = {"header": tag.get_text(), "paragraphs": []}
        elif tag.name == 'p':
            if tag.find_parent("blockquote"):
                continue  
            text = tag.get_text()
            if text:
                current_section["paragraphs"].append(text)

    if current_section["paragraphs"]:
        sections.append(current_section)

    for x in sections:
        x['paragraphs'] = clean_paragraphs(x['paragraphs'])

    return sections


def main():

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    with open(OUTPUT_PATH, "w", encoding="utf-8") as out_f:
        for x in tqdm(data):
            x['text'] = parse_html(x['raw'])
            del x['raw']
            out_f.write(json.dumps(x) + "\n")
    
if __name__ == "__main__":
    main()
        