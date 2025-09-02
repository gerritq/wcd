import requests
import json
from tqdm import tqdm
import sys

LANG=sys.argv[1]
INPUT_PATH = f"../../data/raw/{LANG}_titles.jsonl"
OUTPUT_PATH = f"../../data/raw/{LANG}_raw.jsonl"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}


def get_wikitext(title):
    URL = f"https://{LANG}.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "prop": "revisions",
        "rvprop": "content",
        "rvslots": "main",
        "titles": title
    }
    r = requests.get(URL, params=params, headers=headers).json()
    page = next(iter(r["query"]["pages"].values()))
    return page["revisions"][0]["slots"]["main"]["*"]

def get_html(title):
    url = f"https://{LANG}.wikipedia.org/w/api.php"
    params = {
        "action": "parse",
        "page": title,
        "format": "json",
        "prop": "text",
    }

    response = requests.get(url, params=params, headers=headers)
    data = response.json()

    html = data["parse"]["text"]["*"]
    return html

def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        titles = [json.loads(line)["title"] for line in f]
        # titles = titles[:2]

    out = []
    for title in tqdm(titles):
            text = get_html(title)
            out.append({"title": title, "raw": text})

    with open(OUTPUT_PATH, "w", encoding="utf-8") as out_f:
        for x in out:
            out_f.write(json.dumps(x, ensure_ascii=False) + "\n")

    print(f"Saved {len(titles)} articles to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()