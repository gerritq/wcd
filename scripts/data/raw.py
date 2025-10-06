import os
import requests
import json
from tqdm import tqdm
import sys

BASE_DIR = os.getenv("BASE_WCD")
INPUT_PATH = os.path.join(BASE_DIR, "data/raw/api")
OUTPUT_PATH = os.path.join(BASE_DIR, "data/raw/htmls")

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}


def get_wikitext(lang: str, title: str):
    URL = f"https://{lang}.wikipedia.org/w/api.php"
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

def get_html(lang: str, title: str):
    url = f"https://{lang}.wikipedia.org/w/api.php"
    params = {
        "action": "parse",
        "page": title,
        "format": "json",
        "prop": "text",
    }
    try:
        response = requests.get(url, params=params, headers=headers, timeout=30)
        response.raise_for_status()  # raises if HTTP code != 200
        data = response.json()
        return data["parse"]["text"]["*"]
    except Exception as e:
        print(f"Failed to fetch {url}: {e}")
        return None

def main():
    languages  = [
        "en",  # English
        "nl",  # Dutch
        "no",  # Norwegian (Bokmål is 'nb', Nynorsk is 'nn', 'no' redirects to Bokmål)
        "it",  # Italian
        "pt",  # Portuguese
        "ro",  # Romanian
        # "ru",  # Russian
        # "uk",  # Ukrainian
        # "bg",  # Bulgarian
        # "zh",  # Chinese
        # "ar",  # Arabic
        # "id"   # Indonesian
    ]

    for lang in languages:
        print(f"Running {lang} ...", flush=True)

        INPUT_FILE = os.path.join(INPUT_PATH, f"{lang}_all.jsonl")
        OUTPUT_FILE = os.path.join(OUTPUT_PATH, f"{lang}_htmls.jsonl")
        
        
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]

        # can restrict here
        if lang == "en":
            data = [x for x in data if x['source'] == 'fa']
            print(f'Len data {len(data)}')

        if lang in ["zh", "ru", "pt"]:
            data = [x for x in data if x['source'] != 'views']
            print(f'Len data {len(data)}')

        # data = data[:1000]

        processed_titles = set()
        if os.path.exists(OUTPUT_FILE):
            with open(OUTPUT_FILE, "r", encoding="utf-8") as out_f:
                for line in out_f:
                    try:
                        entry = json.loads(line)
                        processed_titles.add(entry["title"])
                    except Exception:
                        continue
            print(f"Skipping {len(processed_titles)} already processed articles")

        with open(OUTPUT_FILE, "a", encoding="utf-8") as out_f:
            for item in tqdm(data):
                if item['title'] in processed_titles:
                    continue
                raw = get_html(lang, item['title'])
                if not raw:
                    continue
                item.update({"raw": raw})
                out_f.write(json.dumps(item, ensure_ascii=False) + "\n")

        print(f"Saved {len(data)} articles to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()