import requests
import json
import os
import sys

BASE_DIR = os.getenv("BASE_WCD", ".")
OUTPUT_PATH = os.path.join(BASE_DIR, "data/raw/api")

def query_random_articles(lang: str, n: int):
    
    S = requests.Session()
    URL = f"https://{lang}.wikipedia.org/w/api.php"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    titles = []
    rnlimit = 500
    fetched = 0

    while fetched < n:
        params = {
            "action": "query",
            "format": "json",
            "list": "random",
            "rnnamespace": 0,
            "rnlimit": min(rnlimit, n - fetched)
        }

        response = S.get(url=URL, params=params, headers=headers)
        data = response.json()
        if "query" not in data or "random" not in data["query"]:
            print(f"Error fetching for {lang}: {data}")
            break

        for page in data["query"]["random"]:
            titles.append({"title": page["title"]})

        fetched += len(data["query"]["random"])
        print(f"[{lang}] Fetched {fetched}/{n}")

    OUT_FILE = os.path.join(OUTPUT_PATH, f"{lang}_random.jsonl")
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for item in titles:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return titles


def main():
    languages  = [
    # "en",  # English
    # "nl",  # Dutch
    # "no",  # Norwegian 
    # "it",  # Italian
    # "pt",  # Portuguese
    # "ro",  # Romanian
    # "ru",  # Russian
    # "uk",  # Ukrainian
    # "bg",  # Bulgarian
    "id",
    "vi",
    "tr",

    ]
    for lang in languages:
        print(f"\nQuerying random articles for {lang} ===")
        query_random_articles(lang, n=2500)  # adjust `n` if needed


if __name__ == "__main__":
    main()