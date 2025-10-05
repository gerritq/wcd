import os
import requests
from datetime import date, timedelta
from collections import Counter
import json

# https://doc.wikimedia.org/generated-data-platform/aqs/analytics-api/reference/page-views.html


"""
- should rm list an others
- removing everything with : 
-- also rm all movies which is desirable as we saw they often contain lots of claims wo citation which would need one
-- but excludes all wiki pages (Wikipedia, Speical, Help etc.) in all languages
"""

BASE_DIR = os.getenv("BASE_WCD")
DATA_DIR = os.path.join(BASE_DIR, "data/raw/api")


def top_n_articles(lang_code, n, access="all-access"):
    base = "https://wikimedia.org/api/rest_v1/metrics/pageviews/top"
    project = f"{lang_code}.wikipedia"
    days=60 
    access="all-access"
    # title_exclude = ["Special:", "Talk:", "User:", "File:", "Category:", "Template:", "Help:"]
    today = date.today()
    titles = {}

    i=0
    for d in (today - timedelta(i+1) for i in range(days)):
        url = f"{base}/{project}/{access}/{d.year}/{d.month:02d}/{d.day:02d}"
        print(f"Iteration {i}:", url)
        try:
            r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
            if r.status_code != 200:
                continue
            items = r.json().get("items", [])
            if not items:
                continue
            for entry in items[0].get("articles", []):
                title = entry["article"]
                views = entry["views"]
                if ":" in title:
                    print(f"Found colon: {title}")
                    continue
                if title in ("Main_Page",):
                    continue
                titles[title] = views
        except Exception as e:
            print(e)
            continue

        if len(titles) > n:
            break

        i+=1
    return titles

def main():
    n_min=5000
    languages  = [
    "en",  # English
    # "nl",  # Dutch
    # "no",  # Norwegian (Bokmål is 'nb', Nynorsk is 'nn', 'no' redirects to Bokmål)
    #  "it",  # Italian
    #  "pt",  # Portuguese
    #  "ro",  # Romanian
    #  "ru",  # Russian
    #  "uk",  # Ukrainian
    #  "bg",  # Bulgarian
    #  "zh",  # Chinese
    #  "ar",  # Arabic
    #  "id"   # Indonesian
    ]
    for lang in languages:
        print(f"Running {lang} ...")
        results = top_n_articles(lang, n=n_min)
        save_file = os.path.join(DATA_DIR, f"{lang}_views_{n_min}.jsonl")

        with open(save_file, "w", encoding="utf-8") as f:
            for title, views in results.items():
                record = {"title": title, "views": views, "lang": lang}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()