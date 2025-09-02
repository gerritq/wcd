import requests
import json
import os
import sys

LANG=sys.argv[1]

OUTPUT_PATH = f"../../data/raw/{LANG}_titles.jsonl"
FA_NAMES = {
    "en": "Featured articles",
    "pt": "Artigos destacados",
    "pl": "Artykuły_na_Medal",
    "hu": "Kiemelt cikkek",
}

CATEGORY_NAMES = {
    "en": "Category",
    "pt": "Categoria",
    "pl": "Kategoria",
    "hu": "Kategória",
}
def main(limit=500):
    S = requests.Session()
    URL = f"https://{LANG}.wikipedia.org/w/api.php"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    titles = []
    params = {
        "action": "query",
        "format": "json",
        "list": "categorymembers",
        # "cmtitle": f"{CATEGORY_NAMES[LANG]}:{FA_NAMES[LANG]}",
        # "cmtitle": f"Category:Featured_articles",
        # "cmtitle": f"Kategoria:Artykuły_na_Medal", # use this for polish; only version that worked
        "cmlimit": limit,
    }

    print(params)
    while True:
        response = S.get(url=URL, params=params, headers=headers)
        data = response.json()
        print(data)

        for page in data["query"]["categorymembers"]:
            titles.append({"title": page["title"]})

        if "continue" in data:
            params["cmcontinue"] = data["continue"]["cmcontinue"]
        else:
            break

    os.makedirs("../../data/raw", exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for item in titles:
            f.write(json.dumps(item) + "\n")

    print(f"Saved {len(titles)} FA titles for {LANG}")
    return titles

if __name__ == "__main__":
    main()