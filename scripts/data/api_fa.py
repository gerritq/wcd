import requests
import json
import os
import sys

BASE_DIR = os.getenv("BASE_WCD")
OUTPUT_PATH = os.path.join(BASE_DIR, "data/raw/api")

FA_NAMES = {
    "en": "Featured articles",
    "nl": "Etalage-artikelen",
    "no": "Utmerkede artikler",
    "it": "Voci_in_vetrina_su_it.wiki",
    "pt": "!Artigos destacados",
    "ro": "Wikipedia_articole_de_calitate",
    "ru": "Википедия:Избранные_статьи_по_алфавиту",
    "uk": "Вікіпедія:Вибрані_статті",
    "bg": "Избрани_статии_на_български",
    "zh": "典范条目",
    "ar": "مقالات_مختارة",
    "id": "Artikel_pilihan"
}


# missing for: nl,
GOOD_NAMES = {
    "en": "Good articles",
    "no": "Anbefalte artikler",
    "it": "Voci_di_qualità_su_it.wiki",
    "pt": "!Artigos bons",
    "ro": "Articole_bune",
    "ru": "Википедия:Хорошие_статьи_по_алфавиту",
    "uk": "Вікіпедія:Добрі_статті",
    "bg": "Добри_статии",
    "zh": "优良条目讨论",
    "ar": "مقالات_جيدة",
    "id": "Artikel_bagus"
}

# Categorie:Wikipedia:Etalage-artikelen

CATEGORY_NAMES = {
    "en": "Category",
    "nl": "Categorie:Wikipedia",
    "no": "Kategori",
    "it": "Categoria",
    "pt": "Categoria",
    "ro": "Categorie",
    "ru": "Категория",
    "uk": "Категорія",
    "bg": "Категория",
    "zh": "Category",
    "ar": "تصنيف",
    "id": "Kategori"
}

def query_high_quality_articles(lang: str, QUALITY_NAMES: dict):
    quality = "good" if QUALITY_NAMES['en'].startswith("Good") else "fa"
    S = requests.Session()
    URL = f"https://{lang}.wikipedia.org/w/api.php"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    titles = []
    params = {
        "action": "query",
        "format": "json",
        "list": "categorymembers",
        "cmtitle": f"{CATEGORY_NAMES[lang]}:{QUALITY_NAMES[lang]}",
        # "cmtitle": f"Category:Featured_articles",
        # "cmtitle": f"Kategoria:Artykuły_na_Medal", # use this for polish; only version that worked
        "cmlimit": 500,
    }

    while True:
        response = S.get(url=URL, params=params, headers=headers)
        data = response.json()
        # print(data)
        for page in data["query"]["categorymembers"]:
            titles.append({"title": page["title"]})

        if "continue" in data:
            params["cmcontinue"] = data["continue"]["cmcontinue"]
        else:
            break

    OUT_FILE = os.path.join(OUTPUT_PATH, f"{lang}_{quality}.jsonl")
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for item in titles:
            f.write(json.dumps(item) + "\n")

    print(f"Saved {len(titles)} {quality} titles for {lang}")
    return titles

def main():
    languages  = [
    # "en",  # English
    # "nl",  # Dutch
    # "no",  # Norwegian (Bokmål is 'nb', Nynorsk is 'nn', 'no' redirects to Bokmål)
    # "it",  # Italian
    #  "pt",  # Portuguese
    #  "ro",  # Romanian
    #   "ru",  # Russian
    #   "uk",  # Ukrainian
    #   "bg",  # Bulgarian
    #  "zh",  # Chinese
    #  "ar",  # Arabic
      "id"   # Indonesian
    ]
    for lang in languages:
        print(f"Running FA {lang} ...")

        # FA
        query_high_quality_articles(lang, FA_NAMES)

        # Good articles
        if lang in GOOD_NAMES:
            print(f"Running Good {lang} ...")
            query_high_quality_articles(lang, GOOD_NAMES)

if __name__ == "__main__":
    
    
    main()