import requests
import json
import os
import sys

BASE_DIR = os.getenv("BASE_WCD")
OUTPUT_PATH = os.path.join(BASE_DIR, "data/raw/api")

"""
- id: good articles are only half of the mentioned size, as for each article the talk page is mentioned too
"""

# (number from the featured/good articles page; not the category)

FA_NAMES = {
    "en": "Featured articles", # (6,799) https://en.wikipedia.org/wiki/Category:Wikipedia_featured_articles
    "nl": "Etalage-artikelen", # (380) https://nl.wikipedia.org/wiki/Categorie:Wikipedia:Etalage-artikelen
    "no": "Utmerkede artikler", # (351) https://no.wikipedia.org/wiki/Kategori:Utmerkede_artikler
    "it": "Voci_in_vetrina_su_it.wiki", # (583) https://it.wikipedia.org/wiki/Categoria:Voci_in_vetrina_su_it.wiki
    "pt": "!Artigos destacados", # (1512) https://pt.wikipedia.org/wiki/Categoria:!Artigos_destacados
    "ro": "Articole_de_calitate", # (199) https://ro.wikipedia.org/wiki/Categorie:Articole_bune
    "ru": "Википедия:Избранные_статьи_по_алфавиту", # (2072) https://ru.wikipedia.org/wiki/%D0%9A%D0%B0%D1%82%D0%B5%D0%B3%D0%BE%D1%80%D0%B8%D1%8F:%D0%92%D0%B8%D0%BA%D0%B8%D0%BF%D0%B5%D0%B4%D0%B8%D1%8F:%D0%98%D0%B7%D0%B1%D1%80%D0%B0%D0%BD%D0%BD%D1%8B%D0%B5_%D1%81%D1%82%D0%B0%D1%82%D1%8C%D0%B8_%D0%BF%D0%BE_%D0%B0%D0%BB%D1%84%D0%B0%D0%B2%D0%B8%D1%82%D1%83
    "uk": "Вікіпедія:Вибрані_статті", # (246) https://uk.wikipedia.org/wiki/%D0%9A%D0%B0%D1%82%D0%B5%D0%B3%D0%BE%D1%80%D1%96%D1%8F:%D0%92%D1%96%D0%BA%D1%96%D0%BF%D0%B5%D0%B4%D1%96%D1%8F:%D0%92%D0%B8%D0%B1%D1%80%D0%B0%D0%BD%D1%96_%D1%81%D1%82%D0%B0%D1%82%D1%82%D1%96
    "bg": "Избрани_статии_на_български", # (153) https://bg.wikipedia.org/wiki/%D0%9A%D0%B0%D1%82%D0%B5%D0%B3%D0%BE%D1%80%D0%B8%D1%8F:%D0%98%D0%B7%D0%B1%D1%80%D0%B0%D0%BD%D0%B8_%D1%81%D1%82%D0%B0%D1%82%D0%B8%D0%B8_%D0%BD%D0%B0_%D0%B1%D1%8A%D0%BB%D0%B3%D0%B0%D1%80%D1%81%D0%BA%D0%B8
    "id": "Artikel_pilihan", # (421) https://id.wikipedia.org/wiki/Kategori:Artikel_bagus
    "vi": "Bài_viết_chọn_lọc", # (479) https://vi.wikipedia.org/wiki/Th%E1%BB%83_lo%E1%BA%A1i:B%C3%A0i_vi%E1%BA%BFt_ch%E1%BB%8Dn_l%E1%BB%8Dc
    "tr": "Seçkin_maddeler" # (228) https://tr.wikipedia.org/wiki/Kategori:Se%C3%A7kin_maddeler
}   


# missing for: nl,
GOOD_NAMES = {
    "en": "Good articles", # (42,623) https://en.wikipedia.org/wiki/Category:Wikipedia_good_articles
    "no": "Anbefalte artikler", # (1051) https://no.wikipedia.org/wiki/Kategori:Anbefalte_artikler
    "it": "Voci_di_qualità_su_it.wiki", # (569) https://it.wikipedia.org/wiki/Categoria:Voci_di_qualit%C3%A0_su_it.wiki
    "pt": "!Artigos bons", # (1923) https://pt.wikipedia.org/wiki/Categoria:!Artigos_bons
    "ro": "Articole_bune", # (495) https://ro.wikipedia.org/wiki/Categorie:Articole_bune
    "ru": "Википедия:Хорошие_статьи_по_алфавиту", # (5002) https://ru.wikipedia.org/wiki/%D0%9A%D0%B0%D1%82%D0%B5%D0%B3%D0%BE%D1%80%D0%B8%D1%8F:%D0%92%D0%B8%D0%BA%D0%B8%D0%BF%D0%B5%D0%B4%D0%B8%D1%8F:%D0%A5%D0%BE%D1%80%D0%BE%D1%88%D0%B8%D0%B5_%D1%81%D1%82%D0%B0%D1%82%D1%8C%D0%B8_%D0%BF%D0%BE_%D0%B0%D0%BB%D1%84%D0%B0%D0%B2%D0%B8%D1%82%D1%83
    "uk": "Вікіпедія:Добрі_статті", # (1077) https://uk.wikipedia.org/wiki/%D0%9A%D0%B0%D1%82%D0%B5%D0%B3%D0%BE%D1%80%D1%96%D1%8F:%D0%92%D1%96%D0%BA%D1%96%D0%BF%D0%B5%D0%B4%D1%96%D1%8F:%D0%94%D0%BE%D0%B1%D1%80%D1%96_%D1%81%D1%82%D0%B0%D1%82%D1%82%D1%96
    "bg": "Добри_статии", # (59) https://bg.wikipedia.org/wiki/%D0%9A%D0%B0%D1%82%D0%B5%D0%B3%D0%BE%D1%80%D0%B8%D1%8F:%D0%94%D0%BE%D0%B1%D1%80%D0%B8_%D1%81%D1%82%D0%B0%D1%82%D0%B8%D0%B8
    "id": "Artikel_bagus", # (431) https://id.wikipedia.org/wiki/Kategori:Artikel_bagus
    "vi": "Bài_viết_chất_lượng_tốt", # (566) https://vi.wikipedia.org/wiki/Th%E1%BB%83_lo%E1%BA%A1i:B%C3%A0i_vi%E1%BA%BFt_ch%E1%BA%A5t_l%C6%B0%E1%BB%A3ng_t%E1%BB%91t
    "tr": "Kaliteli_maddeler" # (344) https://tr.wikipedia.org/wiki/Kategori:Se%C3%A7kin_maddeler
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
    "id": "Kategori",
    "vi": "Thể_loại",
    "tr": "Kategori"
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
    # "pt",  # Portuguese
    # "ro",  # Romanian
    # "ru",  # Russian
    # "uk",  # Ukrainian
    # "bg",  # Bulgarian
    "id",  # Indonesian
    "vi",  # Vietnamese
    "tr"  # Turkish
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