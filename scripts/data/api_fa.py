import requests
import json
import os
import sys
import time

BASE_DIR = os.getenv("BASE_WCD")
OUTPUT_PATH = os.path.join(BASE_DIR, "data/raw/api")

"""
- id: good articles are only half of the mentioned size, as for each article the talk page is mentioned too


  - sq:
    - FA 43: https://sq.wikipedia.org/wiki/Kategoria:Artikuj_t%C3%AB_p%C3%ABrkryer
    - Good 33: https://sq.wikipedia.org/wiki/Kategoria:Artikuj_t%C3%AB_mir%C3%AB
  - mzd
    - FA(285): 
"""

# (number from the featured/good articles page; not the category)

FA_NAMES = {
    "en": "Featured articles", # (6,799) https://en.wikipedia.org/wiki/Category:Wikipedia_featured_articles
    "de": "Wikipedia:Exzellent", # (2,946) https://de.wikipedia.org/wiki/Kategorie:Wikipedia:Exzellent
    "zh": "典范条目", # (1,030) https://zh.wikipedia.org/wiki/Category:%E5%85%B8%E8%8C%83%E6%9D%A1%E7%9B%AE
    "nl": "Etalage-artikelen", # (380) https://nl.wikipedia.org/wiki/Categorie:Wikipedia:Etalage-artikelen
    "no": "Utmerkede artikler", # (351) https://no.wikipedia.org/wiki/Kategori:Utmerkede_artikler
    "it": "Voci_in_vetrina_su_it.wiki", # (583) https://it.wikipedia.org/wiki/Categoria:Voci_in_vetrina_su_it.wiki
    "pt": "!Artigos destacados", # (1512) https://pt.wikipedia.org/wiki/Categoria:!Artigos_destacados
    "ro": "Articole_de_calitate", # (199) https://ro.wikipedia.org/wiki/Categorie:Articole_bune
    "ru": "Википедия:Избранные_статьи_по_алфавиту", # (2072) https://ru.wikipedia.org/wiki/%D0%9A%D0%B0%D1%82%D0%B5%D0%B3%D0%BE%D1%80%D0%B8%D1%8F:%D0%92%D0%B8%D0%BA%D0%B8%D0%BF%D0%B5%D0%B4%D0%B8%D1%8F:%D0%98%D0%B7%D0%B1%D1%80%D0%B0%D0%BD%D0%BD%D1%8B%D0%B5_%D1%81%D1%82%D0%B0%D1%82%D1%8C%D0%B8_%D0%BF%D0%BE_%D0%B0%D0%BB%D1%84%D0%B0%D0%B2%D0%B8%D1%82%D1%83
    "uk": "Вікіпедія:Вибрані_статті", # (246) https://uk.wikipedia.org/wiki/%D0%9A%D0%B0%D1%82%D0%B5%D0%B3%D0%BE%D1%80%D1%96%D1%8F:%D0%92%D1%96%D0%BA%D1%96%D0%BF%D0%B5%D0%B4%D1%96%D1%8F:%D0%92%D0%B8%D0%B1%D1%80%D0%B0%D0%BD%D1%96_%D1%81%D1%82%D0%B0%D1%82%D1%82%D1%96
    "sr": "Сјајни_чланци", # (373) https://sr.wikipedia.org/wiki/%D0%9A%D0%B0%D1%82%D0%B5%D0%B3%D0%BE%D1%80%D0%B8%D1%98%D0%B0:%D0%A1%D1%98%D0%B0%D1%98%D0%BD%D0%B8_%D1%87%D0%BB%D0%B0%D0%BD%D1%86%D0%B8
    "bg": "Избрани_статии_на_български", # (153) https://bg.wikipedia.org/wiki/%D0%9A%D0%B0%D1%82%D0%B5%D0%B3%D0%BE%D1%80%D0%B8%D1%8F:%D0%98%D0%B7%D0%B1%D1%80%D0%B0%D0%BD%D0%B8_%D1%81%D1%82%D0%B0%D1%82%D0%B8%D0%B8_%D0%BD%D0%B0_%D0%B1%D1%8A%D0%BB%D0%B3%D0%B0%D1%80%D1%81%D0%BA%D0%B8
    "th": "บทความคัดสรร", # (218) https://th.wikipedia.org/wiki/%E0%B8%AB%E0%B8%A1%E0%B8%A7%E0%B8%94%E0%B8%AB%E0%B8%A1%E0%B8%B9%E0%B9%88:%E0%B8%9A%E0%B8%97%E0%B8%84%E0%B8%A7%E0%B8%B2%E0%B8%A1%E0%B8%84%E0%B8%B1%E0%B8%94%E0%B8%AA%E0%B8%A3%E0%B8%A3 
    "id": "Artikel_pilihan", # (421) https://id.wikipedia.org/wiki/Kategori:Artikel_bagus
    "vi": "Bài_viết_chọn_lọc", # (479) https://vi.wikipedia.org/wiki/Th%E1%BB%83_lo%E1%BA%A1i:B%C3%A0i_vi%E1%BA%BFt_ch%E1%BB%8Dn_l%E1%BB%8Dc
    "tr": "Seçkin_maddeler", # (228) https://tr.wikipedia.org/wiki/Kategori:Se%C3%A7kin_maddeler
    "uz": "Tanlangan_maqolalar", # (26) https://uz.wikipedia.org/wiki/Turkum:Tanlangan_maqolalar
    "sq": "Artikuj_të_përkryer", # (43) https://sq.wikipedia.org/wiki/Kategoria:Artikuj_t%C3%AB_p%C3%ABrkryer
    "az": "Vikipediya:Əlifba_sırasına_görə_seçilmiş_məqalələr", # (318) https://az.wikipedia.org/wiki/Kateqoriya:Vikipediya:%C6%8Flifba_s%C4%B1ras%C4%B1na_g%C3%B6r%C9%99_se%C3%A7ilmi%C5%9F_m%C9%99qal%C9%99l%C9%99r
    "mk": "Избрани_статии", # (353) https://mk.wikipedia.org/wiki/%D0%9A%D0%B0%D1%82%D0%B5%D0%B3%D0%BE%D1%80%D0%B8%D1%98%D0%B0:%D0%98%D0%B7%D0%B1%D1%80%D0%B0%D0%BD%D0%B8_%D1%81%D1%82%D0%B0%D1%82%D0%B8%D0%B8
    "hy": "Վիքիպեդիա:Ընտրյալ_հոդվածներ" # (61) https://hy.wikipedia.org/wiki/%D4%BF%D5%A1%D5%BF%D5%A5%D5%A3%D5%B8%D6%80%D5%AB%D5%A1:%D5%8E%D5%AB%D6%84%D5%AB%D5%BA%D5%A5%D5%A4%D5%AB%D5%A1:%D4%B8%D5%B6%D5%BF%D6%80%D5%B5%D5%A1%D5%AC_%D5%B0%D5%B8%D5%A4%D5%BE%D5%A1%D5%AE%D5%B6%D5%A5%D6%80
}

# missing for: nl,
GOOD_NAMES = {
    "en": "Good articles", # (42,623) https://en.wikipedia.org/wiki/Category:Wikipedia_good_articles
    "zh": "優良條目", # (3,348) https://zh.wikipedia.org/wiki/Category:%E5%84%AA%E8%89%AF%E6%A2%9D%E7%9B%AE
    "no": "Anbefalte artikler", # (1051) https://no.wikipedia.org/wiki/Kategori:Anbefalte_artikler
    "it": "Voci_di_qualità_su_it.wiki", # (569) https://it.wikipedia.org/wiki/Categoria:Voci_di_qualit%C3%A0_su_it.wiki
    "pt": "!Artigos bons", # (1923) https://pt.wikipedia.org/wiki/Categoria:!Artigos_bons
    "ro": "Articole_bune", # (495) https://ro.wikipedia.org/wiki/Categorie:Articole_bune
    "ru": "Википедия:Хорошие_статьи_по_алфавиту", # (5002) https://ru.wikipedia.org/wiki/%D0%9A%D0%B0%D1%82%D0%B5%D0%B3%D0%BE%D1%80%D0%B8%D1%8F:%D0%92%D0%B8%D0%BA%D0%B8%D0%BF%D0%B5%D0%B4%D0%B8%D1%8F:%D0%A5%D0%BE%D1%80%D0%BE%D1%88%D0%B8%D0%B5_%D1%81%D1%82%D0%B0%D1%82%D1%8C%D0%B8_%D0%BF%D0%BE_%D0%B0%D0%BB%D1%84%D0%B0%D0%B2%D0%B8%D1%82%D1%83
    "uk": "Вікіпедія:Добрі_статті", # (1077) https://uk.wikipedia.org/wiki/%D0%9A%D0%B0%D1%82%D0%B5%D0%B3%D0%BE%D1%80%D1%96%D1%8F:%D0%92%D1%96%D0%BA%D1%96%D0%BF%D0%B5%D0%B4%D1%96%D1%8F:%D0%94%D0%BE%D0%B1%D1%80%D1%96_%D1%81%D1%82%D0%B0%D1%82%D1%82%D1%96
    "sr": "Добри_чланци", # (461) https://sr.wikipedia.org/wiki/%D0%9A%D0%B0%D1%82%D0%B5%D0%B3%D0%BE%D1%80%D0%B8%D1%98%D0%B0:%D0%94%D0%BE%D0%B1%D1%80%D0%B8_%D1%87%D0%BB%D0%B0%D0%BD%D1%86%D0%B8
    "bg": "Добри_статии", # (59) https://bg.wikipedia.org/wiki/%D0%9A%D0%B0%D1%82%D0%B5%D0%B3%D0%BE%D1%80%D0%B8%D1%8F:%D0%94%D0%BE%D0%B1%D1%80%D0%B8_%D1%81%D1%82%D0%B0%D1%82%D0%B8%D0%B8
    "th": "บทความคุณภาพ", # (198) https://th.wikipedia.org/wiki/%E0%B8%AB%E0%B8%A1%E0%B8%A7%E0%B8%94%E0%B8%AB%E0%B8%A1%E0%B8%B9%E0%B9%88:%E0%B8%9A%E0%B8%97%E0%B8%84%E0%B8%A7%E0%B8%B2%E0%B8%A1%E0%B8%84%E0%B8%B8%E0%B8%93%E0%B8%A0%E0%B8%B2%E0%B8%9E
    "id": "Artikel_bagus", # (431) https://id.wikipedia.org/wiki/Kategori:Artikel_bagus
    "vi": "Bài_viết_chất_lượng_tốt", # (566) https://vi.wikipedia.org/wiki/Th%E1%BB%83_lo%E1%BA%A1i:B%C3%A0i_vi%E1%BA%BFt_ch%E1%BA%A5t_l%C6%B0%E1%BB%A3ng_t%E1%BB%91t
    "tr": "Kaliteli_maddeler", # (344) https://tr.wikipedia.org/wiki/Kategori:Se%C3%A7kin_maddeler
    "sq": "Artikuj_të_mirë", # (33) https://sq.wikipedia.org/wiki/Kategoria:Artikuj_t%C3%AB_mir%C3%AB
    "az": "Vikipediya:Əlifba_sırasına_görə_yaxşı_məqalələr", # (451) https://az.wikipedia.org/wiki/Kateqoriya:Vikipediya:%C6%8Flifba_s%C4%B1ras%C4%B1na_g%C3%B6r%C9%99_yax%C5%9F%C4%B1_m%C9%99qal%C9%99l%C9%99r
    "mk": "Добри_статии", # (44) https://mk.wikipedia.org/wiki/%D0%9A%D0%B0%D1%82%D0%B5%D0%B3%D0%BE%D1%80%D0%B8%D1%98%D0%B0:%D0%94%D0%BE%D0%B1%D1%80%D0%B8_%D1%81%D1%82%D0%B0%D1%82%D0%B8%D0%B8
    "hy": "Վիքիպեդիա:Լավ_հոդվածներ", # (58) https://hy.wikipedia.org/wiki/%D4%BF%D5%A1%D5%BF%D5%A5%D5%A3%D5%B8%D6%80%D5%AB%D5%A1:%D5%8E%D5%AB%D6%84%D5%AB%D5%BA%D5%A5%D5%A4%D5%AB%D5%A1:%D4%BC%D5%A1%D5%BE_%D5%B0%D5%B8%D5%A4%D5%BE%D5%A1%D5%AE%D5%B6%D5%A5%D6%80
    "uz": "Xushsifat_maqolalar", # (41) https://uz.wikipedia.org/wiki/Turkum:Xushsifat_maqolalar
}

# only for uzbek
QUALITY_NAMES = {"uz": "Yaxshi_maqolalar" # (62) https://uz.wikipedia.org/wiki/Turkum:Yaxshi_maqolalar
        }

# Categorie:Wikipedia:Etalage-artikelen
CATEGORY_NAMES = {
    
    "en": "Category",
    "de": "Kategorie",
    "zh": "Category",
    "nl": "Categorie:Wikipedia",
    "no": "Kategori",
    "it": "Categoria",
    "pt": "Categoria",
    "ro": "Categorie",
    "ru": "Категория",
    "uk": "Категорія",
    "sr": "Категорија",
    "bg": "Категория",
    "zh": "Category",
    "th": "หมวดหมู่",
    "id": "Kategori",
    "vi": "Thể_loại",
    "tr": "Kategori",
    "sq": "Kategoria",
    "az": "Kateqoriya",
    "mk": "Категорија",
    "hy": "Կատեգորիա",
    "uz": "Turkum",
    "uz2": "Turkum"
}

# HEADER = {
#     "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
# }

HEADER = {
    "User-Agent": "Collecting FA and Good articles"
}

def query_high_quality_articles(lang: str, quality: str, QUALITY_NAMES: dict):
    
    S = requests.Session()
    URL = f"https://{lang}.wikipedia.org/w/api.php"

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
        response = S.get(url=URL, params=params, headers=HEADER)
        print(response)
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
    # "sr",  # Serbian
    # "bg",  # Bulgarian
    # "id",  # Indonesian
    # "vi",  # Vietnamese
    # "tr"  # Turkish
    # "sq",
    # "az",
    # "mk"
    # "hy",
    # "de",
    # "zh",
    # "uz",
    "th"
    ]
    for lang in languages:
        print(f"Running FA {lang} ...")

        # FA
        
        query_high_quality_articles(lang=lang, 
                                    quality='fa',
                                    QUALITY_NAMES=FA_NAMES)

        # Good articles
        if lang in GOOD_NAMES and lang not in ['en', 'ru', "zh"]:
            print(f"Running Good {lang} ...")
            query_high_quality_articles(lang=lang, 
                                    quality='good',
                                    QUALITY_NAMES=GOOD_NAMES)

        # only fpor uz
        if lang == "uz":
            query_high_quality_articles(lang=lang, 
                                        quality='quality',
                                        QUALITY_NAMES=QUALITY_NAMES
                                        )
        print("sleeping ...")
        time.sleep(10)

if __name__ == "__main__":
    main()