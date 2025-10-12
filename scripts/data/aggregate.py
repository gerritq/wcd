import json
import os
import glob
from collections import defaultdict

BASE_DIR = os.getenv("BASE_WCD")
INPUT_PATH = os.path.join(BASE_DIR, "data/raw/api")
OUT_PATH = os.path.join(BASE_DIR, "data/raw/api")
INFO_FILE = os.path.join(BASE_DIR, "data/info/aggregate_drop.jsonl")

MAIN_PAGES = {
    "en": "Main_Page",                  # English
    "nl": "Hoofdpagina",                # Dutch
    "no": "Forside",                    # Norwegian (Bokmål default)
    "it": "Pagina_principale",          # Italian
    "pt": "Página_principal",           # Portuguese
    "ro": "Pagina_principală",          # Romanian
    "ru": "Заглавная_страница",         # Russian
    "uk": "Головна_сторінка",           # Ukrainian
    "bg": "Начална_страница",           # Bulgarian
    "zh": "Wikipedia:首页",              # Chinese
    "ar": "الصفحة_الرئيسية",            # Arabic
    "id": "Halaman_Utama"               # Indonesian
    }

# For ID, lots of FA/good article pages are discussion (Pembicaraan) pages
REMOVE_PREFIX = ["Pembicaraan:"]

# bg: Потребител = user
# bg: Категория = Category
# uk: Категорія = Category
# bg: Уикипедия = Wikipedia
# ru,bg,uk: Шаблон = template

DROP_PAGES = ["Wikipedia", "Templat", "Template", "Kategori", "Category", "Categorie:Wikipedia", 
              "Categoria", "Categorie", "Категория", "Категорія", "Уикипедия", "Шаблон", 
              "Потребител"]
def main():
    
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

    skip_titles = defaultdict(list)

    for lang in languages:
        all_files = glob.glob(os.path.join(INPUT_PATH, f"{lang}_views*")) \
          + glob.glob(os.path.join(INPUT_PATH, f"{lang}_fa*")) \
            + glob.glob(os.path.join(INPUT_PATH, f"{lang}_good*"))
        print(all_files)

        all_titles = []
        unique_titles = set()
        for file in all_files:
            with open(file, "r", encoding="utf-8") as f:
                data = [json.loads(line) for line in f]

            filename = os.path.basename(file)
            print(filename)

            if "_views" in filename:
                source = "views"
            elif "_fa" in filename:
                source = "fa"
            elif "_good" in filename:
                source = "good"
            else:
                raise ValueError(f"Unknown source type in filename: {filename}")

            for i, item in enumerate(data):
                title = item['title']

                # drop if main page
                main_page = [MAIN_PAGES[lang].lower(), MAIN_PAGES[lang].replace("_", " ").lower()] # drop if main page
                if title in main_page:
                    skip_titles[lang].append(title)
                    continue

                # chinese talk page case
                if lang == "zh":
                    if title.startswith("Talk:"): # zh cases
                        title = title.replace("Talk:", "")
                
                # drop if in predefined list
                if any(title.lower().startswith(p.lower()) for p in DROP_PAGES):
                    skip_titles[lang].append(title)
                    continue

                # for id, rm prefi
                for p in REMOVE_PREFIX:
                    if title.lower().startswith(p.lower()):
                        title = title[len(p):].strip()
                    
                if title not in unique_titles:
                    all_titles.append({"title": title, "source": source})
                    unique_titles.add(title)
            
        OUTPUT_FILE = os.path.join(OUT_PATH, f"{lang}_all.jsonl")
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            for entry in all_titles:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    with open(INFO_FILE, "w", encoding="utf-8") as f:
        for lang, titles in skip_titles.items():
            for t in titles:
                f.write(json.dumps({"lang": lang, "title": t}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()