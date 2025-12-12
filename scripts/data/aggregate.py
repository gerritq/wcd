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
    "de": "Wikipedia:Hauptseite",
    "zh": "首页",
    "nl": "Hoofdpagina",                # Dutch
    "no": "Forside",                    # Norwegian (Bokmål default)
    "it": "Pagina_principale",          # Italian
    "pt": "Página_principal",           # Portuguese
    "ro": "Pagina_principală",          # Rsomanian
    "ru": "Заглавная_страница",         # Russian
    "uk": "Головна_сторінка",           # Ukrainian
    "sr": "Главна_страна",
    "bg": "Начална_страница",           # Bulgarian
    "vi": "Trang_Chính",           # Vietnamese
    "id": "Halaman_Utama",           # Indonesian
    "tr": "Anasayfa",           # Turkish
    "uz": "Bosh_Sahifa",
    "sq": "Faqja_kryesore",
    "az": "Ana_səhifə",
    "mk": "Главна_страница",
    "hy": "Գլխավոր_էջ",
    "th": "หน้าหลัก"
    }

# For ID, lots of FA/good article pages are discussion (Pembicaraan) pages
REMOVE_PREFIX = ["Pembicaraan:"]

# bg: Потребител = user
# bg: Категория = Category
# uk: Категорія = Category
# bg: Уикипедия = Wikipedia
# ru,bg,uk: Шаблон = template

PREFIX = ["Pembicaraan:"]
DROP_PAGES = ["Wikipedia", "Templat", "Template", "Kategori", "Category", "Categorie:Wikipedia", 
              "Categoria", "Categorie", "Категория", "Категорія", "Уикипедия", "Шаблон", 
              "Потребител", "Thể_loại", "Категорија", "Kateqoriya", "Kategoria"]


def main():
    
    languages  = [
        "en",  # English
        "de",
        "zh",
        "nl",  # Dutch
        "no",  # Norwegian (Bokmål is 'nb', Nynorsk is 'nn', 'no' redirects to Bokmål)
        "it",  # Italian
        "pt",  # Portuguese
        "ro",  # Romanian
        "ru",  # Russian
        "uk",  # Ukrainian
        "sr",  # Bulgarian
        "bg",  # Bulgarian
        "vi",
        "id",
        "tr",
        "uz",
        "sq",
        "az",
        "mk",
        "hy",
        "th"
    ]

    skip_titles = defaultdict(list)

    for lang in languages:
        # these langs have enough fa articles and dutch has no good articles
        if lang in ['en', 'ru', "nl", "de", "zh"]:
            all_files = glob.glob(os.path.join(INPUT_PATH, f"{lang}_fa*"))
        else:
            all_files = glob.glob(os.path.join(INPUT_PATH, f"{lang}_fa*")) \
                   + glob.glob(os.path.join(INPUT_PATH, f"{lang}_good*")) \
                    + glob.glob(os.path.join(INPUT_PATH, f"{lang}_quality*")) # for uzb only
            
        
        print("")
        print("="*20)
        print("Language:", lang)
        print("="*20)
        # print(all_files)

        all_titles = []
        unique_titles = set()
        for file in all_files:
            with open(file, "r", encoding="utf-8") as f:
                data = [json.loads(line) for line in f]

            filename = os.path.splitext(os.path.basename(file))[0]
            # print(filename)

            if filename.endswith("views"):
                source = "views"
            elif filename.endswith("fa"):
                source = "fa"
            elif filename.endswith("good"): # for uz
                source = "good"
            elif filename.endswith("quality"): # for uz
                source = "quality"
            elif filename.endswith("random"):
                source = "random"
            else:
                raise ValueError(f"Unknown source type in filename: {filename}")

            for i, item in enumerate(data):
                title = item['title']

                # drop if main page
                main_page = [MAIN_PAGES[lang].lower(), MAIN_PAGES[lang].replace("_", " ").lower()] # drop if main page
                if title.lower() in main_page:
                    skip_titles[lang].append(title)
                    continue

                # chinese talk page case
                # if lang == "zh":
                #     if title.startswith("Talk:"): # zh cases
                #         title = title.replace("Talk:", "")
                
                # drop if in predefined list
                if any(title.lower().startswith(p.lower()) for p in DROP_PAGES):
                    skip_titles[lang].append(title)
                    continue

                # for id
                for p in REMOVE_PREFIX:
                    if title.lower().startswith(p.lower()):
                        title = title[len(p):].strip()
                    
                if title not in unique_titles:
                    all_titles.append({"title": title, "source": source})
                    unique_titles.add(title)
            
        OUTPUT_FILE = os.path.join(OUT_PATH, f"{lang}_all.jsonl")
        print("Len of all titles:", len(all_titles))
        print("N fa:", len([x for x in all_titles if x['source'] == 'fa']))
        print("N good:", len([x for x in all_titles if x['source'] == 'good']))
        # This is uz only
        print("N quality:", len([x for x in all_titles if x['source'] == 'quality']))
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            for entry in all_titles:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    with open(INFO_FILE, "w", encoding="utf-8") as f:
        for lang, titles in skip_titles.items():
            for t in titles:
                f.write(json.dumps({"lang": lang, "title": t}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()