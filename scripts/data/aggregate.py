import json
import os
import glob

BASE_DIR = os.getenv("BASE_WCD")
INPUT_PATH = os.path.join(BASE_DIR, "data/raw/api")
OUT_PATH = os.path.join(BASE_DIR, "data/raw")

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
        "zh",  # Chinese
        "ar",  # Arabic
        "id"   # Indonesian
    ]

    for lang in languages:
        all_files = glob.glob(os.path.join(INPUT_PATH, f"{lang}*"))
        print(len(all_files))
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
                if title.startswith("Talk:"): # zh cases
                    title = title.replace("Talk:", "")
                if item['title'] not in unique_titles:
                    all_titles.append({"title": item["title"], "source": source})
                    unique_titles.add(item['title'])
            
        OUTPUT_FILE = os.path.join(OUT_PATH, f"{lang}_all.jsonl")
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            for entry in all_titles:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()