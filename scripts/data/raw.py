import os
import requests
import json
from tqdm import tqdm
import sys
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--languages", nargs="+", required=True)
args = parser.parse_args()

BASE_DIR = os.getenv("BASE_WCD")
INPUT_PATH = os.path.join(BASE_DIR, "data/raw/api")
OUTPUT_PATH = os.path.join(BASE_DIR, "data/raw/htmls")

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}


def get_topic(article: str, lang: str):
    inference_url = 'https://api.wikimedia.org/service/lw/inference/v1/models/outlink-topic-model:predict'
    
    data = {'page_title': article, "lang": lang}
    try:
        response = requests.post(inference_url, headers=headers, data=json.dumps(data))
    except Exception as e:
        print(f"API Error for article {article} in language {lang}: {e}")
        return None
    
    try:
        r = response.json()
        sorted_results = sorted(r['prediction']['results'], key=lambda x: x['score'], reverse=True)
        topic = sorted_results[0]['topic']
        return topic
    except Exception as e:
        print(f"Parsing Error for article {article} in language {lang}: {e}")
        return None

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

    print(f"All languages: {args.languages} ...", flush=True)
    for lang in args.languages:
        print(f"\tRunning {lang} ...", flush=True)

        INPUT_FILE = os.path.join(INPUT_PATH, f"{lang}_all.jsonl")
        OUTPUT_FILE = os.path.join(OUTPUT_PATH, f"{lang}_htmls.jsonl")
        
        
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]

        # Collect already downloaded titles
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
            for i, item in enumerate(tqdm(data)):
                if i % 500 == 0 and i !=0:
                    time.sleep(60)
                title = item['title']
                
                if title in processed_titles:
                    continue

                # get html
                raw = get_html(lang, title)
                if not raw:
                    continue
                
                # get topic
                topic = get_topic(title, lang)
                item.update({"topic": topic, "raw": raw})
                out_f.write(json.dumps(item, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()