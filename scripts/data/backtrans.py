import os 
import json
import random
import deepl

# ----------------------------------------------------------------
# configs
# ----------------------------------------------------------------

BASE_DIR = os.getenv("BASE_WCD", ".")
IN_DIR = os.path.join(BASE_DIR, "data/sets/translated")
OUT_DIR = os.path.join(BASE_DIR, "data/sets/backtranslated")
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_LANGUAGES = ["uk", "ro", "id", "bg", "uz"] + ["no", "az", "mk", "hy", "sq"]

SMOKE_TEST = True

DEEPL_AUTH_KEY = os.getenv("DEEPL_AUTH_KEY")

random.seed(42)

# ----------------------------------------------------------------
# functions
# ----------------------------------------------------------------

def load_translated_data(target_language: str):
    """load the target data from jsonl"""
    in_path = os.path.join(IN_DIR, f"{target_language}.jsonl")
    data = []
    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            data.append(item)
    return data

def single_backtranslation(item: dict, deepl_client, target_language: str = "EN-US") -> dict:
    """translate text using deepl"""
    translated_text = item["mt"]
    result = deepl_client.translate_text(text=translated_text, target_lang=target_language)

    item['backtranslation'] = result.text
    return item

def backtranslation(items: list[dict], original_language: str, deepl_client=None) -> list[dict]:
    """backtranslate a list of items"""
    translated_items = []
    for item in items:
        translated_item = single_backtranslation(item=item, deepl_client=deepl_client)
        translated_items.append(translated_item)
    
    with open(os.path.join(OUT_DIR, f"{original_language}.jsonl"), "w", encoding="utf-8") as f:
        for item in translated_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def prepare_items(data: list[dict]):
    """prepare items for comet evaluation"""
    random.seed(42)
    data_sample = random.sample(data, 2) if SMOKE_TEST else random.sample(data, 30)
    keys = ['previous_sentence', 'claim', 'subsequent_sentence']
    items = []
    for item in data_sample:
        for key in keys:
            tmp_original = item[key]
            tmp_translated = item[key+"_translated"]

            if tmp_original and tmp_translated:
                items.append({
                    "src": tmp_original,
                    "mt": tmp_translated,
                })
    return items

def main():
    deepl_client = deepl.DeepLClient(DEEPL_AUTH_KEY)

    for target_language in TARGET_LANGUAGES:
        print("="*20)
        print(f"Backtranslating {target_language} to EN")
        print("="*20)

        # load data
        items = load_translated_data(target_language=target_language)

        # prepare items
        items = prepare_items(items)
        
        backtranslation(items=items, deepl_client=deepl_client, original_language=target_language)
        

        
if __name__ == "__main__":
    main()