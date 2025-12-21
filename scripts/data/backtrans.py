import os 
import json
import random
import deepl
import time
from openai import OpenAI


# ----------------------------------------------------------------
# configs
# ----------------------------------------------------------------

BASE_DIR = os.getenv("BASE_WCD", ".")
IN_DIR = os.path.join(BASE_DIR, "data/sets/translated")
OUT_DIR = os.path.join(BASE_DIR, "data/sets/backtranslated")
os.makedirs(OUT_DIR, exist_ok=True)

# TARGET_LANGUAGES = ["uk", "ro", "id", "bg", "uz"] + ["no", "az", "mk", "hy", "sq"]
# TARGET_LANGUAGES = ["uz", "az", "hy", "sq"]
TARGET_LANGUAGES = ["uz"]
SMOKE_TEST = False

DEEPL_AUTH_KEY = os.getenv("DEEPL_AUTH_KEY")

random.seed(42)

PROMPT = "Translate into English. Do not return any explanation or other text. Only the English Translation: {translation}."

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

def query(item: dict, 
          openai_client, 
          model: str = "openai/gpt-4o"
          ) -> int:
    """query openrouter for detecting translation issues"""
    
    translated_text = item["mt"]
    prompt = PROMPT.format(translation=translated_text)
    messages = [
        {"role": "system", "content": "You are an expert translator."},
        {"role": "user", "content": prompt},
    ]
    
    completion = openai_client.chat.completions.create(
        model=model,
        temperature=0.1,
        messages=messages,
    )
    content = completion.choices[0].message.content
    item['backtranslation'] = content
    return item

def single_backtranslation(item: dict, deepl_client, target_language: str = "EN-US") -> dict:
    """translate text using deepl"""
    translated_text = item["mt"]
    result = deepl_client.translate_text(text=translated_text, target_lang=target_language)

    item['backtranslation'] = result.text
    return item

def backtranslation(items: list[dict], original_language: str, client=None) -> list[dict]:
    """backtranslate a list of items"""
    translated_items = []
    for i, item in enumerate(items, start=1):
        
        # need to use openai for some low-resource languages; deepl does not support those below
        if original_language in ['hy', "az", "sq", "uz"]:
            translated_item = query(item=item, openai_client=client)
        else:
            translated_item = single_backtranslation(item=item, deepl_client=client)
            if i%20 == 0:
                time.sleep(60)
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
    openai_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"]
            )
    
    for target_language in TARGET_LANGUAGES:
        print("="*20)
        print(f"Backtranslating {target_language} to EN")
        print("="*20)

        # load data
        items = load_translated_data(target_language=target_language)

        # prepare items
        items = prepare_items(items)
        
        if target_language in ['hy', "az", "sq", "uz"]:
            backtranslation(items=items, 
                            client=openai_client, 
                            original_language=target_language)
        else:
            backtranslation(items=items, 
                        client=deepl_client, 
                        original_language=target_language)

            print("Sleeping to respect usage quotas")
            time.sleep(120)
        

        
if __name__ == "__main__":
    main()