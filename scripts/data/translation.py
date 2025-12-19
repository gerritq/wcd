
import os
import json
import time
from tqdm import tqdm
from datasets import load_from_disk
from google.cloud import translate

# ----------------------------------------------------------------
# configs
# ----------------------------------------------------------------

BASE_DIR = os.getenv("BASE_WCD", ".")
IN_DIR = os.path.join(BASE_DIR, "data/sets/main")
OUT_DIR = os.path.join(BASE_DIR, "data/sets/translated")

MODEL_ID = "general/translation-llm"
PROJECT_ID = os.getenv("GOOGLE_CLOUD_TRANSLATION_PROJECT")

TARGET_LANGUAGES = ["uk", "ro", "id", "bg", "uz"] + ["no", "az", "mk", "hy", "sq"]

SMOKE_TEST = False

# ----------------------------------------------------------------
# func
# ----------------------------------------------------------------

def load_test_data_set(lang: str) -> list[dict]:
    """load test source data"""
    data_dir = os.path.join(IN_DIR, lang)
    ds = load_from_disk(data_dir)["test"]

    if SMOKE_TEST:
        ds = ds.select(range(10))

    return list(ds)

def translate_instance(item: dict, 
                       target_language: str,
                       source_language: str = 'en'
                       ) -> dict:
    

    target_language = 'nb' if target_language == 'no' else target_language
    
    translated_item = item.copy()
    keys = ['section', 'previous_sentence', 'claim', 'subsequent_sentence']
    for key in keys:
        new_key_name=f"{key}_translated"
        tmp_text = item[key]

        if tmp_text == "" or tmp_text is None: # cases where there is no context
            translated_item[new_key_name] = None
            continue

        translated_text = translate_text_with_model(
            text=tmp_text,
            source_language=source_language,
            target_language=target_language,
            project_id=PROJECT_ID,
        )
        translated_item[new_key_name] = translated_text
    
    return translated_item


def translate_test_set(source_test_set: list[dict],
                        source_language: str,
                       target_language: str,
                        ) -> list[dict]:

    """Translate test set"""
    translated_test_set = []

    for item in tqdm(source_test_set):
        translated_item = translate_instance(item=item, 
                                             source_language=source_language,
                                             target_language=target_language
                                             )
        translated_test_set.append(translated_item)

    # save translated test set
    out_dir = os.path.join(OUT_DIR, f"{target_language}.jsonl")
    with open(out_dir, "w", encoding="utf-8") as f_out:
        for item in translated_test_set:
            f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"Translated test set saved to {out_dir}")

def translate_text_with_model(
                                text: str,
                                target_language: str,
                                project_id: str,
                                source_language: str = "en",
                                model_id: str = MODEL_ID,
    ) -> translate.TranslationServiceClient:
    """Translates a given text using Translation custom model."""

    client = translate.TranslationServiceClient()

    location = "us-central1"
    parent = f"projects/{project_id}/locations/{location}"
    model_path = f"{parent}/models/{model_id}"

    # Supported language codes: https://cloud.google.com/translate/docs/languages
    response = client.translate_text(
        request={
            "contents": [text],
            "target_language_code": target_language,
            "model": model_path,
            "source_language_code": source_language,
            "parent": parent,
            "mime_type": "text/plain",
        }
    )
    # Display the translation for each input text provided
    
    try:
        translated_text = response.translations[0].translated_text
    except Exception as e:
        print(f"Translation error: {e}")
        raise e

    return translated_text

def example_usage():
    example_text = "Hello, how are you?"
    target_language = "it"
    source_language = "en"
    project_id = PROJECT_ID

    translate_text_with_model(
        text=example_text,
        source_language=source_language,
        target_language=target_language,
        project_id=project_id,
    )


def main():
    source_language = 'en'
    test_set = load_test_data_set(source_language)

    for target_language in TARGET_LANGUAGES[:1]:
        start = time.time()
        print("\n")
        print("="*20)
        print(f"Translating from {source_language} to {target_language}...")
        print("="*20)
        translate_test_set(source_test_set=test_set, target_language=target_language, source_language=source_language)
        
        end = time.time()
        print(f"Time taken: {(end - start) / 60} minutes.")

if __name__ == "__main__":
    main()

