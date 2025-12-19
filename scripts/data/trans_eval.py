import os 
import json
import random
import deepl
from comet import download_model, load_from_checkpoint
from rouge import Rouge

from bert_score import score
import sacrebleu
import torch 

print("="*15)
print(torch.cuda.get_device_name(0))
print("="*15)

rouge = Rouge()

# ----------------------------------------------------------------
# configs
# ----------------------------------------------------------------

BASE_DIR = os.getenv("BASE_WCD", ".")
IN_DIR = os.path.join(BASE_DIR, "data/sets/backtranslated")
OUT_DIR = os.path.join(BASE_DIR, "data/exp2/trans_eval")
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_LANGUAGES = ["uk", "ro", "id", "bg", "uz"] + ["no", "az", "mk", "hy", "sq"]

SMOKE_TEST = True

DEEPL_AUTH_KEY = os.getenv("DEEPL_AUTH_KEY")

random.seed(42)

# ----------------------------------------------------------------
# fucntions
# ----------------------------------------------------------------

def load_data(target_language: str):
    """load the backtranslated data from jsonl"""
    in_path = os.path.join(IN_DIR, f"{target_language}.jsonl")
    data = []
    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            data.append(item)
    return data


def comet_evaluation(items: list[dict], 
                     model_name: str = "Unbabel/wmt23-cometkiwi-da-xl"
                     ) -> dict:
    
    # rm the backtranslation key
    items_comet = items.copy()
    for item in items_comet:
        del item["backtranslation"]

    model_path = download_model(model_name)
    model = load_from_checkpoint(model_path)

    # Call predict method
    model_output = model.predict(items_comet, batch_size=8, gpus=1)
    score = {"comet_score": model_output['system_score']}
    return score


def backtranslation_evaluation(items: list[dict]):

    # preapre data
    original = [x['src'] for x in items]
    translations = [x['backtranslation'] for x in items]

    lang = "en" # we run bertscore on the en translations

    bleu = sacrebleu.corpus_bleu(translations, [original])
    rouge_scores = rouge.get_scores(translations, original)
    P, R, F1 = score(translations, original, lang=lang, verbose=True)

    out = {"bleu": bleu.score,
           "rouge1": rouge_scores[0]['rouge-1']['f'],
           "rouge2": rouge_scores[0]['rouge-2']['f'],
           "bertscore": float(F1.mean())}
    return out



def main():
    for target_language in TARGET_LANGUAGES:
        print("="*20)
        print(f"Evaluating {target_language}")
        print("="*20)

        items = load_data(target_language=target_language)

        # comet eval
        comet_metrics = comet_evaluation(items)

        # backtranslation eval
        bt_metrics = backtranslation_evaluation(items)
        all_metrics = {**comet_metrics, **bt_metrics}

        # save results
        out_path = os.path.join(OUT_DIR, f"{target_language}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_metrics, f, indent=4, ensure_ascii=False)