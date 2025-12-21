import os
import json
import glob
from itertools import product

BASE_DIR = os.getenv("BASE_WCD")
MODEL_DIR = os.path.join(BASE_DIR, "data/models")

models = ["atl", "vanilla", "classifier"]
languages = ["en", "nl", "no", "it", "pt", "ro", "ru", "uk", "bg", "id", "vi", "tr"]

EPOCHS_LIST = [1, 2, 3]
LR_LIST = [5e-4, 2e-4, 5e-5]
BATCH_LIST = [24]
GRAD_NORM_LIST = [0.4, 0.6, 0.8]

ALL_COMBOS = {
    (epochs, lr, batch_size, grad_norm)
    for epochs, lr, batch_size, grad_norm in product(
        EPOCHS_LIST, LR_LIST, BATCH_LIST, GRAD_NORM_LIST
    )
}

def extract_hparams(meta):
    """
    Get hps from meta
    """
    src = meta
    if "args" in meta and isinstance(meta["args"], dict):
        src = meta["args"]

    try:
        epochs = int(src["epochs"])
        lr = float(src["learning_rate"])
        batch_size = int(src["batch_size"])
        max_grad_norm = float(src["max_grad_norm"])
    except KeyError as e:
        ValueError("Meta has not the key you expect.")
    
    return (epochs, lr, batch_size, max_grad_norm)


def get_existing_combos_for_model_lang(model_name, lang):

    lang_dir = os.path.join(MODEL_DIR, model_name, lang)
    if not os.path.isdir(lang_dir):
        ValueError("Dir does not exist {lang_dir}")

    existing = set()

    # model_* dirs
    paths = glob.glob(os.path.join(lang_dir, "model_*"))
    for p in paths:
        meta_path = os.path.join(p, "meta.json")
        if not os.path.isfile(meta_path):
            continue

        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception as e:
            print(f"Could not read {meta_path}: {e}")
            continue

        hp = extract_hparams(meta)
        if hp is None:
            continue

        existing.add(hp)

    return existing


def main():
    for model_name in models:
        for lang in languages:
            existing = get_existing_combos_for_model_lang(model_name, lang)
            if not existing:
                print(f"\n[{model_name} | {lang}] No runs found.")
                missing = ALL_COMBOS
            else:
                missing = ALL_COMBOS - existing

            print(f"\n[{model_name} | {lang}]")
            print(f"  Existing HP runs: {len(existing)} / {len(ALL_COMBOS)}")
            print(f"  Missing HP runs:  {len(missing)}")

            for (epochs, lr, batch_size, grad_norm) in sorted(missing):
                print(
                    f"    epochs={epochs}, lr={lr:g}, "
                    f"batch_size={batch_size}, max_grad_norm={grad_norm}"
                )


if __name__ == "__main__":
    main()