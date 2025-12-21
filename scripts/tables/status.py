import os
from pathlib import Path
from utils import load_metrics, LANGS
from sys import meta_path
from collections import defaultdict

BASE_DIR = os.getenv("BASE_WCD")
EX1_DIR = os.path.join(BASE_DIR, "data/exp1")

def find_and_collect_all_runs():

    root = Path(EX1_DIR)
    # collect all lang dirs
    all_langs_dirs = [d for d in root.iterdir() if d.is_dir() if d.name not in ["smoke_test"]]

    models = defaultdict(dict)

    for lang_dir in all_langs_dirs:
        # check if empy dir
        if not any(lang_dir.iterdir()):
            print(f"Empty lang dir: {lang_dir}")


        # collect all runs in a lang dir if not empty
        all_lang_runs = [d for d in lang_dir.iterdir() if d.is_dir() if any(d.iterdir())]

        for lang_run_dir in all_lang_runs:
            
            all_meta_files = [f for f in lang_run_dir.iterdir() if f.is_file()]
            if len(all_meta_files) == 0:
                print(f"No meta files found in: {lang_run_dir}")
                continue

            meta_1 = load_metrics(all_meta_files[0])

            if meta_1['model_type'] not in ['slm', 'clf']:
                continue
            
            # HP RUN
            if (meta_1["experiment"] == "binary" and 
                meta_1['seed'] == 42 and
                len(all_meta_files) == 6):

                lang = meta_1["lang"]
                model_type = meta_1["model_type"]
                model_name = meta_1["model_name"]
                atl = meta_1["atl"]

                models[(lang, model_type, model_name, atl)]['hp'] = lang_run_dir

            if (meta_1["experiment"] == "seed" and 
                meta_1['seed'] in [2025,2026] and
                len(all_meta_files) == 1):

                lang = meta_1["lang"]
                model_type = meta_1["model_type"]
                model_name = meta_1["model_name"]
                atl = meta_1["atl"]

                models[(lang, model_type, model_name, atl)][f'seed_{meta_1["seed"]}'] = lang_run_dir


    return models

def main():
    all_models = find_and_collect_all_runs()
    for resource, langs in LANGS.items():
        print("\n\n" + "="*40)
        print("="*40)
        print(f"Resource: {resource}")
        print("="*40)
        print("="*40)

        for lang in langs:
            print("="*20)
            print(f"Lang: {lang}")
            print("="*20)

            for model_type in ['slm', 'clf']:
                print("-"*10)
                print(f"Model type: {model_type}")
                print("-"*10)

                all_models_lang = {k:v for k,v in all_models.items() if k[0] == lang and k[1] == model_type}
                for (lang, model_type, model_name, atl), runs in all_models_lang.items():
                    print(f"\nModel: {model_name} | Type: {model_type} | Lang: {lang} | ATL: {atl}")
                    hp_run = runs.get('hp', None)
                    seed_runs = {k:v for k,v in runs.items() if k.startswith('seed_')}
                    print(f"  HP Run: {hp_run}")
                    for seed, run_dir in seed_runs.items():
                        print(f"  Seed Run ({seed}): {run_dir}")
                                
if __name__ == "__main__":
    main()
