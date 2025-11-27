import os
import re
import json
import glob
from collections import defaultdict
import sys
from pathlib import Path

print("new")

BASE_DIR = os.getenv("BASE_WCD")
SLM_DIR = os.path.join(BASE_DIR, "data/exp1")
PLM_DIR = os.path.join(BASE_DIR, "data/models/plm")

MODEL_MAPPING =  {
    "mBert": "google-bert/bert-base-multilingual-uncased",
    "xlm-r-b": "FacebookAI/xlm-roberta-base",
    "xlm-r-l": "FacebookAI/xlm-roberta-large",
    "mDeberta-b": "microsoft/mdeberta-v3-base",
    "mDeberta-l": "microsoft/deberta-v3-large",
    "llama3_1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama3_3b": "meta-llama/Llama-3.2-3B-Instruct",
    "llama3_8b": "meta-llama/Llama-3.1-8B-Instruct",
    "llama3_70b": "meta-llama/Llama-3.3-70B-Instruct",
    "qwen3_06b": "Qwen/Qwen3-0.6B",
    "qwen3_4b": "Qwen/Qwen3-4B-Instruct-2507",
    "qwen3_8b": "Qwen/Qwen3-8B",
    "qwen3_30b": "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "qwen3_32b": "Qwen/Qwen3-32B",
    "gemma3_12b": "google/gemma-3-12b-it",
    "aya": "CohereLabs/aya-101",
    "gpt-4o-mini": "openai/gpt-4o-mini",
    "gemini-2.5-flash-lite": "google/gemini-2.5-flash-lite"
    }

MODEL_MAPPING_REVERSE = {v: k for k, v in MODEL_MAPPING.items()}

run_re = re.compile(r"run_\d+")
meta_re = re.compile(r"meta_\d+")


LANGS = ["en","nl","no","it","pt","ro","ru","uk","bg","id", "vi", "tr"]

def load_metrics(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_slm_models(path: str, is_context: bool):
    root = Path(path)
    rows = defaultdict(dict)

    for lang_dir in root.iterdir():
        if not lang_dir.is_dir():
            continue
        lang = lang_dir.name
        if lang not in LANGS:
            continue

        for run_dir in lang_dir.iterdir():
            if not (run_dir.is_dir() and run_re.fullmatch(run_dir.name)):
                continue

            best_for_run = None  # will store dict with dev_acc + test_acc etc.

            for meta_path in run_dir.iterdir():
                if not (meta_path.is_file() and meta_re.match(meta_path.stem)):
                    continue

                with meta_path.open() as f:
                    meta = json.load(f)

                # filter by context
                if meta["context"] != is_context:
                    continue

                pe = meta.get("prompt_extension", "")

                if pe:
                    continue
                    
                # map HF model name -> short name
                try:
                    short_name = MODEL_MAPPING_REVERSE[meta["model_name"]]
                except KeyError:
                    # skip unknown models
                    continue

                atl_flag = "atl" if meta["atl"] else "van"
                model_name = f"{short_name} ({atl_flag})"

                dev_metrics = meta.get("dev_metrics", [])
                test_metrics = meta.get("test_metrics", [])

                # go over epochs in *this* meta file
                for dev_entry in dev_metrics:
                    epoch = dev_entry["epoch"]
                    dev_acc = dev_entry["metrics"]["accuracy"]

                    if (
                        best_for_run is None
                        or dev_acc > best_for_run["dev_acc"]
                    ):
                        test_entry = next(
                            (t for t in test_metrics if t["epoch"] == epoch),
                            None
                        )
                        if test_entry is None:
                            continue

                        best_for_run = {
                            "model_name": model_name,
                            "lang": lang,
                            "dev_acc": dev_acc,
                            "dev_f1": dev_entry["metrics"]["f1"],
                            "test_acc": test_entry["metrics"]["accuracy"],
                        }

            # after scanning all meta_* in this run
            if best_for_run:
                m = best_for_run["model_name"]
                l = best_for_run["lang"]
                rows[m][l] = best_for_run["test_acc"]

    return rows

def load_plm_models(path: str,
                model_type: str,
                context: bool,
                display_name: str,
                training_size: int):
    
    rows = defaultdict(dict)
    count = defaultdict(int)

    for lang in LANGS:
        paths = glob.glob(os.path.join(path, lang, "meta_*.json"))
        for p in paths:
            try:
                meta = load_metrics(p)
                lang = meta['lang']
                model_name = meta['model_name'].replace("_", "-")
            except:
                print(f"Error when loading meta for path: {p}")
                continue

            if not (meta['training_size'] == training_size and 
                    meta['model_type'] == model_type and 
                    meta['context'] == context and
                    meta['smoke_test'] == False):
                continue
            
            count[(model_type, model_name, meta['lang'], context, training_size)] += 1
            
            model_number = meta['model_number']
            model_name = model_name + f" ({display_name})"
            try:
                dev_accuracy = meta['dev_metrics']['accuracy']
                test_accuracy = meta['test_metrics']['accuracy']
            except:
                dev_accuracy = meta['dev_metrics']['eval_accuracy']
                test_accuracy = meta['test_metrics']['eval_accuracy']

            score = (dev_accuracy, test_accuracy)

            if model_name not in rows:
                accs = {l: None for l in LANGS}
                rows[model_name] = accs

            if not rows[model_name][lang]:
                rows[model_name][lang] = score
            if rows[model_name][lang] and score[0] > rows[model_name][lang][0]:
                rows[model_name][lang] = score
    
    # keep test scores
    for m, v in rows.items():
        for l, s in v.items():
            if s:
                rows[m][l] = s[1]
    print("\n", f"Collection counts")
    print(count)                
    return rows

def latex_table(rows, context):
    # print LaTeX table
    table = "\n\n"
    colspec = "l" + "c" * len(LANGS)
    header = "Model & " + " & ".join(LANGS) + " \\\\"

    table += "\\begin{tabular}{" + colspec + "}\n"
    table += "\\hline\n"
    table += header + "\n"
    table += "\\hline\n"

    lang_max = {l: float("-inf") for l in LANGS}
    lang_second_max = {l: float("-inf") for l in LANGS}

    for model, accs in rows.items():
        for lang, v in accs.items():
            if v is None:
                continue
            # update max and second max in one go
            if v > lang_max[lang]:
                lang_second_max[lang] = lang_max[lang]
                lang_max[lang] = v
            elif v > lang_second_max[lang]:
                lang_second_max[lang] = v
    
    prev=None
    for name, accs in rows.items():
        cells = []
        regime = re.search(r"\((.*?)\)", name).group(1)
        for l in LANGS:
            if l not in accs.keys():
                v = None
            else:
                v = accs[l]
            if isinstance(v, (int, float)) and v == lang_max[l]:
                cells.append(f"\\textbf{{{v:.3f}}}")
            elif isinstance(v, (int, float)) and v == lang_second_max[l]:
                cells.append(f"\\underline{{{v:.3f}}}")
            else:
                cells.append(f"{v:.3f}" if isinstance(v, (int, float)) else "--")

        if prev and prev != regime:
            table += f"\\hline \n"    
        table += f"{name} & " + " & ".join(cells) + " \\\\\n"
        
        prev = regime

    table += "\\hline\n"
    table += "\\end{tabular}\n"

    # Save to file
    with open(f"table1_c{1 if context else 0}.tex", "w", encoding="utf-8") as f:
        f.write(table)

    print(table)

def merge_defaultdicts(d,d1):
    for k,v in d1.items():
        if (k in d):
            d[k].update(d1[k])
        else:
            d[k] = d1[k]
    return d

def main():

    context = int(sys.argv[1])
    assert context in [0,1], f"Context is not binary {context}"
    context = bool(context)

    rows_plm = load_plm_models(path=PLM_DIR, 
                           model_type="vanilla", 
                           display_name='hp',
                           context=context,
                           training_size=-1 
                           )

    rows_slm = load_slm_models(path=SLM_DIR,
                               is_context=context,)
    
    
    r = merge_defaultdicts(rows_plm, rows_slm)
    print(r)
    latex_table(r, context)

if __name__ == "__main__":
    main()
