import os
import re
import json
import glob
from collections import defaultdict

BASE_DIR = os.getenv("BASE_WCD")
PLM_DIR = os.path.join(BASE_DIR, "data/models/plm")
SLM_SFT_DIR = os.path.join(BASE_DIR, "data/models/vanilla")
SLM_PWL_DIR = os.path.join(BASE_DIR, "data/models/atl")
SLM_C_DIR = os.path.join(BASE_DIR, "data/models/classifier")

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


LANGS = ["en","nl","no","it","pt","ro","ru","uk","bg","id", "vi", "tr"]

def load_metrics(path):
    with open(os.path.join(path, "meta.json"), "r", encoding="utf-8") as f:
        return json.load(f)

def load_models(path: str, variant: str):
    rows = defaultdict(dict)
    for lang in LANGS:
        paths = glob.glob(os.path.join(path, lang, "model_*"))
        dirs = [p for p in paths if re.search(r"model_\d+$", os.path.basename(p))]
        for dir_ in dirs:
            try:
                meta = load_metrics(dir_)
                lang = meta['lang']
            except:
                print(f"No meta for path: {dir_}")
                continue
            model_number = meta['model_number']
            model_name = meta['model_name'].replace("_", "-") + f" ({variant})"
            
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
    return rows

def latex_table(rows):
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
    with open("table1.tex", "w", encoding="utf-8") as f:
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

    rows_plm = load_models(PLM_DIR, "HP")
    rows_sft = load_models(SLM_SFT_DIR, "SFT")
    rows_pwl = load_models(SLM_PWL_DIR, "ATL")
    rows_c = load_models(SLM_C_DIR, "CH")

    print(rows_sft)
    
    r = merge_defaultdicts(rows_plm, rows_sft)
    r = merge_defaultdicts(r, rows_pwl)
    r = merge_defaultdicts(r, rows_c)

    latex_table(r)

if __name__ == "__main__":
    main()
