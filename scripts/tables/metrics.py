import os
import re
import json
import glob
import sys
from collections import defaultdict

models = sys.argv[1]

BASE_DIR = os.getenv("BASE_WCD")
METRICS_DIR = os.path.join(BASE_DIR, f"data/metrics/{models}")

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
    "qwen_06b": "Qwen/Qwen3-0.6B",
    "qwen3_4b": "Qwen/Qwen3-4B-Instruct-2507",
    "qwen3_8b": "Qwen/Qwen3-8B",
    "qwen3_30b": "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "qwen3_32b": "Qwen/Qwen3-32B",
    "aya": "CohereLabs/aya-101",
    "gpt-4o-mini": "openai/gpt-4o-mini",
    "openai/gpt-4o-mini": "gpt-4o-mini",
    "gemini-2.5-flash-lite": "google/gemini-2.5-flash-lite"
    }

LANGS = ["en","nl","no","it","pt","ro","ru","uk","bg","id"]

MODEL_MAPPING_REVERSE = {v: k for k, v in MODEL_MAPPING.items()}

def load_metrics(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_accuracy_by_lang(rec):
    out = {l: None for l in LANGS}
    data_lang = rec.get("data")
    # own language accuracy (from test_eval)
    for l in LANGS:
        if models == 'plm':
            try:
                out[l] = rec["eval"][l]["eval_accuracy"]
            except:
                print(f"No evaluation for {l}")
        if models == "slm":
            try:
                out[l] = rec["eval"][l]["accuracy"]
            except:
                print(f"No evaluation for {l}")
        if models == "llm":
            try:
                out[rec['data']] = rec["accuracy"]
            except:
                print(f"No evaluation for {l}")
    return out

files = glob.glob(os.path.join(METRICS_DIR, "model_*.json"))
            
rows = []

if models != 'llm':
    for fp in files:
        rec = load_metrics(fp)
        name = os.path.splitext(os.path.basename(fp))[0]  # e.g., model_1
        model_name = MODEL_MAPPING_REVERSE[rec['model']]
        name = f"{model_name} - {name} - ({rec['data']})".replace("_", "-")
        lang = rec['data']
        accs = get_accuracy_by_lang(rec)
        rows.append((name, accs, lang))
    rows_sorted = sorted(rows, key=lambda x: LANGS.index(x[2]) if x[2] in LANGS+['mix'] else len(LANGS))
else:
    
    mds = defaultdict(dict)
    for fp in files:
        rec = load_metrics(fp)
        name = os.path.splitext(os.path.basename(fp))[0]  # e.g., model_1
        model_name = MODEL_MAPPING_REVERSE[rec['model']]
        name = f"{model_name} - Shots {int(rec['shots'])}".replace("_", "-")
        lang = rec['data']
        acc = rec['accuracy']
        
        mds[name][lang] = acc
    
    rows_sorted = []
    for model, values, in mds.items():
        rows_sorted.append((model, values, None))

print(rows_sorted)


max_by_lang = {}
for lang in LANGS:
    metrics = []
    for r in rows_sorted:
        try:
            metrics.append(r[1][lang])
        except:
            pass
    metrics = [x for x in metrics if x]
    if metrics:
        max_by_lang[lang] = max(metrics)

print(max_by_lang)

# print LaTeX table
table = "\n"
colspec = "l" + "c" * len(LANGS)
header = "Model & " + " & ".join(LANGS) + " \\\\"

table += "\\begin{tabular}{" + colspec + "}\n"
table += "\\hline\n"
table += header + "\n"
table += "\\hline\n"

for name, accs, _ in rows_sorted:
    cells = []
    # find the max value among numbers (ignore None or non-floats)
    vals = [v for v in accs.values() if isinstance(v, (int, float))]

    for l in LANGS:
        try:
            v = accs[l]
            if v == max_by_lang[l]:
                cells.append(f"\\textbf{{{v:.3f}}}")
            else: 
                cells.append(f"{v:.3f}" if isinstance(v, (int, float)) else "--")
        except:
            cells.append("--")
    
        # except:
        #     cells.append("--")
        # if v and v == max_by_lang[l]:
        #     cells.append(f"\\textbf{{{v:.3f}}}")
        # else:
        #     cells.append(f"{v:.3f}" if isinstance(v, (int, float)) else "--")

    table += f"{name} & " + " & ".join(cells) + " \\\\\n"

table += "\\hline\n"
table += "\\end{tabular}\n"

# Save to file
with open(f"{models}.tex", "w", encoding="utf-8") as f:
    f.write(table)

print(table)