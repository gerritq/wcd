import os
import re
import json
import glob

BASE_DIR = os.getenv("BASE_WCD")
METRICS_DIR = os.path.join(BASE_DIR, "data/metrics/slm")

MODEL_NAME_MAP = {
    "meta-llama/Llama-3.2-1B-Instruct": "llama3_1b",
    "meta-llama/Llama-3.2-3B-Instruct": "llama3_3b",
    "meta-llama/Llama-3.1-8B-Instruct": "llama3_8b",
    "meta-llama/Llama-3.3-70B-Instruct": "llama3_70b",
    "Qwen/Qwen3-0.6B": "qwen_06b",
    "Qwen/Qwen3-4B-Instruct-2507": "qwen3_4b",
    "Qwen/Qwen3-8B": "qwen3_8b",
    "Qwen/Qwen3-30B-A3B-Instruct-2507": "qwen3_30b",
    "Qwen/Qwen3-32B": "qwen3_32b"
}

LANGS = ["en","nl","no","it","pt","ro","ru","uk","bg","id"]

def load_metrics(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_accuracy_by_lang(rec):
    out = {l: None for l in LANGS}
    data_lang = rec.get("data")
    # own language accuracy (from test_eval)
    for l in LANGS:
        out[l] = (rec.get("eval", {}).get(l, {}) or {}).get("accuracy")
    return out

files = sorted(glob.glob(os.path.join(METRICS_DIR, "model_*.json")),
               key=lambda p: int(re.search(r"model_(\d+)\.json$", p).group(1)) if re.search(r"model_(\d+)\.json$", p) else 1e9)

rows = []
for fp in files:
    rec = load_metrics(fp)
    name = os.path.splitext(os.path.basename(fp))[0]  # e.g., model_1
    model_name = MODEL_NAME_MAP[rec['model']]
    name = f"{model_name} - {name} - ({rec['data']})"
    lang = rec['data']
    accs = get_accuracy_by_lang(rec)
    rows.append((name, accs, lang))

rows_sorted = sorted(rows, key=lambda x: LANGS.index(x[2]) if x[2] in LANGS+['mix'] else len(LANGS))

# print LaTeX table
table = ""
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
    max_val = max(vals) if vals else None

    for l in LANGS:
        v = accs[l]
        if isinstance(v, (int, float)) and v == max_val:
            cells.append(f"\\textbf{{{v:.3f}}}")
        else:
            cells.append(f"{v:.3f}" if isinstance(v, (int, float)) else "--")

    table += f"{name} & " + " & ".join(cells) + " \\\\\n"

table += "\\hline\n"
table += "\\end{tabular}\n"

# Save to file
with open("slm.tex", "w", encoding="utf-8") as f:
    f.write(table)

print(table)