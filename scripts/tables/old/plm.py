import os
import re
import json
import glob

BASE_DIR = os.getenv("BASE_WCD")
METRICS_DIR = os.path.join(BASE_DIR, "data/metrics/plm")

MODEL_NAME_MAP = {
    "google-bert/bert-base-multilingual-uncased": "mBERT",
    "FacebookAI/xlm-roberta-base": "XLM-R",
    "FacebookAI/xlm-roberta-large" : "xlm-r-l",
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
        out[l] = (rec.get("eval", {}).get(l, {}) or {}).get("eval_accuracy")
    return out

files = sorted(glob.glob(os.path.join(METRICS_DIR, "model_*.json")),
               key=lambda p: int(re.search(r"model_(\d+)\.json$", p).group(1)) if re.search(r"model_(\d+)\.json$", p) else 1e9)

rows = []
for fp in files:
    rec = load_metrics(fp)
    name = os.path.splitext(os.path.basename(fp))[0]  # e.g., model_1
    model_name = MODEL_NAME_MAP[rec['model']]
    name = f"{model_name} - {name} - ({rec['data']})".replace("_", "-")
    accs = get_accuracy_by_lang(rec)
    rows.append((name, accs))

# print LaTeX table
output_path = "plm.tex"

table = ""
colspec = "l" + "c" * len(LANGS)
header = "Model & " + " & ".join(LANGS) + " \\\\"

table += "\\begin{tabular}{" + colspec + "}\n"
table += "\\hline\n"
table += header + "\n"
table += "\\hline\n"

for name, accs in rows:
    cells = []
    for l in LANGS:
        v = accs[l]
        cells.append(f"{v:.3f}" if isinstance(v, (int, float)) else "--")
    table += f"{name} & " + " & ".join(cells) + " \\\\\n"

table += "\\hline\n"
table += "\\end{tabular}\n"

# Save to file
with open("plm.tex", "w", encoding="utf-8") as f:
    f.write(table)

print(table)