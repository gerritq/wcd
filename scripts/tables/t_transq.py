import json
import os


# ----------------------------------------------------------------# configs
# ----------------------------------------------------------------

BASE_DIR = os.getenv("BASE_WCD", ".")
IN_DIR = os.path.join(BASE_DIR, "data/exp2/trans_eval")

LANGS = {"high": ["en", "pt", "de", "ru", "it", "vi", "tr", "nl"],
         "medium": ["uk", "ro", "id", "bg", "uz"],
         "low": ["no", "az", "mk", "hy", "sq"],
         }

def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def collect_data():
    # find all json files in n input dir
    data = {}
    for file_name in os.listdir(IN_DIR):
        if file_name.endswith(".json"):
            file_path = os.path.join(IN_DIR, file_name)
            metrics = load_json(file_path)
            lang = file_name.split(".")[0]
            data[lang] = metrics

    return data

def create_latex_table(data):
    
    # print LaTeX table
    table = "\n\n"
    colspec = "l" + "c" * 4
    
    header = "\\textbf{Language} & \\textbf{BLEU} & \\textbf{ROUGE-1} & \\textbf{ROUGE-2} & \\textbf{BERT Score} \\\\"
    table += "\\begin{tabular}{" + colspec + "}\n"
    table += header + "\n"
    table += "\\toprule\n"

    for lang, metrics in data.items():
        row = f"{lang} & {metrics['bleu']:.2f} & {metrics['rouge1']:.2f} & {metrics['rouge2']:.2f} & {metrics['bertscore']:.2f} \\\\"
        table += row + "\n"

    table += "\\bottomrule\n"
    table += "\\end{tabular}\n"
    table += "\n\n"

    print("\n\n")
    print(table)
    print("\n\n")

def main():
    all_data = collect_data()

    create_latex_table(all_data)

if __name__ == "__main__":
    main()
