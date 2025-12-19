import os 
import json
import random
import pandas as pd
from datasets import load_from_disk,concatenate_datasets
# ----------------------------------------------------------------
# configs
# ----------------------------------------------------------------

# ------------------------------------------------------------------------------------------
# configs
# ------------------------------------------------------------------------------------------
BASE_DIR = os.getenv("BASE_WCD")
DATA_DIR = os.path.join(BASE_DIR, "data/sets/main")

EXAMPLE_LANGS = ["en", "de"] # "uz", "id", "no", "sq"

random.seed(42)


# ----------------------------------------------------------------
# functions
# ----------------------------------------------------------------
def load_and_sample_Data(lang: str) -> list[dict]:
    """Load annotated data from a JSONL file into a DataFrame."""
    data_dir = os.path.join(DATA_DIR, lang)
    ds = load_from_disk(data_dir)

    all_data = []

    for s in [ds["train"], ds["dev"], ds["test"]]:
        all_data.extend(list(s))
        
    pos = [item for item in all_data if item['label'] == 1 and item['previous_sentence'] is not None and item['subsequent_sentence'] is not None]
    neg = [item for item in all_data if item['label'] == 0 and item['previous_sentence'] is not None and item['subsequent_sentence'] is not None]

    sample = pos[:1] + neg[:1]

    return sample

def collect_all_data() -> list[dict]:
    all_samples = []
    for lang in EXAMPLE_LANGS:
        samples = load_and_sample_Data(lang)
        all_samples.extend(samples)
    return all_samples


def latex_table_examples(all_samples: list[dict]):

    # print LaTeX table
    table = "\n\n"
    colspec = "l" + "p{2cm} p{1cm} p{3cm} p{3cm} p{3cm} c"
    
    header = "\\textbf{Language} & \\textbf{Title}  & \\textbf{Section} & \\textbf{Claim} & \\textbf{Previous Sentence} &  \\textbf{Subsequent Sentence} & \\textbf{Label} \\\\"
    table += "\\begin{tabular}{" + colspec + "}\n"
    table += header + "\n"
    table += "\\midrule\n"

    prev_lang = ""
    for i, item in enumerate(all_samples):
        lang = item["lang"]
        title = item["title"]
        section = item["section"]
        claim = item["claim"]
        previous_sentence = item["previous_sentence"]
        subsequent_sentence = item["subsequent_sentence"]
        label = item["label"]
        if lang == prev_lang:
            lang = ""
            table += f" & {title} & {section} & {claim} & {previous_sentence} & {subsequent_sentence} & {label} \\\\\n"
            if i != len(all_samples) - 1:
                table += "\\cmidrule(lr){2-7}\n"
        else:
            table += f"\\multirow{{2}}{{*}}{{{lang}}} & {title} & {section} & {claim} & {previous_sentence} & {subsequent_sentence} & {label} \\\\\n"
            prev_lang = lang

    table += "\\bottomrule\n"
    table += "\\end{tabular}\n"
    table += "\n\n"

    table = table.replace("%", "\\%")
    print("\n\n")
    print(table)
    print("\n\n")

def main():
    all_samples = collect_all_data()
    latex_table_examples(all_samples)

if __name__ == "__main__":
    main()