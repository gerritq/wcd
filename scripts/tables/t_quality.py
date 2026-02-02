import os 
import json
import random
import pandas as pd
from utils import LANGS, LANG_ORDER, MODEL_DISPLAY_NAMES

# ----------------------------------------------------------------
# configs
# ----------------------------------------------------------------

BASE_DIR = os.getenv("BASE_WCD", ".")
IN_DIR = os.path.join(BASE_DIR, "data/quality")

LANGS = {"high": ["en", "pt", "de", "ru", "it", "vi", "tr", "nl"],
         "medium": ["uk", "ro", "id", "bg", "uz"],
         "low": ["no", "az", "mk", "hy", "sq"],
         }

# ----------------------------------------------------------------
# functions
# ----------------------------------------------------------------

def load_annotated_dataframes():
    df = pd.read_excel(os.path.join(IN_DIR, "mwcd_quality_annotated.xlsx"))
    return df

def prepare_data(df: pd.DataFrame):
    print(df.columns)
    df_resources = (
        df.groupby(["resource"])
          .agg({
              "claim_correct": "mean",
              "label_correct": "mean",
              "context_correct": "mean",
          })
          .reset_index()
    )

    # sort by resource high, medium, low
    resource_order = ["high", "medium", "low"]        
    df_resources["resource"] = pd.Categorical(df_resources["resource"], categories=resource_order, ordered=True)
    df_resources = df_resources.sort_values("resource")

    df_langs = (
        df.groupby(["language"])
          .agg({
              "claim_correct": "mean",
              "label_correct": "mean",
              "context_correct": "mean",
          })
          .reset_index()
    )

    return df_resources, df_langs


def latex_table_langs(df_langs: pd.DataFrame):

    # print LaTeX table
    table = "\n\n"
    colspec = "l" + "c" * 3
    
    header = "\\textbf{Language} & \\textbf{Claim Accuracy (\\%)} & \\textbf{Label Accuracy (\\%)} & \\textbf{Valid Context (\\%)} \\\\"
    table += "\\begin{tabular}{" + colspec + "}\n"
    table += header + "\n"
    table += "\\toprule\n"

    for resource, langs in zip(
    ["high", "medium", "low"],
    [LANGS["high"], LANGS["medium"], LANGS["low"]]
    ):
        
        if resource == "high":
            table += "\\multicolumn{4}{l}{\\textbf{High Resource}} \\\\\n"
            table += "\\midrule\n"
        if resource == "medium":
            table += "\\midrule\n"
            table += "\\multicolumn{4}{l}{\\textbf{Medium Resource}} \\\\\n"
            table += "\\midrule\n"
        elif resource == "low":
            table += "\\midrule\n"
            table += "\\multicolumn{4}{l}{\\textbf{Low Resource}} \\\\\n"
            table += "\\midrule\n"
            

        for lang in langs:
            row = df_langs[df_langs["language"] == lang].iloc[0]
            claim = f"{row['claim_correct']:.2f}"
            label = f"{row['label_correct']:.2f}"
            context = f"{row['context_correct']:.2f}"
            table += f"{lang} & {claim} & {label} & {context} \\\\\n"
        
    table += "\\bottomrule\n"
    table += "\\end{tabular}\n"
    table += "\n\n"
    print(table)


def latex_table_resources(df_resources: pd.DataFrame):

    # print LaTeX table
    table = "\n\n"
    colspec = "l" + "c" * 3
    
    # generate
    header = "\\textbf{Resource} & \\textbf{Valid Claim (\\%)} & \\textbf{Label Accuracy (\\%)} & \\textbf{Valid Context (\\%)} \\\\"
    table += "\\begin{tabular}{" + colspec + "}\n"
    table += header + "\n"
    table += "\\toprule\n"

    for _, row in df_resources.iterrows():
        resource = row["resource"][0].upper() + row["resource"][1:]
        claim = f"{row['claim_correct']:.2f}"
        label = f"{row['label_correct']:.2f}"
        context = f"{row['context_correct']:.2f}"
        table += f"{resource} & {claim} & {label} & {context} \\\\\n"

    table += "\\bottomrule\n"
    table += "\\end{tabular}\n"
    table += "\n\n"
    print(table)


def main():
    # load data
    df = load_annotated_dataframes()

    # prepare dataa
    df_resources, df_langs = prepare_data(df)

    # print LaTeX tables
    latex_table_langs(df_langs)
    latex_table_resources(df_resources)

if __name__ == "__main__":
    main()