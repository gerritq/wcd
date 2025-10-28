import csv
import json
import os
import pandas as pd

BASE_DIR = os.getenv("BASE_WCD")
DATA_DIR = os.path.join(BASE_DIR, "data/raw/cn")
SETS_DIR = os.path.join(BASE_DIR, "data/sets/cn")

def load_raw_data():
    data = []
    
    files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(".txt")]
    for file in files:
        if "all_citations" in file:
            label = 1
        else:
            label = 0
        with open(file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                row['label'] = label
                data.append(row)
    
    return pd.DataFrame(data)

def load_annotated_data():
    path = os.path.join(DATA_DIR, "en_wiki_subset_statements_all_citations_sample_with_labels.csv")
    df = pd.read_csv(path, sep="\t", encoding="utf-8")  
    # on_bad_lines='skip'
    return df
def main():

    data = load_raw_data()
    annotated_data = load_annotated_data()
    print(annotated_data.columns)
    merged = pd.merge(data, annotated_data, on="statement", how="left")

    print(merged.shape)
    print("NA vote 1", merged["vote1"].isna().sum())

    out_path = os.path.join(SETS_DIR, "cn.csv")
    merged.to_csv(out_path)

if __name__ == "__main__":
    main()