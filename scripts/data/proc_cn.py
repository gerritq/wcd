import csv
import json
import os
from copy import deepcopy
from nltk.tokenize import sent_tokenize

INPUT_PATHS = [("/scratch/prj/inf_nlg_ai_detection/wcd/data/raw/en_wiki_subset_statements_no_citations_sample.txt", 0 ),
               ("/scratch/prj/inf_nlg_ai_detection/wcd/data/raw/en_wiki_subset_statements_all_citations_sample.txt", 1)]
OUTPUT_PATH = "/scratch/prj/inf_nlg_ai_detection/wcd/data/sets"

def main():

    # OG DATA
    data = []
    multiple_sentences = {0: 0, 1: 0}
    for input_path, label in INPUT_PATHS:
        with open(input_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                row['claim'] = row.pop("statement")
                row['label'] = label
                n_sentences = len(sent_tokenize(row['claim']))
                row['n_sentences'] = n_sentences

                if n_sentences > 1:

                    multiple_sentences[label]+=1

                data.append(row)

    with open(os.path.join(OUTPUT_PATH, 'cn_fa.jsonl'), "w", encoding="utf-8") as f_out:
        for item in data:
            json.dump(item, f_out)
            f_out.write("\n")
    # print(data[0])
    # print(data[1])
    
    print(" === Main data ===")
    print("N", len(data))
    print("Statements with multiple sentences: ", multiple_sentences)
    print("")

    # DATA WITH SINGLE SENTENCES
    data_ss = deepcopy(data)

    multiple_sentences={0: 0 , 1: 0}
    for x in data_ss:
        if x['n_sentences'] > 1:
            x['claim'] = sent_tokenize(x['claim'])[0]
            multiple_sentences[x['label']] += 1

    with open(os.path.join(OUTPUT_PATH, 'cn_fa_ss.jsonl'), "w", encoding="utf-8") as f_out:
        for item in data_ss:
            json.dump(item, f_out)
            f_out.write("\n")

    print(" === With single sentences data ===")
    print("N affected: ", multiple_sentences)
    print("")

    # DATA WITH SINGLE SENTENCES & NOT LEAD SECTION

    data_ss_nl = deepcopy(data_ss)
    lead_section = {0: 0 , 1: 0}
    data_out = []

    for x in data_ss_nl:
        if x['section'] == "MAIN_SECTION":
            lead_section[x['label']] += 1
            continue
        else:
            data_out.append(x)

    with open(os.path.join(OUTPUT_PATH, 'cn_fa_ss_nl.jsonl'), "w", encoding="utf-8") as f_out:
        for item in data_out:
            json.dump(item, f_out)
            f_out.write("\n")


    print(" === With single sentences & no lead data ===")
    print("N", len(data_out))
    print("N lead sections dropped: ", lead_section)

    print("Label 1 remaining: ", len([x for x in data_out if x['label'] == 1]))
    print("Label 0 remaining: ", len([x for x in data_out if x['label'] == 0]))
    assert len([x for x in data_out if x['section'] == "MAIN_SECTION"]) == 0, "still has lead section"
    
    print("")

if __name__ == "__main__":
    main()