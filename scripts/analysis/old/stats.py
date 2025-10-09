import os
import numpy as np
import re
import json

BASE_DIR="/scratch/prj/inf_nlg_ai_detection/wcd"
INPUT_DATA=['data/sets/en_sents.jsonl',
            'data/sets/pt_sents.jsonl',
            'data/sets/hu_sents.jsonl',
            'data/sets/pl_sents.jsonl', 
            'data/sets/cn_fa.jsonl',
            'data/sets/cn_fa_ss.jsonl',
            'data/sets/cn_fa_ss_nl.jsonl']

def has_citation(text):
    return int(bool(re.search(r'\[\d+\]', text)))

def stats(name, data):
    char_len = []
    token_len = []
    ends_with_period = 0
    has_cit=0
    labels={0: 0, 1: 0}

    for item in data:
        claim = item['claim']
        char_len.append(len(claim))
        token_len.append(len(claim.split()))
        if name.endswith("sents"):
            labels[item['label_2']] +=1
        else:
            labels[item['label']] +=1
        if claim[-1] in ['.', '!', '?']:
            ends_with_period +=1
        has_cit += has_citation(claim)
        

    print(f" === {name} === ")
    print(f"N {len(data)}")
    print(f"Char mean {np.mean(char_len):.2f} std {np.std(char_len):.2f}")
    print(f"Token mean {np.mean(token_len):.2f} std {np.std(token_len):.2f}")
    print(f"N has period {ends_with_period}")
    print(f"N has citation {has_cit}")
    print("Label distribution:", labels)
    print("")

def main():
    for input_data in INPUT_DATA:
        name = input_data.split('/')[-1].replace(".jsonl", "")
        input_path = os.path.join(BASE_DIR, input_data)
        with open(input_path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
    
        stats(name, data)

if __name__ == "__main__":
    main()

        
        