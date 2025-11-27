import os
import json
from tqdm import tqdm

# directory path
DIR = "/scratch/prj/inf_nlg_ai_detection/wcd/data/metrics/llm"

def main():
    files = [f for f in os.listdir(DIR) if f.endswith(".json")]
    print(f"Found {len(files)} JSON files in {DIR}")

    for fname in tqdm(files):
        path = os.path.join(DIR, fname)

        # load file
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # rename key if it exists
        data['eval'] = {data["data"]: data['eval']}
        
        # save back
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()