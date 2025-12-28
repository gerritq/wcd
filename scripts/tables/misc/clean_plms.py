"""clean plm runs with low learning rate; not needed"""

import os
import json
import shutil

BASE_DIR = os.getenv("BASE_WCD")
EX1_DATA = os.path.join(BASE_DIR, "data/exp2/eval")


def remove_empty_dirs(root: str):
    for dirpath, dirnames, filenames in os.walk(root, topdown=False):
        if not dirnames and not filenames:
            os.rmdir(dirpath)
            print(f"Removed empty dir: {dirpath}")

def remove_plm_with_low_lr(root: str, lr_threshold: float = 1e-5):
    for dirpath, dirnames, filenames in os.walk(root):
    
        meta_files = sorted(
            f for f in filenames if f.startswith("meta_")
        )

        if len(meta_files) != 4:
            continue

        meta_1_path = os.path.join(dirpath, meta_files[0])
        with open(meta_1_path, "rb") as f:
            meta_1 = json.load(f)


        if meta_1['model_type'] == "plm" and meta_1["lower_lr"] is True:
            print(f"[DELETE] Removing directory: {dirpath}")
            shutil.rmtree(dirpath) 

def main():
    remove_empty_dirs(EX1_DATA)
    remove_plm_with_low_lr(EX1_DATA)

if __name__ == "__main__":
    main()