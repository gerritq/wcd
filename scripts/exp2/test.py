from datasets import load_from_disk

def head_data(path):
    ds = load_from_disk(path)
    print("TRAIN:", ds["train"][:3])
    print("DEV:", ds["dev"][:3])
    print("TEST:", ds["test"][:3])

if __name__ == "__main__":
    head_data("/scratch/prj/inf_nlg_ai_detection/wcd/data/sets/annotation/en")
    print("")
    head_data("/scratch/prj/inf_nlg_ai_detection/wcd/data/sets/main/en")