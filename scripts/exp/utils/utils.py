import json
import os
from pathlib import Path
from argparse import Namespace

BASE_DIR = os.getenv("BASE_WCD")
EX1 = os.path.join(BASE_DIR, "data/exp1")
EX2 = os.path.join(BASE_DIR, "data/exp2")

def load_metrics(path):
    """"Load a single meta_file"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def find_saved_model_dir(args: Namespace) -> str:
    """
    Find trained model in exp2/models that matches the args.
    Return the model dir path.
    """

    model_dir  = Path(os.path.join(EX2, "models"))
    all_model_runs = [d for d in model_dir.glob("run_*") if d.is_dir()]

    found_model_dir = None
    for single_run in all_model_runs:
        meta_file = single_run / "meta_1.json"
        if not meta_file.exists():
            continue

        meta_1 = load_metrics(meta_file)

        if (not args.model_type == meta_1["model_type"] or 
            not args.model_name == meta_1["model_name"] or
            not args.atl == meta_1["atl"] or
            not args.seed == meta_1["seed"] or
            not args.source_langs == meta_1["source_langs"]
            ):
            continue

        found_model_dir = meta_1["model_dir"]
        
        if found_model_dir:
            optimal_hps = {"learning_rate": meta_1["learning_rate"],
                                     "max_grad_norm": meta_1["max_grad_norm"],
                                     "weight_decay": meta_1["weight_decay"],
                                     "epochs": meta_1["epochs"],
                                     "batch_size": meta_1["batch_size"],}
        print("="*20)
        print(f"Found saved model dir: {single_run}")
        print("="*20)        
    
    if not found_model_dir:
        raise ValueError("Could not find specified model.")
    return found_model_dir, optimal_hps

def find_best_hp(args: Namespace,
                 all_meta_file_paths: list[Path]) -> dict:
    """
    Takes a list of all meta file paths of a run dir.
    Finds the best hp according to the args.
    Then returns a dict with the best hp configs.

    """


    best_dev_metric = float('-inf')
    optimal_hp_config = None

    for meta_file in all_meta_file_paths:
        meta_tmp = load_metrics(meta_file)

        # loop dev and test and find the optimal 
        for dev_metrics in meta_tmp["dev_metrics"]:
            dev_value = dev_metrics["metrics"][args.metric]

            if dev_value > best_dev_metric:
                best_dev_metric = dev_value
                optimal_hp_config = {"learning_rate": meta_tmp["learning_rate"],
                                     "max_grad_norm": meta_tmp["max_grad_norm"],
                                     "weight_decay": meta_tmp["weight_decay"],
                                     "epochs": dev_metrics["epoch"],
                                     "batch_size": meta_tmp["batch_size"],}
            
    return optimal_hp_config

def find_best_hp_run(args: Namespace,
                     ) -> dict:
    """
    Takes the dir where all results are stored. 
    Find the best hp run according to the args.
    Returns a dict with the best hyperparameter configurations
    """
    language_dir = Path(os.path.join(EX1, args.lang))
    all_language_runs = list(language_dir.glob("run_*"))
    optimal_hp_dir = None
    optimal_hp_config = None

    n_check = 0
    for single_run in all_language_runs:
        
        all_meta_file_paths = list(single_run.glob("meta_*.json"))

        # skip if run dir is empty
        if not all_meta_file_paths:
            continue

        # skip if run dir has no the number of expected hp files (6 for both plm and slm)
        if len(all_meta_file_paths) != 6:
            continue

        # sort meta files by number
        all_meta_file_paths = sorted(all_meta_file_paths, key=lambda x: int(x.stem.split("_")[-1]))

        meta_1 = load_metrics(all_meta_file_paths[0])

        # now skip based on args
        if (not args.model_type == meta_1["model_type"] or 
            not args.model_name == meta_1["model_name"] or
            not args.atl == meta_1["atl"] or
            not args.lang == meta_1["lang"] or
            not meta_1["seed"] == 42 # this is the seed used during hp tuning            
            ):
            continue
        
        n_check += 1
        optimal_hp_config = find_best_hp(args=args, all_meta_file_paths=all_meta_file_paths)
        optimal_hp_dir = single_run
        print("="*20)
        print(f"Optimal HP run dir: {single_run}")
        print("="*20)        
        
    if n_check > 1:
        raise ValueError("Multiple matching runs found.")


    if optimal_hp_config:
        return optimal_hp_config
    return None


def get_save_path(args: Namespace) -> str:
    if args.smoke_test:
        if args.experiment in ["binary", "seed"]:
            test_dir = os.path.join(EX1, "smoke_test")

        if args.experiment in ["save"]:
            test_dir = args.model_dir

        if args.experiment in ["cl"]:
            test_dir = os.path.join(EX2, "eval", "smoke_test")

        # Get model number and return test dir
        os.makedirs(test_dir, exist_ok=True)
        model_number = get_model_number(test_dir)
        save_path = os.path.join(test_dir, f"meta_{model_number}.json")
        return save_path


    # BINARY EXPERIMENT: this is the full hp search run for plms,slms,and clfs
    if args.experiment in ["binary", "seed"]:
        if not args.run_dir:
            raise ValueError("For experiment 1 run dir must be given.")
        model_number = get_model_number(args.run_dir)
        save_path = os.path.join(args.run_dir, f"meta_{model_number}.json")

    # SAVE MODEL FOR CL
    if args.experiment == "save":
        if not args.model_dir:
            raise ValueError("For experiment 2 model dir must be given.")
        model_number = get_model_number(args.model_dir)
        save_path = os.path.join(args.model_dir, f"meta_{model_number}.json")

    # CL EVALUATION
    if args.experiment == "cl_eval":
        save_dir = os.path.join(EX2, "eval", args.lang)
        os.makedirs(save_dir, exist_ok=True)
        model_number = get_model_number(save_dir)
        save_path = os.path.join(save_dir, f"meta_{model_number}.json")
        
    directory = os.path.dirname(save_path)
    os.makedirs(directory, exist_ok=True)

    return save_path

def get_model_number(model_dir: str) -> int:
    """
    Identifies the latest meta_<n>.json file in model_dir.
    Returns the next available number.
    """
    meta_files = [f for f in os.listdir(model_dir) if f.startswith("meta_") and f.endswith(".json")]

    numbers = []
    for fname in meta_files:
        num = int(fname.split("_")[1].split(".")[0])
        numbers.append(num)

    return max(numbers) + 1 if numbers else 1