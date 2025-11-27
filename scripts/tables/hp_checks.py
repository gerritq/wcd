import json
import re
from pathlib import Path

ROOT = Path("/scratch/prj/inf_nlg_ai_detection/wcd/data/exp1")

run_re = re.compile(r"run_\d+")
meta_re = re.compile(r"meta_\d+")

results = []

for lang_dir in ROOT.iterdir():
    if not lang_dir.is_dir():
        continue

    for run_dir in lang_dir.iterdir():
        if not (run_dir.is_dir() and run_re.fullmatch(run_dir.name)):
            continue

        best_for_run = None

        for meta_path in run_dir.iterdir():
            if not (meta_path.is_file() and meta_re.match(meta_path.stem)):
                continue

            with open(meta_path) as f:
                meta = json.load(f) 

            hps = f"LR: {meta['learning_rate']} | MaxGradNorm {meta['max_grad_norm']}"

            dev_metrics = meta.get("dev_metrics", [])
            test_metrics = meta.get("test_metrics", [])

            # pick best dev inside this meta file
            for dev_entry in dev_metrics:
                epoch = dev_entry["epoch"]
                dev_acc = dev_entry["metrics"]["accuracy"]

                if best_for_run is None or dev_acc > best_for_run["dev_acc"]:
                    # find matching test metric for same epoch
                    test_entry = next(
                        (t for t in test_metrics if t["epoch"] == epoch),
                        None
                    )

                    if "prompt_extension" in meta.keys():
                        pe = meta["prompt_extension"]
                    else:
                        pe = "not defined"
                    best_for_run = {
                        "run_dir": str(run_dir),
                        "meta_file": str(meta_path),
                        'lang': meta['lang'],
                        'atl': meta['atl'],
                        'context': meta['context'],
                        'prompt_extension': pe,
                        "hps": hps,
                        "epoch": epoch,
                        "dev_acc": dev_acc,
                        "dev_f1": dev_entry["metrics"]["f1"],
                        "test_metrics": test_entry["metrics"] if test_entry else None,

                    }

        if best_for_run:
            results.append(best_for_run)

# print all run-wise best results
for r in results:
    print("\n========================================")
    print(f"Run:          {r['run_dir']}")
    print(f"Meta file:    {r['meta_file']}")
    print(f"Best epoch:   {r['epoch']}")
    print(f"Lang:         {r['lang']}")
    print(f"ATL:          {r['atl']}")
    print(f"Context:      {r['context']}")
    print(f"Prompt Extension:      {r['prompt_extension']}")
    print(f"HPs:          {r['hps']}")
    print(f"Dev acc:      {r['dev_acc']:.6f}")
    print(f"Dev f1:       {r['dev_f1']:.6f}")
    print("Test metrics:", r["test_metrics"])