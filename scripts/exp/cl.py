from email import parser
import uuid
from utils.training import evalaute_cross_lingual
from utils import prompts
from utils.models import build_slm, MODEL_MAPPING
from utils.data import get_tokenizer, get_cross_lingual_evaluation_data

from datetime import datetime
import torch
import argparse
from transformers import set_seed
import os
import json
import tempfile
import copy

from run import get_save_path, get_model_number

BASE_DIR = os.getenv("BASE_WCD")
EX2_MODELS = os.path.join(BASE_DIR, "data/exp2/models")
EX2_EVAL = os.path.join(BASE_DIR, "data/exp2/eval")

def single_target_language_evaluation(args):

    # Get saving path
    save_path = get_save_path(args)
            
    meta = vars(args).copy()

    set_seed(args.seed)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("="*20)
    print(f"Device {device}")
    print("="*20)

    print("="*20)
    print("HP SETTINGS")
    for k, v in meta.items():
        print(f"{k}: {v}")
    print("="*20)
    
    # this should load the model from the checkpoint
    slm = build_slm(model_type=args.model_type, 
                    model_name=args.model_name,
                    model_dir=args.model_dir,
                    from_checkpoint=args.from_checkpoint,
                    quantization=args.quantization,
    )

    # load test tokenizer
    tokenizer_test = get_tokenizer(args=args,
                                    inference=True
    )

    # Just for generation
    if not slm.model.config.pad_token_id:
        slm.model.config.pad_token_id = tokenizer_test.pad_token_id
         
    test_dataloader,label_dist = get_cross_lingual_evaluation_data (
                                        args=args, 
                                        tokenizer_test=tokenizer_test, 
    )
    
    test_metrics, duration = evalaute_cross_lingual (
                                        args=args,
                                        model=slm.model,
                                        tokenizer_test=tokenizer_test,
                                        test_dataloader=test_dataloader
                                        )

    meta['label_dist'] = label_dist
    meta['date'] = datetime.now().isoformat()
    meta['duration'] = duration
    meta['max_memory'] =  torch.cuda.max_memory_allocated() / (1024 ** 3)
    meta['bf16'] = (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    meta['test_metrics'] = test_metrics
    meta['model_checkpoint_dir'] = args.model_dir

    with open(save_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return meta

def main():
    parser = argparse.ArgumentParser()
    # Required
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--atl", type=int, required=True)
    parser.add_argument("--context", type=int, required=True)
    parser.add_argument("--smoke_test", type=int, required=True)
    parser.add_argument("--prompt_template", type=str, required=True)

    # HPs
    parser.add_argument("--quantization", type=int, default=1) # we always quantise

    # Eval
    parser.add_argument("--experiment", type=str, default="cl_eval")
    parser.add_argument("--source_langs", nargs='+', default=[])
    parser.add_argument("--target_lang", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--from_checkpoint", type=int, default=0)    
    args = parser.parse_args()     

   # Checks
    if args.model_type not in ['slm', 'clf', "plm"]:
        raise ValueError(f"Unknown model type: {args.model_type}")
    if args.prompt_template not in ["minimal", "instruct", "verbose"]:
        raise ValueError(f"Unknown model type: {args.model_type}")

    assert (
        args.smoke_test in [0, 1]
        and args.quantization in [0, 1]
        and args.context in [0, 1]
        and args.atl in [0, 1]
        and args.from_checkpoint in [0, 1]
        ), "Incorrect boolean values"

    args.smoke_test = bool(args.smoke_test)
    args.quantization = bool(args.quantization)
    args.context = bool(args.context)
    args.atl = bool(args.atl)
    args.from_checkpoint = bool(args.from_checkpoint)

    # Select the HF model name
    suffix = "_base" if args.model_type == "clf" else ""
    args.model_name = MODEL_MAPPING[args.model_name+suffix]

    # Run eval
    single_target_language_evaluation(args)

if __name__ == "__main__":
    main()