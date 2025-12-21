from email import parser
import uuid
from utils.training import evaluate_wrapper
from utils import prompts
from utils.models import build_slm, MODEL_MAPPING
from utils.data import get_data, get_tokenizer, get_cross_lingual_evaluation_data
from utils.utils import get_save_path, get_model_number,find_saved_model_dir

from datetime import datetime
import torch
import argparse
from transformers import set_seed
import os
import json
import tempfile
import copy
from argparse import Namespace

from run import get_save_path, get_model_number

CL_TRAINING_SIZES = [50, 100, 250, 500]

BASE_DIR = os.getenv("BASE_WCD")
EX1_DATA = os.path.join(BASE_DIR, "data/exp1")
EX2_MODELS = os.path.join(BASE_DIR, "data/exp2/models")
EX2_EVAL = os.path.join(BASE_DIR, "data/exp2/eval")


def single_target_zero_shot_evaluation(args: Namespace):

    # Just cofigs
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("="*20)
    print(f"Device {device}")
    print("="*20)

    # find trained model
    model_dir = find_saved_model_dir(args)
    args.model_dir = model_dir


    slm = build_slm(model_type=args.model_type, 
                    model_name=args.model_name,
                    model_dir=args.model_dir,
                    from_checkpoint=args.from_checkpoint,
                    quantization=args.quantization,
    )

    # tokenisers
    tokenizer_train = get_tokenizer(args=args,
                                    inference=False
    )

    tokenizer_test = get_tokenizer(args=args,
                                    inference=True
    )

    # Just for generation
    if not slm.model.config.pad_token_id:
        slm.model.config.pad_token_id = tokenizer_test.pad_token_id
         
         
    train_dataloader, dev_train_dataloader, dev_test_dataloader, test_dataloader, label_dist = get_data(
        args=args, 
        tokenizer_train=tokenizer_train,
        tokenizer_test=tokenizer_test, 
    )
    
    test_metrics = evaluate_wrapper(model_type=args.model_type,
                               model=slm.model,
                               tokenizer=tokenizer_test,
                               dataloader=test_dataloader,)
    
    new_meta = vars(args).copy()
    new_meta['label_dist'] = label_dist
    new_meta['date'] = datetime.now().isoformat()
    new_meta['max_memory'] =  torch.cuda.max_memory_allocated() / (1024 ** 3)
    new_meta['bf16'] = (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    new_meta['test_metrics_0_shot'] = test_metrics


    return new_meta


def zero_shot_evaluation(args: Namespace):

    for target_lang in args.target_langs:
        args.lang = target_lang

        print("="*20)
        print(f"Zero-shot evaluation on target lang: {target_lang}")
        print("="*20)

        meta = single_target_zero_shot_evaluation(args)

        # Save
        save_path = get_save_path(args)
                
        with open(save_path, "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

def main():
    parser = argparse.ArgumentParser()
    # Required
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--atl", type=int, required=True)
    parser.add_argument("--context", type=int, required=True)
    parser.add_argument("--smoke_test", type=int, required=True)
    parser.add_argument("--training_size", type=int, required=True)
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--prompt_template", type=str, required=True)

    # HPs
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--max_grad_norm", type=float, required=True)
    parser.add_argument("--weight_decay", type=float, required=True)
    parser.add_argument("--quantization", type=int, default=1) # we always quantise
    parser.add_argument("--seed", type=int, default=42)

    # CL
    parser.add_argument("--source_langs", nargs='+', default=[])
    parser.add_argument("--target_langs", nargs='+', default=[])
    parser.add_argument("--lang_settings", nargs='+', default=[])
    parser.add_argument("--cl_settings", nargs='+', default=[])
    parser.add_argument("--lang_setting", type=int, default="main")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--from_checkpoint", type=int, default=1)    
    args = parser.parse_args()     

   # Checks
    if args.model_type not in ['slm', 'clf', "plm"]:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    if args.prompt_template not in ["minimal", "instruct", "verbose"]:
        raise ValueError(f"Unknown model prompt template: {args.prompt_template}")

    if not all(x in {"main", "translation"} for x in args.lang_settings):
        raise ValueError(f"Invalid settings: {args.lang_settings}")

    if not all(x in {"0shot", "xshot"} for x in args.cl_settings):
        raise ValueError(f"Invalid settings: {args.cl_settings}")

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


    # setthe load from checpoint flag
    args.from_checkpoint = True

    # Loop through defined settings in args: cl_settings and lang_settings
    for cl_setting in args.cl_settings:
        if args.cl_setting == "0shot":
            for lang_setting in args.lang_settings:
                args.lang_setting = lang_setting
                print("="*20)
                print(f"Starting zero-shot evaluation for lang setting: {lang_setting}")
                print("="*20)
                zero_shot_evaluation(args)

        if args.cl_setting == "xshot":
            raise NotImplementedError("Cross-lingual few-shot not implemented yet.")

if __name__ == "__main__":
    main()