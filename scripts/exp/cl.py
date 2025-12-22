
from utils.training import evaluate_wrapper
from utils import prompts
from utils.models import build_slm, MODEL_MAPPING
from utils.data import get_data, get_tokenizer
from utils.utils import get_save_path, get_model_number,find_saved_model_dir
from transformers import PreTrainedTokenizerBase

from datetime import datetime
import torch
import argparse
from transformers import set_seed
import os
import json
import copy
from argparse import Namespace
from run import single_stage_training

# ------------------------------------------------------------------------
# configs
# ------------------------------------------------------------------------

CL_TRAINING_SIZES = [50, 100, 250, 500]

BASE_DIR = os.getenv("BASE_WCD")
EX1_DATA = os.path.join(BASE_DIR, "data/exp1")
EX2_MODELS = os.path.join(BASE_DIR, "data/exp2/models")
EX2_EVAL = os.path.join(BASE_DIR, "data/exp2/eval")

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

# ------------------------------------------------------------------------
# zero shot
# ------------------------------------------------------------------------

def single_target_zero_shot_evaluation(args: Namespace,
                                       slm: torch.nn.Module,
                                       tokenizer_train: PreTrainedTokenizerBase,
                                       tokenizer_test: PreTrainedTokenizerBase,
    ) -> dict:
         
         
    train_dataloader, dev_train_dataloader, dev_test_dataloader, test_dataloader, label_dist = get_data(
        args=args, 
        tokenizer_train=tokenizer_train,
        tokenizer_test=tokenizer_test, 
    )
    
    test_metrics = evaluate_wrapper(model_type=args.model_type,
                               model=slm.model,
                               tokenizer_test=tokenizer_test,
                               dataloader=test_dataloader,)
    
    new_meta = vars(args).copy()
    new_meta['label_dist'] = label_dist
    new_meta['date'] = datetime.now().isoformat()
    new_meta['max_memory'] =  torch.cuda.max_memory_allocated() / (1024 ** 3)
    new_meta['bf16'] = (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    new_meta['test_metrics_0_shot'] = test_metrics


    return new_meta


def zero_shot_evaluation(args: Namespace):

    # find trained model
    model_dir, _ = find_saved_model_dir(args)
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


    for target_lang in args.target_langs:
        args.lang = target_lang

        print("="*20)
        print(f"Zero-shot evaluation on target lang: {target_lang}")
        print("="*20)

        meta = single_target_zero_shot_evaluation(args=args, 
                                                  slm=slm, 
                                                  tokenizer_train=tokenizer_train, 
                                                  tokenizer_test=tokenizer_test)

        # Save
        save_path = get_save_path(args)
                
        with open(save_path, "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)


# ------------------------------------------------------------------------
# few shot
# ------------------------------------------------------------------------


def few_shot_evaluation(args: Namespace):

    # find trained model
    model_dir, optimal_hps = find_saved_model_dir(args)
    
    # update args with optimal hps
    args.learning_rate = optimal_hps["learning_rate"]
    args.max_grad_norm = optimal_hps["max_grad_norm"]
    args.weight_decay = optimal_hps["weight_decay"]
    args.epochs = optimal_hps["epochs"]
    args.batch_size = optimal_hps["batch_size"]
    args.model_dir = model_dir

    if args.lower_lr:
        args.learning_rate = 1e-6
        print("="*20)
        print(f"Lowering learning rate to {args.learning_rate} for few-shot training")
        print("="*20)

    # loop over target langs
    for target_lang in args.target_langs:
        args.lang = target_lang

        print("="*20)
        print(f"Few-shot evaluation on target lang: {target_lang}")
        print("="*20)

        # saves internally
        _ = single_target_few_shot_evaluation(args=args)

def single_target_few_shot_evaluation(args: Namespace):

    few_shot_args = copy.deepcopy(args)
    
    for training_size in CL_TRAINING_SIZES:
        few_shot_args.training_size = training_size

        print("="*20)
        print(f"Few-shot training for {args.lang} with {training_size} samples")
        print("="*20)

        _ = single_stage_training(args=few_shot_args)


def run_x_shot(args: Namespace):
    
    # for both settting zero and few shot
    for cl_setting in args.cl_settings:
        args.cl_setting = cl_setting
        
        # loop over seeds
        for seed in args.seeds:
            set_seed(int(seed))
            args.seed = int(seed)
            print("="*20)
            print(f"Set seed to {args.seed}")

            # zero shot
            if cl_setting == "zero":
                # loop over translation or wild
                for lang_setting in args.lang_settings:
                    args.lang_setting = lang_setting
                    args.cl_setting = cl_setting
                    print(f"Starting zero-shot evaluation for lang setting: {lang_setting}")
                    print("="*20)
                    zero_shot_evaluation(args)
            
            # few shot
            if cl_setting == "few":
                # loop over translation or wild
                for lang_setting in args.lang_settings:
                    args.lang_setting = lang_setting
                    args.cl_setting = cl_setting
                    print(f"Starting few-shot evaluation for lang setting: {lang_setting}")
                    print("="*20)
                    few_shot_evaluation(args)

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

    # Defaults
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--metric", type=str, default="f1")
    parser.add_argument("--lang_setting", type=str, default="main")
    parser.add_argument("--lang", type=str, default="")
    

    # CL
    parser.add_argument("--source_langs", type=str, default="")
    parser.add_argument("--target_langs", type=str, default="")
    parser.add_argument("--lang_settings", type=str, default="")
    parser.add_argument("--cl_settings", type=str, default="")
    parser.add_argument("--seeds", type=str, default="42")
    parser.add_argument("--from_checkpoint", type=int, default=1)    
    parser.add_argument("--save_checkpoint", type=int, default=0)
    parser.add_argument("--lower_lr", type=int, default=0)
    args = parser.parse_args()     



   # Checks
    if args.model_type not in ['slm', 'clf', "plm"]:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    if args.prompt_template not in ["minimal", "instruct", "verbose"]:
        raise ValueError(f"Unknown model prompt template: {args.prompt_template}")

    assert (
        args.smoke_test in [0, 1]
        and args.quantization in [0, 1]
        and args.context in [0, 1]
        and args.atl in [0, 1]
        and args.from_checkpoint in [0, 1]
        and args.save_checkpoint in [0, 1]
        and args.lower_lr in [0, 1]
        ), "Incorrect boolean values"

    args.smoke_test = bool(args.smoke_test)
    args.quantization = bool(args.quantization)
    args.context = bool(args.context)
    args.atl = bool(args.atl)
    args.from_checkpoint = bool(args.from_checkpoint)
    args.save_checkpoint = bool(args.save_checkpoint)
    args.lower_lr = bool(args.lower_lr)


    # create lists form strings
    args.source_langs = args.source_langs.split() if args.source_langs else []
    args.target_langs = args.target_langs.split() if args.target_langs else []
    args.lang_settings = args.lang_settings.split() if args.lang_settings else []
    args.cl_settings = args.cl_settings.split() if args.cl_settings else []
    args.seeds = args.seeds.split() if args.seeds else []

    if not all(x in {"main", "translation"} for x in args.lang_settings):
        raise ValueError(f"Invalid settings: {args.lang_settings}")

    if not all(x in {"zero", "few"} for x in args.cl_settings):
        raise ValueError(f"Invalid settings: {args.cl_settings}")

    # Select the HF model name
    suffix = "_base" if args.model_type == "clf" else ""
    args.model_name = MODEL_MAPPING[args.model_name+suffix]


    # ensure this is always loading ffrom checkpoint
    args.from_checkpoint = True

    run_x_shot(args=args)

if __name__ == "__main__":
    main()