from utils.training import train
from utils import prompts
from utils.models import build_slm, MODEL_MAPPING
from utils.data import get_data, get_tokenizer
from utils.utils import get_save_path,find_best_hp_run

from datetime import datetime
import torch
import argparse
from transformers import set_seed
import os
import json
import tempfile
import copy
"""
- classifier class needs different data processing

"""

BASE_DIR = os.getenv("BASE_WCD")
EX1 = os.path.join(BASE_DIR, "data/exp1")
EX2 = os.path.join(BASE_DIR, "data/exp2")

def single_stage_training(args):

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
    print(f"GPUs available: {torch.cuda.device_count()}")
    print("="*20)

    print("="*20)
    print("HP SETTINGS")
    for k, v in meta.items():
        print(f"{k}: {v}")
    print("="*20)
    
    slm = build_slm(model_type=args.model_type, 
                    model_name=args.model_name,
                    model_dir=args.model_dir,
                    from_checkpoint=args.from_checkpoint,
                    quantization=args.quantization,
    )

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
    
    train_loss, dev_loss, dev_metrics, test_metrics, duration = train(args=args,
                                                        model=slm.model, 
                                                        tokenizer_test=tokenizer_test,
                                                        train_dataloader=train_dataloader,
                                                        dev_train_dataloader=dev_train_dataloader,
                                                        dev_test_dataloader=dev_test_dataloader,
                                                        test_dataloader=test_dataloader
    )
    
    meta['label_dist'] = label_dist
    meta['date'] = datetime.now().isoformat()
    meta['duration'] = duration
    meta['max_memory'] =  torch.cuda.max_memory_allocated() / (1024 ** 3)
    meta['bf16'] = (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    meta['train_loss'] = train_loss
    meta['dev_loss'] = dev_loss
    meta['dev_metrics'] = dev_metrics
    meta['test_metrics'] = test_metrics

    # save model
    if args.save_checkpoint:
        slm.model.save_pretrained(args.model_dir)

    with open(save_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return meta

# def two_stage_training(args):
    
#     stage1_args = copy.deepcopy(args)
#     print("="*20)
#     print(f"STAGE 1: SOURCE LANGUAGE TRAINING  {args.training_langs}")
#     print("="*20)

#     # add saving checkpoint flag
#     stage1_args.save_checkpoint = True

#     # create a random folder with tempfile
#     model_dir = os.path.join(EX2, stage1_args.test_lang)
#     os.makedirs(model_dir, exist_ok=True)
#     model_dir = tempfile.mkdtemp(dir=model_dir)
#     stage1_args.model_dir = model_dir
        
#     stage_1_meta = single_stage_training(stage1_args)
    
#     print("="*20)
#     print(f"STAGE 2: EVALUATION ON TARGET LANGUAGE  {args.test_lang}")
#     print("="*20)

#     # Define args for stage 2
#     stage2_args = copy.deepcopy(args)
#     stage2_args.experiment = "second_stage"
#     stage2_args.model_dir = model_dir # this is the local dir where the first stage model is saved
#     stage2_args.from_checkpoint = True # flag to load from checkpoint   
    
#     for training_size in CL_TRAINING_SIZES:
#         stage2_args.training_size = training_size
#         print("="*20)
#         print(f"Fine-tuning with training size: {training_size} ---")
#         stage_2_meta = single_stage_training(stage2_args)

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
    parser.add_argument("--lang", type=str, default="")
    parser.add_argument("--run_dir", type=str, default="")
    parser.add_argument("--model_dir", type=str, default="")
    parser.add_argument("--notes", type=str, default="")
    parser.add_argument("--train_log_step", type=int, default=20)
    parser.add_argument("--prompt_extension", type=str, default="")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metric", type=str, default="f1")
    parser.add_argument("--setting", type=str, default="main")
    parser.add_argument("--lang_setting", type=str, default="main")
    
    # EXP2
    parser.add_argument("--source_langs", type=str, default="")
    parser.add_argument("--target_langs", type=str, default="")
    parser.add_argument("--lang_settings", type=str, default="")
    parser.add_argument("--cl_settings", type=str, default="")
    parser.add_argument("--save_checkpoint", type=int, default=0)
    parser.add_argument("--from_checkpoint", type=int, default=0)
    
    args = parser.parse_args()        
    
    # Checks
    if args.model_type not in ['slm', 'clf', "plm"]:
        raise ValueError(f"Unknown model type: {args.model_type}")
    if args.prompt_template not in ["minimal", "instruct", "verbose"]:
        raise ValueError(f"Unknown model type: {args.model_type}")

    # need to reduce batch size with verbose prompts
    if args.prompt_template == "verbose":
        args.batch_size = 16
        args.max_length = 768
        print("="*20)
        print("REDUCED BATCH SIZE TO {args.batch_size} AND INCREASED MAX LENGTH TO {args.max_length}")
        print("="*20)

    assert (
        args.smoke_test in [0, 1]
        and args.quantization in [0, 1]
        and args.context in [0, 1]
        and args.atl in [0, 1]
        and args.save_checkpoint in [0, 1]
        and args.from_checkpoint in [0, 1]
        ), "Incorrect boolean values"

    args.smoke_test = bool(args.smoke_test)
    args.quantization = bool(args.quantization)
    args.context = bool(args.context)
    args.atl = bool(args.atl)
    args.save_checkpoint = bool(args.save_checkpoint)
    args.from_checkpoint = bool(args.from_checkpoint)

    # create lists form strings
    args.source_langs = args.source_langs.split() if args.source_langs else []
    args.target_langs = args.target_langs.split() if args.target_langs else []
    args.lang_settings = args.lang_settings.split() if args.lang_settings else []
    args.cl_settings = args.cl_settings.split() if args.cl_settings else []

    # Select the HF model name
    suffix = "_base" if args.model_type == "clf" else ""
    args.model_name = MODEL_MAPPING[args.model_name+suffix]

    # Select the exp version
    if args.experiment in ["binary"]: # this is the full hp run
        single_stage_training(args)

    # This is the seed run, finding optimal HP first
    if args.experiment in ["seed", "save"]:
        # first find optimal hp
        optimal_hp_config = find_best_hp_run(args=args)
        if not optimal_hp_config:
            raise ValueError(f"No optimal HP config found for {args.model_type} - {args.model_name} - {args.atl}.")
        
        # assign optimal hp to args
        args.epochs = optimal_hp_config['epochs']
        args.learning_rate = optimal_hp_config['learning_rate']
        args.max_grad_norm = optimal_hp_config['max_grad_norm']
        args.weight_decay = optimal_hp_config['weight_decay']
        args.batch_size = optimal_hp_config['batch_size']

        print("=" * 20)
        print("Optimal HP configuration found and updated args:")
        for k, _ in optimal_hp_config.items():
            print(f"{k}: {getattr(args, k)}")
        print("=" * 20)

        if args.experiment == "save":
            # create an model dir for saing
            model_dir = os.path.join(EX2, "models")
            os.makedirs(model_dir, exist_ok=True)
            model_dir = tempfile.mkdtemp(dir=model_dir, prefix=f"run_")
            args.model_dir = model_dir

            # add the save checkpoint flag
            args.save_checkpoint = True

        # run 
        single_stage_training(args)


if __name__ == "__main__":
    main()