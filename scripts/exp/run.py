from utils.training import train
from utils import prompts
from utils.models import build_slm, MODEL_MAPPING
from utils.data import get_data, get_tokenizer

from datetime import datetime
import torch
import argparse
from transformers import set_seed
import os
import json
"""
- classifier class needs different data processing

"""

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--lang", type=str, required=True)
    parser.add_argument("--annotation_type", type=str, default="")
    parser.add_argument("--quantization", type=int, required=True)
    parser.add_argument("--smoke_test", type=int, required=True)
    parser.add_argument("--training_size", type=int, required=True)
    parser.add_argument("--context", type=int, required=True)
    parser.add_argument("--notes", type=str, required=True)
    parser.add_argument("--explanation", type=str, default="none")
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--max_grad_norm", type=float, required=True)
    parser.add_argument("--weight_decay", type=float, required=True)
    parser.add_argument("--atl", type=int, required=True)
    parser.add_argument("--train_log_step", type=int, default=20)
    parser.add_argument("--prompt_extension", type=str, default="")
    args = parser.parse_args()

    # checks and turn to bool
    assert (
        args.smoke_test in [0, 1]
        and args.quantization in [0, 1]
        and args.context in [0, 1]
        and args.atl in [0, 1]
        ), "Incorrect boolean values"

    if args.model_type not in ['slm', 'classifier']:
        raise ValueError(f"Unknown model type: {model_type}")

    args.smoke_test = bool(args.smoke_test)
    args.quantization = bool(args.quantization)
    args.context = bool(args.context)

    # Create flag for whether eval needs moer tokens
    args.evaluation_explanation_flag = True if args.explanation == "basic" else False

    # Get hf model name
    args.model_name = MODEL_MAPPING[args.model_name]
    meta = vars(args).copy()

    set_seed(42)

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
    
    slm = build_slm(model_type=args.model_type, 
                    model_name=args.model_name,
                    quantization=args.quantization,
    )

    tokenizer_train = get_tokenizer(model_type=args.model_type, 
                                    model_name=args.model_name,
                                    inference=False
    )

    tokenizer_test = get_tokenizer(model_type=args.model_type, 
                                    model_name=args.model_name,
                                    inference=True
    )

    # Just for generation
    if not slm.model.config.pad_token_id:
        slm.model.config.pad_token_id = tokenizer_test.pad_token_id

    train_dataloader, dev_train_dataloader, dev_test_dataloader, test_dataloader = get_data(
        
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
    
    meta['date'] = datetime.now().isoformat()
    meta['duration'] = duration
    meta['max_memory'] =  torch.cuda.max_memory_allocated() / (1024 ** 3)
    meta['train_loss'] = train_loss
    meta['dev_loss'] = dev_loss
    meta['dev_metrics'] = dev_metrics
    meta['test_metrics'] = test_metrics

    model_number = get_model_number(args.run_dir)
    save_path = os.path.join(args.run_dir, f"meta_{model_number}.json")
    with open(save_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()