import os
import re
import torch
import json
from argparse import Namespace
from datasets import load_from_disk, Dataset
from transformers import set_seed

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'exp'))
from utils.models import build_slm, MODEL_MAPPING
from utils.data import get_tokenizer, ds_apply_chat_templates, preprocess_function_generation, init_eval_collate_fn
from utils.training import evaluate_wrapper
from utils import prompts
from torch.utils.data import DataLoader

# ------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------

BASE_DIR = os.getenv("BASE_WCD")
DATA_DIR = os.path.join(BASE_DIR, "data/sets/main")
MODEL_DIR = "/scratch/prj/inf_nlg_ai_detection/wcd/data/exp2/models/run_79ydn6gi"

# Device setup
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("="*20)
print(f"Device: {device}")
print("="*20)

# ------------------------------------------------------------------------
# Data functions
# ------------------------------------------------------------------------

def mask_numbers_in_claim(example: dict) -> dict:
    """
    Mask all numbers in the 'claim' field with [MASK].
    """
    claim = example["claim"]
    # Replace all sequences of digits with [MASK]
    masked_claim = re.sub(r'\d+', '[MASK]', claim)
    example["claim"] = masked_claim
    return example


def load_test_data_with_masked_claims(lang: str) -> Dataset:
    """
    Load test set from data/sets/main/{lang} and mask all numbers in claims.
    Returns only the test set.
    """
    data_path = os.path.join(DATA_DIR, lang)
    
    print("="*20)
    print(f"Loading data from: {data_path}")
    print("="*20)
    
    ds = load_from_disk(data_path)
    test = ds["test"]
    
    # Mask numbers in claims
    test = test.map(mask_numbers_in_claim)


    print("="*20)
    print("EXAMPLES")
    idx=0
    for i, x in enumerate(test):
        print(x['claim'])
        if "[MASK]" in x['claim']:
            idx+=1
        if idx==5:
            break
    print("="*20)
    
    print("="*20)
    print(f"Loaded and masked {len(test)} test examples")
    print("="*20)
    
    return test


# ------------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------------
def tokenize_fn(example: dict, 
                tokenizer
                ):
    """
    Basic tok function.
    Returns input_ids, am, and labels.
    """
    # enc = tokenizer(
    #     example["text"],
    #     truncation=True,
    #     max_length=max_length,
    #     padding="max_length",
    #     return_attention_mask=True,
    # )
    enc = tokenizer(
            example["text"],
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_attention_mask=True,
        )
    enc["labels"] = enc["input_ids"].copy()
    return enc    

def evaluate_masked_claims(args: Namespace):
    """
    Evaluate a trained model on test data with masked numbers in claims.
    Similar to zero_shot_evaluation but with modified test set.
    """
    
    # Load model
    print("="*20)
    print(f"Loading model from: {MODEL_DIR}")
    print("="*20)
    
    slm = build_slm(
        model_type=args.model_type,
        model_name=args.model_name,
        model_dir=MODEL_DIR,
        from_checkpoint=True,
        quantization=args.quantization,
    )
    slm.model.to(device)
    
    # Load tokenizers
    tokenizer_test = get_tokenizer(args=args, inference=True)
    
    # Set pad token for generation
    if not slm.model.config.pad_token_id:
        slm.model.config.pad_token_id = tokenizer_test.pad_token_id
    
    # Load test data with masked claims
    test_ds = load_test_data_with_masked_claims(args.lang)
    
    # Apply prompt template
    prompt = prompts.INSTRUCT_PROMPT
    prompt['user'] = prompt['user_context']

    test_ds = ds_apply_chat_templates(
        ds=test_ds,
        tokenizer=tokenizer_test,
        prompt_template=prompt,
        preprocess_function=preprocess_function_generation
    )

    test_tok = test_ds.map(tokenize_fn, 
                                    fn_kwargs={"tokenizer": tokenizer_test}, 
                                               batched=False
                                           )
    
    # Create dataloader
    eval_collate_fn = init_eval_collate_fn(tokenizer_test)
    test_dataloader = DataLoader(
        test_tok,
        batch_size=8,  # EVAL_BATCH from data.py
        collate_fn=eval_collate_fn,
        shuffle=False
    )
    
    # Evaluate
    print("="*20)
    print("Starting evaluation on masked test set")
    print("="*20)
    
    test_metrics = evaluate_wrapper(
        model_type=args.model_type,
        model=slm.model,
        tokenizer_test=tokenizer_test,
        dataloader=test_dataloader,
    )
    
    # Prepare results
    results = {
        "model_dir": MODEL_DIR,
        "lang": args.lang,
        "model_type": args.model_type,
        "model_name": args.model_name,
        "test_metrics_masked": test_metrics,
        "max_memory_gb": torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0,
    }
    
    print("="*20)
    print("Evaluation complete!")
    print(f"Results: {json.dumps(test_metrics, indent=2)}")
    print("="*20)
    
    return results


# ------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------

def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, required=True, help="Model type: slm, clf, or plm")
    parser.add_argument("--model_name", type=str, required=True, help="Model name (e.g., llama3_8b)")
    parser.add_argument("--lang", type=str, required=True, help="Language code (e.g., en, pt)")
    parser.add_argument("--context", type=int, default=1, help="Use context (0 or 1)")
    parser.add_argument("--quantization", type=int, default=1, help="Use quantization (0 or 1)")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Convert boolean flags
    args.context = bool(args.context)
    args.quantization = bool(args.quantization)
    
    # Set seed
    set_seed(args.seed)
    
    # Map model name to HF path
    suffix = "_base" if args.model_type == "clf" else ""
    args.model_name = MODEL_MAPPING[args.model_name + suffix]
    
    print("="*20)
    print("Configuration:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print("="*20)
    
    # Run evaluation
    results = evaluate_masked_claims(args)
    
    # Save results (optional)
    output_path = f"nums_results/nums_eval_{args.lang}_{args.model_type}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()