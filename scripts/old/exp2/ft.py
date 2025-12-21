from transformers import (
    set_seed,
    TrainingArguments,
    Trainer,
)
from models import build_slm

from utils import (evaluation,
                   collect_and_save_losses,
                   get_run_number,
                   get_model_number,
                   compute_metrics_classification,
                   collect_checkpoints)
import time
from datetime import datetime
import argparse
import os
import json
import torch
import tempfile
import sys
from tqdm import tqdm
BASE_DIR = os.getenv("BASE_WCD")
TMP_DIR = os.path.join(BASE_DIR, ".tmp")
os.makedirs(TMP_DIR, exist_ok=True)
TMP_DIR = tempfile.mkdtemp(prefix="tmp_", dir=TMP_DIR)

set_seed(42)

"""

Current version does not work for the classification head ..

"""

def run_fine_tuning():

    start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--lang", type=str, required=True)
    parser.add_argument("--quantization", type=int, required=True)
    parser.add_argument("--smoke_test", type=int, required=True)
    parser.add_argument("--training_size", type=int, required=True)
    parser.add_argument("--context", type=int, required=True)
    parser.add_argument("--notes", type=str, required=True)
    parser.add_argument("--save_last_epoch", type=int, required=True)
    parser.add_argument("--explanation", type=str, required=True)
    parser.add_argument("--run_number", type=str, required=True)

    # HPs
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--max_grad_norm", type=float, required=True)
    parser.add_argument("--gradient_accumulation_steps", type=int, required=True)

    args = parser.parse_args()
    
    assert (
        args.smoke_test in [0, 1]
        and args.quantization in [0, 1]
        and args.context in [0, 1]
        and args.save_last_epoch in [0, 1]
    ), "Incorrect boolean values"

    if args.model_type not in ['vanilla', 'atl', 'classifier']:
        raise ValueError(f"Unknown model type: {model_type}")
    args.smoke_test = bool(args.smoke_test)
    args.quantization = bool(args.quantization)
    args.context = bool(args.context)
    args.save_last_epoch = bool(args.save_last_epoch)

    # Create flag for whether eval needs moer tokens
    args.evaluation_explanation_flag = True if args.explanation == "basic" else False

    # create meta
    base_meta = vars(args).copy()

    print("="*20)
    print("HP SETTINGS")
    for k, v in base_meta.items():
        print(f"{k}: {v}")
    print("="*20)

    # Load the mdoel
    slm = build_slm(model_type=args.model_type, 
                    model_name=args.model_name,
                    lang=args.lang,
                    quantization=args.quantization,
                    training_size=args.training_size,
                    context=args.context,
                    smoke_test=args.smoke_test,
                    explanation=args.explanation,
                    )

    # Training args
    training_args = TrainingArguments(
        output_dir=TMP_DIR,
        save_strategy="epoch",
        report_to="none",
        logging_strategy="steps",
        logging_steps=20,
        lr_scheduler_type="cosine",
        eval_strategy="steps",
        eval_steps=args.training_size // (2 * args.batch_size), # eval every half epoch
        warmup_ratio=0.03,
        weight_decay=0.01,
        gradient_accumulation_steps=args.gradient_accumulation_steps, # we keep this alwatys the same
        gradient_checkpointing=False, # set with peft when loading the model
        bf16=torch.cuda.is_bf16_supported(),
        per_device_eval_batch_size=16,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        max_grad_norm=args.max_grad_norm,
        )

    
    metrics_fn = compute_metrics_classification if slm.model_type == "classifier" else None

    trainer = Trainer(
        model=slm.model,
        train_dataset=slm.train_tok,
        eval_dataset=slm.dev_train_tok,
        args=training_args,
        compute_metrics=metrics_fn,
    )

    trainer.train()
    
    # Run eval
    checkpoints = collect_checkpoints(TMP_DIR, args.epochs, args.save_last_epoch)

    # Free up some space
    log_history = trainer.state.log_history
    del trainer
    torch.cuda.empty_cache()
    
    all_meta = {}
    for i, checkpoint in enumerate(checkpoints, start=1):
        print("="*20)
        print(f"Evaluating checkpoint {i}")
        print("="*20)
        
        meta = base_meta.copy()
        meta.update(checkpoint)

        # run eval
        if args.model_type != "classifier":    
            dev_metrics = evaluation(model_type=args.model_type,
                                    ds=slm.dev_test_tok,
                                    model_path=meta['path'],
                                    tokenizer_eval=slm.tokenizer_eval,
                                    smoke_test=slm.smoke_test,
                                    explanation=args.evaluation_explanation_flag,
                                    bnb_config=slm.bnb_config)
            
            test_metrics = evaluation(model_type=args.model_type,
                                    ds= slm.test_tok,
                                    model_path=meta['path'],
                                    tokenizer_eval=slm.tokenizer_eval,
                                    smoke_test=slm.smoke_test,
                                    explanation=args.evaluation_explanation_flag,
                                    bnb_config=slm.bnb_config)
        else:
            pass
            # dev_metrics = trainer.evaluate(slm.dev_tok)
            # test_metrics = trainer.evaluate(slm.test_tok)
        end = time.time()

        

        # Add to meta    
        meta['training_data_n'] = len(slm.train_tok)
        meta['dev_data_n'] = len(slm.dev_train_tok)
        meta['test_data_n'] = len(slm.test_tok)
        meta['dev_metrics'] = dev_metrics
        meta['test_metrics'] = test_metrics

        meta['time_mins'] = (end - start) / 60.0
        meta['date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        meta['max_memory_allocated'] = torch.cuda.max_memory_allocated() / 1024**2

        # save_path = os.path.join(save_dir, f"meta_{model_number}.json")
        # # Neat solution here if there is a race:
        # # Use mode 'x' and the FileExistsError https://docs.python.org/3/library/functions.html#open

        # with open(save_path, "w") as f:
        #     json.dump(meta, f, indent=2, ensure_ascii=False)

        all_meta[f"epoch_{i}"] = meta
        print("="*10)
        print("RUN SUCCESSFULL.")
        print("Test Metrics:")
        print(meta['test_metrics'])
    
    # Save metrics all in the same run
    save_dir = os.path.join(slm.model_dir, f"run_{args.run_number}")
    os.makedirs(save_dir, exist_ok=True)
    
    # Save meta in the current run
    model_number = get_model_number(save_dir)
    meta['model_number'] = model_number
    
    all_meta['log_history'] = log_history
    save_path = os.path.join(save_dir, f"meta_{model_number}.json")
    with open(save_path, "w") as f:
        json.dump(all_meta, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    run_fine_tuning()