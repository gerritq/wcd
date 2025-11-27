from transformers import (
    set_seed,
    TrainingArguments,
    Trainer,
)
from models import build_slm

from utils import (evaluation,
                   collect_and_save_losses,
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

    # HPs
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--max_grad_norm", type=float, required=True)
    parser.add_argument("--gradient_accumulation_steps", type=int, required=True)

    args = parser.parse_args()
    
    assert (args.smoke_test in [0,1] and
            args.quantization in [0,1] and
            args.context in [0,1] and
            args.save_last_epoch in [0,1],), "Incorrect boolean values"

    if args.model_type not in ['vanilla', 'atl', 'classifier']:
        raise ValueError(f"Unknown model type: {model_type}")
    args.smoke_test = bool(args.smoke_test)
    args.quantization = bool(args.quantization)
    args.context = bool(args.context)
    args.save_last_epoch = bool(args.save_last_epoch)

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
        eval_steps=60,
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

    del trainer
    torch.cuda.empty_cache()
    
    # Run eval
    checkpoints = collect_checkpoints(TMP_DIR, args.epochs, args.save_last_epoch)

    # Free up some space
    del trainer
    torch.cuda.empty_cache()

    for i, checkpoint in enumerate(checkpoints):
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
                                    explanation=False,
                                    bnb_config=slm.bnb_config)
            
            test_metrics = evaluation(model_type=args.model_type,
                                    ds= slm.test_tok,
                                    model_path=meta['path'],
                                    tokenizer_eval=slm.tokenizer_eval,
                                    smoke_test=slm.smoke_test,
                                    explanation=False,
                                    bnb_config=slm.bnb_config)
        else:
            
            dev_metrics = trainer.evaluate(slm.dev_tok)
            test_metrics = trainer.evaluate(slm.test_tok)
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

        model_number = get_model_number(slm.model_dir)
        meta['model_number'] = model_number

        # Neat solution here if there is a race:
        # Use mode 'x' and the FileExistsError https://docs.python.org/3/library/functions.html#open

        with open(os.path.join(slm.model_dir, f"meta_{model_number}.json"), "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        print("="*10)
        print("RUN SUCCESSFULL.")
        print("Test Metrics:")
        print(meta['test_metrics'])
        
    # add loss plot
    # collect_and_save_losses(meta['loss_history'], model_path)
    ##################################################################

    # # save model
    # model_number = get_model_number(slm.model_dir)
    # model_path = os.path.join(slm.model_dir, f"model_{model_number}")
    # trainer.save_model(model_path)
    # meta['loss_history'] = trainer.state.log_history


    # # Free space for non quant inference
    # # del trainer
    # # torch.cuda.empty_cache()

    # # run eval
    # if args.model_type != "classifier":
    #     dev_metrics = evaluation(ds=slm.dev_test_tok,
    #                             model_path=model_path,
    #                             tokenizer_eval=slm.tokenizer_eval,
    #                             smoke_test=slm.smoke_test,
    #                             explanation=slm.explanation,
    #                             bnb_config=slm.bnb_config)

    #     # # get eval loss
    #     # dev_train_metrics = trainer.evaluate(slm.dev_train_tok)
    #     # dev_metrics.update(dev_train_metrics)
        
    #     test_metrics = evaluation(ds=slm.test_tok,
    #                             model_path=model_path,
    #                             smoke_test=slm.smoke_test,
    #                             explanation=slm.explanation,
    #                             tokenizer_eval=slm.tokenizer_eval,
    #                             bnb_config=slm.bnb_config)
    # else:
    #     dev_metrics = trainer.evaluate(slm.dev_tok)
    #     test_metrics = trainer.evaluate(slm.test_tok)

    # end = time.time()

    # # Add to meta    
    # meta['training_data_n'] = len(slm.train_tok)
    # meta['dev_data_n'] = len(slm.dev_train_tok)
    # meta['test_data_n'] = len(slm.test_tok)
    # meta['dev_metrics'] = dev_metrics
    # meta['test_metrics'] = test_metrics

    # meta['model_number'] = model_number
    # meta['time_mins'] = (end - start) / 60.0
    # meta['date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # meta['max_memory_allocated'] = torch.cuda.max_memory_allocated() / 1024**2

    # with open(os.path.join(model_path, "meta.json"), "w") as f:
    #     json.dump(meta, f, indent=2, ensure_ascii=False)

    # # add loss plot
    # collect_and_save_losses(meta['loss_history'], model_path)


if __name__ == "__main__":
    run_fine_tuning()