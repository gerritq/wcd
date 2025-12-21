from transformers import DataCollatorWithPadding
from transformers import (
    set_seed,
    TrainingArguments,
    Trainer,
)
from models import build_plm

from utils import (compute_metrics,
                   collect_and_save_losses,
                   get_model_number)
import time
from datetime import datetime
import argparse
import os
import json
import torch

set_seed(42)

def run_fine_tuning():
    
    start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--lang", type=str, required=True)
    parser.add_argument("--training_size", type=int, required=True)
    parser.add_argument("--smoke_test", type=int, required=True)
    parser.add_argument("--context", type=int, required=True)
    parser.add_argument("--notes", type=str, required=True)

    # HPs
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--batch_size", type=int, required=True)

    args = parser.parse_args()
    
    assert (args.smoke_test in [0,1] and 
            args.context in [0,1]), "Incorrect boolean values"

    args.smoke_test = bool(args.smoke_test)
    args.context = bool(args.context)

    # create meta
    meta = vars(args)
    print("="*20)
    print("HP SETTINGS")
    print(meta)
    print("="*20)
    
    # Load the mdoel
    plm = build_plm(model_type=args.model_type, 
                    model_name=args.model_name,
                    lang=args.lang,
                    training_size=args.training_size,
                    smoke_test=args.smoke_test,
                    context=args.context
                    )

    # Use data collator and pass tokenizer to trainer
    data_collator = DataCollatorWithPadding(tokenizer=plm.tokenizer, padding=True)

    # Training args
    training_args = TrainingArguments(
        output_dir=None,
        report_to="none",
        logging_strategy="steps",
        logging_steps=20,
        lr_scheduler_type="cosine",
        eval_strategy="steps",
        eval_steps=60,
        warmup_ratio=0.03,
        gradient_accumulation_steps=1, # no need for grad acc for plms
        gradient_checkpointing=False,
        bf16=torch.cuda.is_bf16_supported(),
        per_device_eval_batch_size=16,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        )

    trainer = Trainer(
        model=plm.model,
        args=training_args,
        train_dataset=plm.train_tok,
        eval_dataset=plm.dev_tok,
        compute_metrics=compute_metrics,
        processing_class=plm.tokenizer,
        data_collator=data_collator
    )

    trainer.train()

    dev_metrics = trainer.evaluate(eval_dataset=plm.dev_tok)
    test_metrics = trainer.evaluate(eval_dataset=plm.test_tok)

    # save model
    model_number = get_model_number(plm.model_dir)
    # model_path = os.path.join(plm.model_dir, f"model_{model_number}")
    # trainer.save_model(model_path)

    end = time.time()

    # Add to meta    
    meta['dev_metrics'] = dev_metrics
    meta['test_metrics'] = test_metrics

    meta['model_number'] = model_number
    meta['time_mins'] = (end - start) / 60.0
    meta['date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    meta['loss_history'] = trainer.state.log_history
    
    with open(os.path.join(plm.model_dir, f"meta_{model_number}.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # add loss plot
    # collect_and_save_losses(trainer.state.log_history, model_path)

    print("="*10)
    print("RUN SUCCESSFULL.")
    print("Test Metrics:")
    print(test_metrics)

if __name__ == "__main__":
    run_fine_tuning()