import os
import re
import json
import argparse
import time
from datetime import datetime
import torch
import numpy as np
from utils import (
                    MODEL_MAPPING, 
                    append_meta_file, 
                    get_model_number,
                    collect_and_save_losses,
                    evaluation_non_tok,
                    get_tokenizer,
                    get_train_dev,
                    get_test,
                    get_max_sequence_length,
                    preprocess_function,
                    pwl_tokenizer,
                    get_assistant_tag,
                    check_labels
                    )
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
import optuna
from peft import LoraConfig, get_peft_model, AutoPeftModelForCausalLM
import tempfile
from prompts import SYSTEM_PROMPTS_SLM
from transformers import EarlyStoppingCallback

"""
can pass custom tokeniser to collator@!!!!!


"""
set_seed(42)

parser = argparse.ArgumentParser()
parser.add_argument("--lang", type=str, required=True)
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--trials", type=int, required=True)
parser.add_argument("--pwl", type=int, required=True)
parser.add_argument("--smoke_test", type=int, required=True)
args = parser.parse_args()
args.model = MODEL_MAPPING[args.model]
args.smoke_test = bool(args.smoke_test)
args.pwl = bool(args.pwl)

BASE_DIR = os.getenv("BASE_WCD")
DATA_DIR = os.path.join(BASE_DIR, "data/sets/main")
if args.pwl:
    MODEL_DIR = os.path.join(BASE_DIR, "data/models/slm/pwl_hp")
    TEMP_DIR = os.path.join(MODEL_DIR, ".rnd_pwl_hp")
    os.makedirs(TEMP_DIR, exist_ok=True)
    TEMP_DIR = tempfile.mkdtemp(dir=TEMP_DIR)
else:
    MODEL_DIR = os.path.join(BASE_DIR, "data/models/slm/van_hp")
    TEMP_DIR = os.path.join(MODEL_DIR, ".rnd_van_hp")
    os.makedirs(TEMP_DIR, exist_ok=True)
    TEMP_DIR = tempfile.mkdtemp(dir=TEMP_DIR)


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    # print(logits.shape) # torch.Size([16, 194, 151936]) -- dev_len, max_seq_len, vocab leng
    return logits.argmax(dim=-1)


decode_tokenizer =  get_tokenizer(args, inference=True)

label_re = re.compile(r'["\']label["\']\s*:\s*([01])')

def extract_label(txt):
    m = label_re.search(txt)
    if not m:
        return None
    return int(m.group(1))

def compute_metrics(eval_preds):
    pred_ids, label_ids = eval_preds 
    if isinstance(pred_ids, tuple):
        pred_ids = pred_ids[0]    
    
    total = len(pred_ids)
    valid = 0

    preds_bin, golds_bin = [], []
    for p, y in zip(pred_ids, label_ids):
        mask = (y != -100)
        y_span = y[mask]
        p_span = p[mask]

        gold_txt = decode_tokenizer.decode(y_span, skip_special_tokens=True)
        pred_txt = decode_tokenizer.decode(p_span, skip_special_tokens=True)
        print("gold", gold_txt)
        print("\nprediction", pred_txt)
        print("---")
        g = extract_label(gold_txt)
        p_ = extract_label(pred_txt)

        print("gold label", g)
        print("\nprediction label", p_)
        
        if g is None or p_ is None:
            continue
        valid += 1
        golds_bin.append(g)
        preds_bin.append(p_)

    acc = float((np.array(preds_bin) == np.array(golds_bin)).mean()) if (valid == total and total !=0) else 0.0
    return {"accuracy": acc, "valid": valid, "total": total}

# def compute_metrics(eval_preds):
#     preds, labels = eval_preds

#     if isinstance(preds, tuple):
#         preds = preds[0]

#     # Replace -100 in the preds as we can't decode them
#     preds = np.where(preds != -100, preds, decode_tokenizer.pad_token_id)
#     preds = np.where(labels != -100, preds, decode_tokenizer.pad_token_id)

#     # Decode
#     decoded_preds = decode_tokenizer.batch_decode(preds, skip_special_tokens=True)

#     print(decoded_preds[0])
#     print(labels[0])

    # # Replace -100 in the labels as we can't decode them
    # labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
    # # Decode reference summaries into text
    # decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
    # # ROUGE expects a newline after each sentence
    # decoded_preds = ["\n".join(pred.strip()) for pred in decoded_preds]

    # decoded_labels = ["\n".join(label.strip()) for label in decoded_labels]
    # # Compute ROUGscores
    # result = rouge_score.compute(
    #     predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    # )
    # # Extract the median scores
    # result = {key: value * 100 for key, value in result.items()}
    # return {k: round(v, 4) for k, v in result.items()}


def get_train_dev_tok_full_loss(args, tokenizer):
    """returns the tokenised train and dev sets with the (default) full loss"""

    def tokenize_fn(example, tokenizer, max_length):
        enc = tokenizer(
            example["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_attention_mask=True,
        )
        enc["labels"] = enc["input_ids"].copy()
        return enc

    # executation starts here
    # get the data, already in chat format!
    train, dev = get_train_dev(args, tokenizer)
    print("\tTraining instance:", train['text'][0])
    
    # get the max seq length per set
    train_max_seq_length = get_max_sequence_length(train, tokenizer)
    dev_max_seq_length = get_max_sequence_length(dev, tokenizer)
    print("\ttrain max seq length", train_max_seq_length)
    print("\tdev max seq length", dev_max_seq_length)
    
    # tok data
    train_tok = train.map(tokenize_fn,
                         fn_kwargs={
                            "tokenizer": tokenizer,
                            "max_length": train_max_seq_length,
                            },
                          batched=False, 
                          remove_columns=train.column_names)
    dev_tok = dev.map(tokenize_fn, 
                        fn_kwargs={
                            "tokenizer": tokenizer,
                            "max_length": dev_max_seq_length,
                            },
                        batched=False, 
                        remove_columns=dev.column_names)

    return train_tok, dev_tok

def get_train_dev_tok_pwl(args, tokenizer):
    """returns the tokenised train and dev sets with assistant loss (pwl)"""
            
    # execution starts here
    # get train and test
    train, dev = get_train_dev(args, tokenizer)
    print("\tTraining instance:", train['text'][0])

    # find assistant tag and ids
    assistant_tag, assistant_tag_ids = get_assistant_tag(tokenizer)

    # get max seq length
    train_max_seq_length = get_max_sequence_length(train, tokenizer)
    dev_max_seq_length = get_max_sequence_length(dev, tokenizer)
    print("\ttrain max seq length", train_max_seq_length)
    print("\tdev max seq length", dev_max_seq_length)

    # tpokenise data with the custom pwl tokeniser
    train_tok = train.map(pwl_tokenizer,
                            fn_kwargs={"tokenizer": tokenizer,
                            "assistant_tag_ids": assistant_tag_ids,
                            "max_seq_length": train_max_seq_length},
                            remove_columns=train.column_names
                        )
    dev_tok = dev.map(pwl_tokenizer,
                            fn_kwargs={"tokenizer": tokenizer,
                            "assistant_tag_ids": assistant_tag_ids,
                            "max_seq_length": dev_max_seq_length},
                        remove_columns=dev.column_names
                        )

    return train_tok, dev_tok
    
    
def get_configs(args, tokenizer):

    bnb_config = BitsAndBytesConfig(
                                load_in_4bit=True, 
                                bnb_4bit_use_double_quant=True, 
                                bnb_4bit_quant_type="nf4", 
                                bnb_4bit_compute_dtype=torch.bfloat16
                                )

    lora_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=.05,
            r=16,
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
    )

    def optuna_hp_space(trial):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True),
            "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [4, 8]),
            "max_grad_norm": trial.suggest_categorical("max_grad_norm", [0.1, 0.3, 0.5, 0.8, 1, 1.2]),
            "weight_decay": trial.suggest_categorical("weight_decay", [0.0, 0.001, 0.01, 0.1]),
            "num_train_epochs": trial.suggest_categorical("num_train_epochs", [1, 2, 3]),
        }
        

    def model_init(trial):

        base = AutoModelForCausalLM.from_pretrained(
                args.model,
                quantization_config=bnb_config,
                trust_remote_code=True,
                )
        model = get_peft_model(base, lora_config)
        model.print_trainable_parameters()
        model.gradient_checkpointing_enable()  # PEFT + checkpointing
        # This does the trick to enable check pointing: https://discuss.huggingface.co/t/i-used-to-have-no-problem-with-peft-fine-tuning-after-hundreds-of-trainings-but-now-i-have-encountered-the-error-runtimeerror-element-0-of-tensors-does-not-require-grad-and-does-not-have-a-grad-fn/168829/3
        model.enable_input_require_grads()  
        model.config.use_cache = False         # disable cache
        return model
    
    # for load best model and early stoppping, it is important that we set eval __and__ save 
    # strategy to epoch; we also need to the metric for best model and lower is better
    training_args = TrainingArguments(
        output_dir=TEMP_DIR,
        report_to="none",
        logging_strategy="steps",
        logging_steps=20,
        lr_scheduler_type="linear",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        warmup_ratio=0.03,
        gradient_accumulation_steps=4, # we keep this alwatys the same
        gradient_checkpointing=False, # set with peft when loading the model
        bf16=True, 
        per_device_eval_batch_size=16,
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy", 
        greater_is_better=True,
        )

    return training_args, model_init, optuna_hp_space, bnb_config


def find_and_save_model(best_run, model_dir):
    run_dir = os.path.join(TEMP_DIR, f"run-{best_run.run_id}")
    subdirs = [os.path.join(run_dir, d) for d in os.listdir(run_dir) if os.path.isdir(os.path.join(run_dir, d))]
    if len(subdirs) != 1:
        raise ValueError(f"Expected exactly one folder in {run_dir}, found {len(subdirs)}.")
    model = AutoPeftModelForCausalLM.from_pretrained(subdirs[0])
    model.save_pretrained(model_dir)
    
    state_file = os.path.join(subdirs[0], "trainer_state.json")
    with open(state_file, "r") as f:
        state = json.load(f)
    return state
    
def run(args):
    start = time.time()

    # Get tokenizer and data
    tokenizer = get_tokenizer(args)

    if args.pwl:
        train_tok, dev_tok = get_train_dev_tok_pwl(args, tokenizer)
    else:
        train_tok, dev_tok = get_train_dev_tok_full_loss(args, tokenizer)
    test = get_test(args, tokenizer)
    
    if args.smoke_test:
        train_tok, dev_tok, test = train_tok.select(range(64)), dev_tok.select(range(64)), test.select(range(16))
        if args.pwl:
            print("\nCheck labels:")
            check_labels(tokenizer, train_tok[0])
            check_labels(tokenizer, train_tok[1])

    # get configs
    training_args, model_init, optuna_hp_space, bnb_config = get_configs(args, tokenizer)

    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=dev_tok,
        # after two rounds without improvement
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2,
                                         early_stopping_threshold=0.0)],
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics = preprocess_logits_for_metrics,
        processing_class=tokenizer
        )

    best_run = trainer.hyperparameter_search(
        hp_space=optuna_hp_space,
        backend="optuna",
        n_trials=args.trials,
        direction="maximize",
        compute_objective=lambda m: m["eval_accuracy"],
        )

    print("Best run:")
    print(best_run)

    # get model number, and find and save model to model_dir
    model_number = get_model_number(MODEL_DIR)
    model_dir = os.path.join(MODEL_DIR, f"model_{model_number}")
    meta = find_and_save_model(best_run, model_dir)
    
    
    # eval
    tokenizer = get_tokenizer(args, inference=True)
    metrics = evaluation_non_tok(test, model_dir, tokenizer, bnb_config)
    meta['test_metrics'] = metrics

    end = time.time()

    # compile and save meta
    meta['model_number'] = f"model_{model_number}"
    meta['time_mins'] = (end - start) / 60.0
    meta['model'] = args.model
    meta['date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    meta['data'] = args.lang
    meta['pwl'] = args.pwl
    meta['base_eval_acc'] = True

    print(meta)
    append_meta_file(meta, MODEL_DIR)
    with open(os.path.join(model_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # add loss plot
    collect_and_save_losses(meta['log_history'], model_dir)


def main(args):

    print("="*10, f"Running MODEL {args.model} PWL {args.pwl} - LANG {args.lang}","="*10)

    run(args)

if __name__ == "__main__":
    main(args)

