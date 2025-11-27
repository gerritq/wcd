import os
import json
import argparse
import time
from datetime import datetime
from tqdm import tqdm
import numpy as np
import torch
import shutil
from pwl import get_data, check_labels
from utils import (
                    MODEL_MAPPING, 
                    append_meta_file, 
                    get_model_number,
                    collect_and_save_losses,
                    evaluation_non_tok,
                    get_tokenizer,
                    get_train_dev,
                    get_test,
                    get_max_sequence_length
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

"""
can pass custom tokeniser to collator@!!!!!


"""
set_seed(42)

BASE_DIR = os.getenv("BASE_WCD")
DATA_DIR = os.path.join(BASE_DIR, "data/sets/main")
MODEL_DIR = os.path.join(BASE_DIR, "data/models/pwl/hp")
TEMP_DIR = os.path.join(BASE_DIR, "scripts/classification/.hp_random")
os.makedirs(TEMP_DIR, exist_ok=True)
TEMP_DIR = tempfile.mkdtemp(dir=TEMP_DIR)

def tokenise_train_dev(data, 
                  tokenizer,
                  max_seq_length=4096):
    """custom tokenise to implement pwl"""

    def get_assistant_tag(tokenizer):
        """function to get the assistant tag
        
        return the assistant tag and ids
        """
        messages = ([{"role": "user", "content":"test"}])
        no_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        with_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        if with_prompt.startswith(no_prompt):
            assistant_tag = with_prompt[len(no_prompt):]
        else:
            raise ValueError("Cannot identify the generation token.")
        
        assistant_tag_ids = tokenizer(assistant_tag, add_special_tokens=False)["input_ids"]

        return assistant_tag, assistant_tag_ids

    def preprocess_function(example, tokenizer):
        """preprocess to obtain the prompt"""
        claim = example['claim']
        label = example['label']
        lang = example['lang'][:2] # in case we test more data eg en_8k

        system = SYSTEM_PROMPTS_SLM[lang]['system']
        user = SYSTEM_PROMPTS_SLM[lang]['user'].format(claim=claim)
        assistant = SYSTEM_PROMPTS_SLM[lang]['assistant'].format(label=label)
        
        messages = {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant}
        ]
        }

        example['text'] = tokenizer.apply_chat_template(messages['messages'],
                                                        tokenize=False,
                                                        add_generation_prompt=False,
                                                        enable_thinking=False
                                                        )
        return example
        
    def custom_tokenizer(example, tokenizer, assistant_tag_ids, max_seq_length):
        def find_sublist_reverse(sub, lst):
            for i in range(len(lst) - len(sub), -1, -1):  # start from the back
                if lst[i:i+len(sub)] == sub:
                    return i
            return -1

        text_tok = tokenizer(example['text'], truncation=True, max_length=max_seq_length, padding='max_length')
        text_input_ids = text_tok['input_ids']
        am  = text_tok["attention_mask"]

        # assistant tag tokens
        assistant_tag_index = find_sublist_reverse(assistant_tag_ids, text_input_ids)
        assert assistant_tag_index != -1, f"Could not find the assistant tag."
        # assert (assistant_tag_index / len(text_input_ids)) > .8, f"Assistant tag not in the final 20% of the text."

        # generate completion mask
        # no need to shift lables, done by the model internally:
        # https://discuss.huggingface.co/t/where-does-the-transformers-do-the-target-text-shifting-in-causal-lm/32408
        labels = [-100]*len(text_input_ids)
        start = assistant_tag_index + len(assistant_tag_ids)
        labels[start:] = text_input_ids[start:]
        for i, m in enumerate(am):
            if m == 0:
                labels[i] = -100
        assert len(labels) == len(text_input_ids), "Labels length is incorrect."

        text_tok['labels'] =  labels
        return text_tok


    # find assistant tag and ids
    assistant_tag, assistant_tag_ids = get_assistant_tag(tokenizer)

    # get messages
    ds_messages = data.map(preprocess_function, fn_kwargs={"tokenizer": tokenizer},)
    print(ds_messages['text'][0])
    max_seq_length = get_max_sequence_length(ds_messages, tokenizer) + 54
    print("max seq length", max_seq_length)

    ds_tok = ds_messages.map(custom_tokenizer,
                              fn_kwargs={"tokenizer": tokenizer,
                              "assistant_tag_ids": assistant_tag_ids,
                              "max_seq_length": max_seq_length}
                            )

    return ds_tok



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
            "learning_rate": trial.suggest_loguniform("learning_rate", 5e-6, 5e-4),
            "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [4, 8]),
            "max_grad_norm": trial.suggest_uniform("max_grad_norm", 0.1, 2.0),
            "weight_decay": trial.suggest_loguniform("weight_decay", 1e-6, 1e-1),
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
        per_device_train_batch_size=4, #  
        # num_train_epochs=5,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss", 
        greater_is_better=False,
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
    train, dev = get_data(args)
    test = get_test(args, tokenizer)
    if args.smoke_test:
        train, dev, test = train.select(range(64)), dev.select(range(64)), test.select(range(16))
    train_tok, dev_tok = tokenise_train_dev(train, tokenizer), tokenise_train_dev(dev, tokenizer)

    if args.smoke_test:
        check_labels(tokenizer, train_tok[0])
        check_labels(tokenizer, train_tok[1])

    # get configs
    training_args, model_init, optuna_hp_space, bnb_config = get_configs(args, tokenizer)

    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=dev_tok
        )

    best_run = trainer.hyperparameter_search(
        hp_space=optuna_hp_space,
        backend="optuna",
        n_trials=args.trials,
        direction="minimize",
        compute_objective=lambda m: m["eval_loss"],
        )

    print(best_run)
    model_number = get_model_number(MODEL_DIR)
    model_dir = os.path.join(MODEL_DIR, f"model_{model_number}")
    meta = find_and_save_model(best_run, model_dir)
    meta['model_number'] = f"model_{model_number}"
    # get best checkpoint

    tokenizer = get_tokenizer(args, inference=True)
    test = get_test(args, tokenizer)
    if args.smoke_test:
        test = test.select(range(64))
    metrics = evaluation_non_tok(test, model_dir, tokenizer, bnb_config)
    meta['test_metrics'] = metrics

    print(meta)

    append_meta_file(meta, MODEL_DIR)
    with open(os.path.join(model_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--trials", type=int, required=True)
    parser.add_argument("--smoke_test", type=int, required=True)
    args = parser.parse_args()
    args.model = MODEL_MAPPING[args.model]
    args.smoke_test = bool(args.smoke_test)

    print("="*10, f"Running MODEL {args.model} - LANG {args.lang}","="*10)

    best_run = run(args)

    print(f"\tHP search done in {(time.time() - start) / 60:.2f} mins")
    print("\nBest run:", best_run)

if __name__ == "__main__":
    main()

