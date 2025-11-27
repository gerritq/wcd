import os
import torch
import random

from transformers import AutoTokenizer, PreTrainedTokenizerBase
from torch.utils.data import DataLoader

from datasets import Dataset
from utils import prompts
from typing import Callable, List, Dict, Callable
from datasets import load_from_disk, concatenate_datasets
from argparse import Namespace

from transformers import set_seed

set_seed(42)
# --------------------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------------------

EVAL_BATCH=16

BASE_DIR = os.getenv("BASE_WCD")
DATA_DIR = os.path.join(BASE_DIR, "data/sets")
MODEL_DIR = os.path.join(BASE_DIR, "data/exp2")

def get_all_data_sets(path: str) -> List[Dataset]:
    """
    Takes a path and language.
    Returns all three datasets.
    """
    ds = load_from_disk(path)
    return ds['train'], ds['dev'], ds['test']

def resample_data(ds: Dataset, total_size: int) -> Dataset:
    """
    Resample a dataset to total_size with 50/50 label balance.
    Returns a new Dataset; does not use or modify `self`.
    """
    assert total_size % 2 == 0, f"Total size ({total_size}) must be even."
    n_per_label = total_size // 2

    pos_all = [x for x in ds if x["label"] == 1]
    neg_all = [x for x in ds if x["label"] == 0]

    pos = pos_all[:min(len(pos_all), n_per_label)]
    neg = neg_all[:min(len(neg_all), n_per_label)]

    combined = pos + neg
    random.shuffle(combined)

    pos_count, neg_count = 0, 0
    for x in combined:
        if x["label"] == 1:
            pos_count += 1
        else:
            neg_count += 1
    assert pos_count == neg_count, "Dataset unbalanced after resampling."

    return Dataset.from_list(combined)

# --------------------------------------------------------------------------------------------------
# Tokenization functions
# --------------------------------------------------------------------------------------------------

def get_tokenizer(model_type: str, model_name: str, inference=False):
    """
    Loads and prepares the tokenizer.
    Return it for training or inference (bool keyword)
    """
    if inference:
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    # if tokenizer has no padding token, then reuse the end of sequence token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.model_max_length > 100_000:
        tokenizer.model_max_length = 512*2 
    if model_type != "classifier":
        if not tokenizer.chat_template:
            raise Exception("tokenizer has not cha template.")
    return tokenizer

def atl_loss_tokenize(example: dict,
                      prompt: dict,
                      tokenizer: PreTrainedTokenizerBase
                      ) -> dict:
    """
    ATL tokenization.
    Applies chat template and tokenizes
    Returns a tokenized dataset which is not padded!
    """
    # Build messages
    system_msg = prompt['system']
    user_msg = prompt['user'].format(**example)
    assistant_msg = prompt['assistant'].format(**example)

    messages_full = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": assistant_msg},
    ]
    messages_prompt = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    # Get raw strings from chat template
    full_text = tokenizer.apply_chat_template(
        messages_full,
        tokenize=False,
        add_generation_prompt=False,  # includes assistant content
    )
    prompt_text = tokenizer.apply_chat_template(
        messages_prompt,
        tokenize=False,
        add_generation_prompt=True,   # stops right before assistant content
    )

    # Tokenize both
    full_enc = tokenizer(
        full_text,
        truncation=True,
        max_length=tokenizer.model_max_length,
    )
    prompt_enc = tokenizer(
        prompt_text,
        truncation=True,
        max_length=tokenizer.model_max_length,
    )

    input_ids = full_enc["input_ids"]
    attention_mask = full_enc["attention_mask"]

    prompt_len = len(prompt_enc["input_ids"])

    # Build labels: only supervise assistant tokens
    labels = [-100] * len(input_ids)
    if prompt_len < len(input_ids):
        labels[prompt_len:] = input_ids[prompt_len:]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

def preprocess_function_training(example: dict, 
                                prompt_template: dict, 
                                tokenizer: PreTrainedTokenizerBase) -> dict:
    """
    Preprocess function for training.
    Takes an example, applies the chat template, and returns it.
    """

    system = prompt_template['system']
    user = prompt_template['user'].format(**example)
    assistant = prompt_template['assistant'].format(**example)
    
    messages = {
    "messages": [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant}
    ]
    }

    example["text"] = tokenizer.apply_chat_template(messages['messages'], tokenize=False)

    return example

def preprocess_function_generation(example: dict, 
                                   prompt_template: dict, 
                                   tokenizer: PreTrainedTokenizerBase) -> dict:
    """
    Preprocess function for evaluation -- set generation to true and no assitant message.
    """
    system = prompt_template['system']
    user = prompt_template['user'].format(**example)
    
    messages = {
    "messages": [
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ]
    }

    example["text"] = tokenizer.apply_chat_template(messages['messages'], 
                                                    tokenize=False, 
                                                    add_generation_prompt=True,
                                                    enable_thinking=False)
    return example

def ds_apply_chat_templates(ds: Dataset, 
                            tokenizer: PreTrainedTokenizerBase, 
                            prompt_template:str,
                            preprocess_function: Callable
                            ) -> Dataset:
    """
    Takes a ds.
    Applies the chat template for train/dev or test and returns the data.
    """
    remove_columns = [x for x in ds.column_names if x!="label"]

    ds_chat = ds.map(preprocess_function, 
                     remove_columns=remove_columns,
                     fn_kwargs={"tokenizer": tokenizer,
                                "prompt_template": prompt_template},)

    return ds_chat

def tokenize_fn(example: Dict, 
                tokenizer: PreTrainedTokenizerBase
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
            return_attention_mask=True,
        )
    enc["labels"] = enc["input_ids"].copy()
    return enc    
    
def atl_check_tokenize(example: dict, 
                       tokenizer: PreTrainedTokenizerBase):
    ids = example["input_ids"]
    labels = example["labels"]

    supervised_ids = [
        token_id
        for token_id, label_val in zip(ids, labels)
        if label_val != -100
    ]

    print("\n=== DECODED SUPERVISED (ASSISTANT) TOKENS ===")
    print(tokenizer.decode(supervised_ids, skip_special_tokens=False))

def init_collate_fn(tokenizer_train) -> Callable:
    def collate_fn(features):
        # 1) Pad input_ids and attention_mask with tokenizer
        batch = tokenizer_train.pad(
            {k: [f[k] for f in features] for k in ["input_ids", "attention_mask"]},
            padding=True,
            return_tensors="pt",
        )

        # 2) Collect labels as Python lists (handles both list and tensor)
        raw_labels = []
        for f in features:
            lbl = f["labels"]
            if isinstance(lbl, torch.Tensor):
                raw_labels.append(lbl.tolist())
            else:
                raw_labels.append(lbl)

        # 3) Pad labels to max length with -100
        max_len = max(len(l) for l in raw_labels)
        padded_labels = [
            l + [-100] * (max_len - len(l))
            for l in raw_labels
        ]

        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)

        return batch
    return collate_fn

def init_eval_collate_fn(tokenizer_test: PreTrainedTokenizerBase) -> Callable:
    def eval_collate_fn(features):
        batch = tokenizer_test.pad(
            {
                "input_ids":      [f["input_ids"] for f in features],
                "attention_mask": [f["attention_mask"] for f in features],
            },
            padding=True,
            return_tensors="pt",
        )

        # Optional: include labels for accuracy (only if test_dataset has them)
        if "label" in features[0]:
            batch["label"] = torch.tensor([f["label"] for f in features], dtype=torch.long)

        return batch
    return eval_collate_fn
# --------------------------------------------------------------------------------------------------
# Get data function wrapper
# --------------------------------------------------------------------------------------------------

def prepare_data(args: Namespace,
                 tokenizer_train,
                 tokenizer_test,
                ) -> tuple:
    
    # For experiment with prompts
    prompt_extension = "_" + args.prompt_extension if args.prompt_extension else ""
    
    if args.explanation == "none":
        data_dir = os.path.join(DATA_DIR, "main", args.lang)
        prompt = prompts.VANILLA_PROMPTS[args.lang+prompt_extension]
    if args.explanation == "basic":
        data_dir = os.path.join(DATA_DIR, "main", args.lang, "_", args.annotation_version)
        prompt = prompts.RATIONALE_LABEL_PROMPTS[args.lang+prompt_extension]
    if args.explanation == "mix":
        data_dir = os.path.join(DATA_DIR, "main", args.lang, "_", args.annotation_version)
        prompt = prompts.VANILLA_PROMPTS[args.lang+prompt_extension] # label prompt
        prompt_rationale = prompts.RATIONALE_PROMPTS[args.lang+prompt_extension] # rationale prompt
        prompt_rationale['user'] = (prompt_rationale['user_context'] if args.context 
                        else prompt_rationale['user_claim']
                        )

    # Define the user message (ie context)
    prompt['user'] = (prompt['user_context'] if args.context 
                        else prompt['user_claim']
                    )

    # Get all data
    train, dev, test = get_all_data_sets(data_dir)

    # Resample training
    if args.training_size < len(train):
        print("="*20)
        print(f"Len original training data {len(train)}")
        train = resample_data(train, args.training_size)
        print("="*20)
        print(f"Training data resampled to {len(train)}")
        print("="*20)

    # Get chat templates
    if not args.atl:
        train_chat = ds_apply_chat_templates(ds=train, 
                                            tokenizer=tokenizer_train,
                                            prompt_template=prompt,
                                            preprocess_function=preprocess_function_training
                                            )
        if args.explanation == "mix":
            # Double the train data
            train_chat_rationale = ds_apply_chat_templates(ds=train, 
                                            tokenizer=tokenizer_train,
                                            prompt_template=prompt_rationale,
                                            preprocess_function=preprocess_function_training
                                            )
            train_chat = concatenate_datasets([train_chat, train_chat_rationale]).shuffle(seed=42)
            print("="*20)
            print(f"Total size of mixed training data {len(train_chat)}")
            print("="*20)

        dev_train_chat = ds_apply_chat_templates(ds=dev, 
                                                tokenizer=tokenizer_train,
                                                prompt_template=prompt,
                                                preprocess_function=preprocess_function_training
                                                )

        print("="*20)
        print("EXAMPLE TRAINING CHAT TEMPLATES")
        print("Train instance 1:\n", train_chat[0]['text'], "\n\n")
        print("Train instance 2:\n", train_chat[1]['text'], "\n")
        print("="*20)

    dev_test_chat = ds_apply_chat_templates(ds=dev, 
                                            tokenizer=tokenizer_test,
                                            prompt_template=prompt,
                                            preprocess_function=preprocess_function_generation
                                            )

    test_chat = ds_apply_chat_templates(ds=test, 
                                        tokenizer=tokenizer_test,
                                        prompt_template=prompt,
                                        preprocess_function=preprocess_function_generation
                                        )

    print("="*20)
    print("EXAMPLE TEST CHAT TEMPLATES")
    print("Test instance 1:\n", test_chat[0]['text'], "\n\n")
    print("Test instance 2:\n", test_chat[1]['text'], "\n")
    print("="*20)

    # Tokenise data
    if args.atl:
        print("="*20)
        print(f"EXAMPLE ATL {args.explanation.upper()} CHAT TEMPLATES")
        print(preprocess_function_training(example=train[0],
                            prompt_template=prompt,
                            tokenizer=tokenizer_train)['text']
        )
        print("="*20)
        # Note: we perform atl on the raw data not the chat templates
        # We use the chat templ. to identfy the assistant tokens
        if args.explanation != "mix":
            train_tok = train.map(lambda x: atl_loss_tokenize(x, prompt=prompt,
                                                                tokenizer=tokenizer_train), 
                                    remove_columns=train.column_names
                                    )
        else:
            # combine train label and tok
            train_label_tok = train.map(lambda x: atl_loss_tokenize(x, prompt=prompt,
                                                    tokenizer=tokenizer_train), 
                        remove_columns=train.column_names
                        )
            train_rationale_tok = train.map(lambda x: atl_loss_tokenize(x, prompt=prompt_rationale,
                                                    tokenizer=tokenizer_train), 
                        remove_columns=train.column_names
                        )
            
            train_tok = concatenate_datasets([train_label_tok, train_rationale_tok])
            print("="*20)
            print(f"EXAMPLE ATL {args.explanation.upper()} TRAIN CHAT TEMPLATES")
            print(preprocess_function_training(example=train[0],
                                               prompt_template=prompt_rationale,
                                               tokenizer=tokenizer_train)['text']
            )
            print("="*20)
        
        # dev is for mix and others the same
        # why: because we only want label peformance to go up (not rationale gen)
        dev_train_tok = dev.map(lambda x: atl_loss_tokenize(x, prompt=prompt,
                                                               tokenizer=tokenizer_train), 
                                        remove_columns=train.column_names
                                        )
    else:
        train_tok = train_chat.map(tokenize_fn, 
                                   fn_kwargs={"tokenizer": tokenizer_train}, 
                                   batched=False
                                   )
        dev_train_tok = dev_train_chat.map(tokenize_fn, 
                                           fn_kwargs={"tokenizer": tokenizer_train},
                                           batched=False
                                           )
    
    # For eval, we do not need ATL
    dev_test_tok = dev_test_chat.map(tokenize_fn, 
                                    fn_kwargs={"tokenizer": tokenizer_test}, 
                                               batched=False
                                           )
    test_tok = test_chat.map(tokenize_fn, 
                                    fn_kwargs={"tokenizer": tokenizer_test}, 
                                               batched=False
                                           )

    if args.atl:
        print("="*20)
        print("EXAMPLE ASSISTANT TOKEN LOSS")
        print('TRAIN')
        atl_check_tokenize(example=train_tok[0], tokenizer=tokenizer_train)
        print('\nDEV')
        atl_check_tokenize(example=dev_train_tok[0], tokenizer=tokenizer_train)
        print("="*20)

    if args.smoke_test:
        train_tok = train_tok.select(range(96))
        dev_train_tok = dev_train_tok.select(range(96))
        dev_test_tok = dev_test_tok.select(range(32))
        test_tok = test_tok.select(range(32))

    return train_tok, dev_train_tok, dev_test_tok, test_tok

def get_data(args: Namespace,
             tokenizer_train,
             tokenizer_test,
            ) -> tuple:

    train, dev_train, dev_test, test = prepare_data(args=args,
                                                    tokenizer_train=tokenizer_train,
                                                    tokenizer_test=tokenizer_test,
                                                    )

    train_dataloader = DataLoader(
        train,
        shuffle=True,
        batch_size=args.batch_size,
        collate_fn=init_collate_fn(tokenizer_train)
    )

    dev_train_dataloader = DataLoader(
        dev_train,
        batch_size=EVAL_BATCH,
        collate_fn=init_collate_fn(tokenizer_train),
    )

    dev_test_dataloader = DataLoader(
        dev_test,
        batch_size=EVAL_BATCH,
        collate_fn=init_eval_collate_fn(tokenizer_test),
    )

    test_dataloader = DataLoader(
        test,
        batch_size=EVAL_BATCH,
        collate_fn=init_eval_collate_fn(tokenizer_test),
    )

    return train_dataloader, dev_train_dataloader, dev_test_dataloader, test_dataloader