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

# --------------------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------------------

"""
- Resampling only for monolingual setting.
- training size has no effect in multilingual setting.

"""

EVAL_BATCH=16

PROMPT_LANGUAGE_MAP = {"en": "English",
                       "nl": "Dutch",
                       "no": "Norwegian",
                       "it": "Italian",
                       "pt": "Portuguese",
                       "ro": "Romanian",
                       "ru": "Russian",
                       "uk": "Ukrainian",
                       "bg": "Bulgarian",
                       "id": "Indonesian",
                       "vi": "Vietnamese",
                       "tr": "Turkish"
                        }

BASE_DIR = os.getenv("BASE_WCD")
DATA_DIR = os.path.join(BASE_DIR, "data/sets")
MODEL_DIR = os.path.join(BASE_DIR, "data/exp2")

def filter_long_context_examples(example: dict) -> bool:
    max_chars = 1000

    text = (
        f"Section: {example['section']}\n"
        f"Previous Sentence: {example['previous_sentence']}\n"
        f"Claim: {example['claim']}\n"
        f"Subsequent Sentence: {example['subsequent_sentence']}"
    )

    return len(text) <= max_chars


def get_monolingual_data_set(args: Namespace, 
                             data_path: str
                            ) -> List[Dataset]:
    """Data loader for the monolinugual setting."""
    data_dir = os.path.join(data_path, args.lang)
    ds = load_from_disk(data_dir)
    
    train = ds["train"]
    dev = ds["dev"]
    test = ds["test"]
    
    # Resample training
    if args.training_size < len(train):
        print("="*20)
        print(f"Len original training data {len(train)}")
        train = resample_data(train, args.training_size)
        print("="*20)
        print(f"Training data resampled to {len(train)}")
        print("="*20)

    # return data dict
    ds = {"train": train,
          "dev": dev,
          "test": test
          }
    return ds

def get_multilingual_data_sets(args: Namespace, 
                               data_path: str
                               ) -> List[Dataset]:
    """Data loader for the multilingual training setting. 
    Loads specified training langauuges in args.
    Keeps the dev and test of the target language.
    """
    train = []
    for training_lang in args.training_langs:
        data_dir = os.path.join(data_path, training_lang)
        ds_lang = load_from_disk(data_dir)

        # get train
        train_lang = ds_lang["train"]
        if args.training_size < len(train_lang):
             train_lang = resample_data(train_lang, args.training_size)
        train.extend(train_lang)
    
    # get target lang dev and test
    data_dir = os.path.join(data_path, args.test_lang)
    ds_target = load_from_disk(data_dir)    
    dev = ds_target["dev"]
    test = ds_target["test"]

    print("="*20)
    print("MULTILINGUAL TRAINING DATA")
    print("N:", len(train))
    print("="*20)

    # return datast dict
    dataset = {"train": Dataset.from_list(train),
               "dev": dev,
               "test": test
               }
    return dataset

def get_all_data_sets(args: Namespace, data_path: str) -> List[Dataset]:
    """
    Takes args and path. Loads data, renames lang to match the language name expected in promtps,
    and filter overly long context items.
    Return train, dev, test/
    """
    if args.experiment == "cl":
        ds = get_multilingual_data_sets(args=args, data_path=data_path)
    else: # binary, size, second_stage
        ds = get_monolingual_data_set(args=args, data_path=data_path)
    
    # change the lang name
    for split in ["train", "dev", "test"]:
        ds[split] = ds[split].map(
            lambda x: {**x, "lang": PROMPT_LANGUAGE_MAP[x["lang"]]}
        )

    if args.context:
        train_before = len(ds["train"])
        dev_before   = len(ds["dev"])
        test_before  = len(ds["test"])

        # filter out those long context items due to oom errors
        ds["train"] = ds["train"].filter(filter_long_context_examples)
        ds["dev"]   = ds["dev"].filter(filter_long_context_examples)
        ds["test"]  = ds["test"].filter(filter_long_context_examples)

        train_removed = train_before - len(ds["train"])
        dev_removed   = dev_before - len(ds["dev"])
        test_removed  = test_before - len(ds["test"])

        print("="*20)
        print("Filtered long-context examples:")
        print(f"Train removed: {train_removed}")
        print(f"Dev removed:   {dev_removed}")
        print(f"Test removed:  {test_removed}")
        print("="*20)

        if args.smoke_test:
            print("="*20)
            print("SMOKE _TEST")
            print("="*20)
            return ds["train"].select(range(96)), ds["dev"].select(range(32)), ds["test"].select(range(32))
    
    return ds["train"], ds["dev"], ds["test"]

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
    if model_type != "classifier":
        if not tokenizer.chat_template:
            raise Exception("tokenizer has not cha template.")
    
    tokenizer.model_max_length = 512
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
    system_msg = prompt['system'].format(**example)
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

    system = prompt_template['system'].format(**example)
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
    system = prompt_template['system'].format(**example)
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
            max_length=tokenizer.model_max_length,
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


def init_collate_classification_fn(tokenizer) -> Callable:
    def collate_fn(features):
        
        batch_inputs = {
            "input_ids": [f["input_ids"] for f in features],
            "attention_mask": [f["attention_mask"] for f in features],
        }

        
        batch = tokenizer.pad(
            batch_inputs,
            padding=True,
            return_tensors="pt",
        )

        labels = [int(f["labels"]) for f in features]
        batch["labels"] = torch.tensor(labels, dtype=torch.long)

        return batch

    return collate_fn
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
        data_dir = os.path.join(DATA_DIR, "main")
        prompt = prompts.VANILLA_PROMPTS
    if args.explanation == "basic":
        data_dir = os.path.join(DATA_DIR, "main")
        prompt = prompts.RATIONALE_LABEL_PROMPTS
    if args.explanation == "mix":
        data_dir = os.path.join(DATA_DIR, "main")
        prompt = prompts.VANILLA_PROMPTS # label prompt
        prompt_rationale = prompts.RATIONALE_PROMPTS # rationale prompt
        prompt_rationale['user'] = (prompt_rationale['user_context'] if args.context 
                        else prompt_rationale['user_claim']
                        )

    # Define the user message (ie context)
    prompt['user'] = (prompt['user_context'] if args.context 
                        else prompt['user_claim']
                    )

    # Get all data
    train, dev, test = get_all_data_sets(args=args, data_path=data_dir)

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

    # if args.smoke_test:
    #     train_tok = train_tok.select(range(96))
    #     dev_train_tok = dev_train_tok.select(range(96))
    #     dev_test_tok = dev_test_tok.select(range(32))
    #     test_tok = test_tok.select(range(32))

    return train_tok, dev_train_tok, dev_test_tok, test_tok

def get_data_lm(args: Namespace,
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

def tokenize_fn_classification(example: Dict, 
                              tokenizer: PreTrainedTokenizerBase
    ):
    """
    Basic tok function.
    Returns input_ids, am, and labels.
    """
    
    enc = tokenizer(
            example["text"],
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_attention_mask=True,
        )
    enc["labels"] = example["label"]
    return enc    

def get_data_classifier(args: Namespace,
             tokenizer_train: PreTrainedTokenizerBase,
            ) -> tuple:
    
    def build_context(example: dict) -> str:
        text = (f"Section: {example['section']}\n"
                f"Previous Sentence: {example['previous_sentence']}\n"
                f"Claim: {example['claim']}"
                f"Subsequent Sentence: {example['subsequent_sentence']}")
        return text
    
    if args.explanation == "none":
        data_dir = os.path.join(DATA_DIR, "main")
    if args.explanation == "basic":
        data_dir = os.path.join(DATA_DIR, "main")

    train, dev, test = get_all_data_sets(args=args, data_path=data_dir)
        
    # Tokenize function expects a text field
    if args.context:
        train = train.map(lambda x: {'text': build_context(x)})
        dev = dev.map(lambda x: {'text': build_context(x)})
        test = test.map(lambda x: {'text': build_context(x)})
    else:
        train = train.rename_column("claim", "text")
        dev = dev.rename_column("claim", "text")
        test = test.rename_column("claim", "text")       
    
    train_tok = train.map(tokenize_fn_classification, 
                                   fn_kwargs={"tokenizer": tokenizer_train}, 
                                   batched=False
                                   )
    dev_tok = dev.map(tokenize_fn_classification, 
                                   fn_kwargs={"tokenizer": tokenizer_train}, 
                                   batched=False
                                   )
    test_tok = test.map(tokenize_fn_classification, 
                                   fn_kwargs={"tokenizer": tokenizer_train}, 
                                   batched=False
                                   )
    
    # if args.smoke_test:
    #     train_tok = train_tok.select(range(96))
    #     dev_tok = dev_tok.select(range(96))
    #     test_tok = test_tok.select(range(32))

    train_dataloader = DataLoader(
        train_tok,
        shuffle=True,
        batch_size=args.batch_size,
        collate_fn=init_collate_classification_fn(tokenizer_train),
    )

    dev_dataloader = DataLoader(
        dev_tok,
        batch_size=EVAL_BATCH,
        collate_fn=init_collate_classification_fn(tokenizer_train),
    )

    test_dataloader = DataLoader(
        test_tok,
        batch_size=EVAL_BATCH,
        collate_fn=init_collate_classification_fn(tokenizer_train),
    )
    # return dev twice to match the return of lm
    return train_dataloader, dev_dataloader, dev_dataloader, test_dataloader


def get_data(args: Namespace,
             tokenizer_train,
             tokenizer_test,
            ) -> tuple:
    
    set_seed(args.seed)
    
    if args.model_type == "slm":
        return get_data_lm(args,
                           tokenizer_train,
                           tokenizer_test,
                           )
    if args.model_type == "classifier":
        return get_data_classifier(args,
                                   tokenizer_train,
                                   )