from transformers import (AutoTokenizer, 
                          BitsAndBytesConfig,
                          PreTrainedTokenizerBase)
from torch.utils.data import Dataset

from prompts import SYSTEM_PROMPTS_SLM
from typing import Dict, List, Callable

MODEL_MAPPING =  {
    "mBert": "google-bert/bert-base-multilingual-uncased",
    "xlm-r-b": "FacebookAI/xlm-roberta-base",
    "xlm-r-l": "FacebookAI/xlm-roberta-large",
    "mDeberta-b": "microsoft/mdeberta-v3-base",
    "mDeberta-l": "microsoft/deberta-v3-large",
    "llama3_1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama3_3b": "meta-llama/Llama-3.2-3B-Instruct",
    "llama3_8b": "meta-llama/Llama-3.1-8B-Instruct",
    "llama3_70b": "meta-llama/Llama-3.3-70B-Instruct",
    "qwen3_06b": "Qwen/Qwen3-0.6B",
    "qwen3_4b": "Qwen/Qwen3-4B-Instruct-2507",
    "qwen3_8b": "Qwen/Qwen3-8B",
    "qwen3_30b": "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "qwen3_32b": "Qwen/Qwen3-32B",
    "gemma3_12b": "google/gemma-3-12b-it",
    "gpt_oss": "openai/gpt-oss-20b",
    "mistral_8b": "mistralai/Ministral-8B-Instruct-2410",
    "ds_llama": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "aya": "CohereLabs/aya-101",
    "gpt-4o-mini": "openai/gpt-4o-mini",
    "gemini-2.5-flash-lite": "google/gemini-2.5-flash-lite"
    }

# CONFIGS
bnb_config = BitsAndBytesConfig(
                            load_in_4bit=True, 
                            bnb_4bit_use_double_quant=True, 
                            bnb_4bit_quant_type="nf4", 
                            bnb_4bit_compute_dtype=torch.bfloat16
                            )

lora_config_lm = LoraConfig(
        lora_alpha=16,
        lora_dropout=.05,
        r=16,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
)


##########################################################################################
# Meta functions
##########################################################################################

MODEL_MAPPING =  {
    "mBert": "google-bert/bert-base-multilingual-uncased",
    "xlm-r-b": "FacebookAI/xlm-roberta-base",
    "xlm-r-l": "FacebookAI/xlm-roberta-large",
    "mDeberta-b": "microsoft/mdeberta-v3-base",
    "mDeberta-l": "microsoft/deberta-v3-large",
    "llama3_1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama3_3b": "meta-llama/Llama-3.2-3B-Instruct",
    "llama3_8b": "meta-llama/Llama-3.1-8B-Instruct",
    "llama3_70b": "meta-llama/Llama-3.3-70B-Instruct",
    "qwen3_06b": "Qwen/Qwen3-0.6B",
    "qwen3_4b": "Qwen/Qwen3-4B-Instruct-2507",
    "qwen3_8b": "Qwen/Qwen3-8B",
    "qwen3_30b": "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "qwen3_32b": "Qwen/Qwen3-32B",
    "gemma3_12b": "google/gemma-3-12b-it",
    "gpt_oss": "openai/gpt-oss-20b",
    "mistral_8b": "mistralai/Ministral-8B-Instruct-2410",
    "ds_llama": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "aya": "CohereLabs/aya-101",
    "gpt-4o-mini": "openai/gpt-4o-mini",
    "gemini-2.5-flash-lite": "google/gemini-2.5-flash-lite"
    }

def append_meta_file(meta: dict, model_dir: str):
    """appends to the meta file in the model dir; creates the file if it does not exist"""
    meta_path = os.path.join(model_dir, "meta_overview.jsonl")
    with open(meta_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")

def get_model_number(model_dir: str) -> int:
    """identifies the latest model number on the model doir"""
    model_names = [os.path.splitext(d)[0] for d in os.listdir(model_dir) if d.startswith("model_")]
    
    numbers = []
    for name in model_names:
        
        num = int(name.split("_")[1])
        numbers.append(num)
        
    next_number = max(numbers) + 1 if numbers else 1
    return next_number

def collect_and_save_losses(history, model_dir):
    """inputs the trainer history and the model dir
    collects train and eval losses and saves a plot in the model dir"""

    def plot_loss_curves(train_losses, eval_losses, out_dir):

        out_path = os.path.join(out_dir, "loss_plot.pdf")

        train_epochs = [x["epoch"] for x in train_losses]
        train_vals = [x["loss"] for x in train_losses]
        eval_epochs = [x["epoch"] for x in eval_losses]
        eval_vals = [x["eval_loss"] for x in eval_losses]

        plt.figure(figsize=(6, 4))
        plt.plot(train_epochs, train_vals, marker="o", label="Train Loss", color="steelblue")
        plt.plot(eval_epochs, eval_vals, marker="s", label="Eval Loss", color="tomato")

        plt.title("Training and Evaluation Loss over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        
    train_losses, eval_losses = [], []
    for log in history:
        if "loss" in log:
            train_losses.append({"epoch": log.get("epoch"), "loss": log["loss"]})
        if "eval_loss" in log:
            eval_losses.append({"epoch": log.get("epoch"), "eval_loss": log["eval_loss"]})

    if train_losses and eval_losses:
        plot_loss_curves(train_losses, eval_losses, model_dir)


##########################################################################################
# Tokenizer
##########################################################################################


def get_tokenizer(model_name: str, inference=False):
    if inference:
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    # if tokenizer has no padding token, then reuse the end of sequence token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.model_max_length > 100_000:
        tokenizer.model_max_length = 512*2 
    if not tokenizer.chat_template:
        raise Exception("Tokeniser has not cha template.")
    return tokenizer

##########################################################################################
# Data prep
##########################################################################################

def get_all_data_sets(path: str, lang: str) -> List[Dataset]:
    ds = load_from_disk(os.path.join(path, args.lang))
    return ds['train'], ds['dev'], ds['test']

def preprocess_function_training(example: Dict, 
                                prompt_template: str, 
                                tokenizer: PreTrainedTokenizerBase) -> Dict:
    """
    Preprocess function for training.
    Takes an example, applies the chat template, and returns it.
    """

    claim = example['claim']
    label = example['label']
    lang = example['lang'][:2] # in case we test more data eg en_8k

    system = prompt_template[lang]['system']
    user = prompt_template[lang]['user'].format(claim=claim)
    assistant = prompt_template[lang]['assistant'].format(label=label)
    
    messages = {
    "messages": [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant}
    ]
    }

    example["text"] = tokenizer.apply_chat_template(messages['messages'], tokenize=False)

    return example

def preprocess_function_generation(example: Dict, 
                                   prompt_template: Dict, 
                                   tokenizer: PreTrainedTokenizerBase) -> Dict:
    """
    Preprocess function for evaluation -- set generation to true and no assitant message.
    """

    claim = example['claim']
    lang = example['lang'][:2]

    system = prompt_template[lang]['system']
    user = prompt_template[lang]['user'].format(claim=claim)
    
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
                            preprocess_function: Callable) -> Dataset:
    """
    Takes a ds.
    Applies the chat template for train/dev or test and returns the data.
    """
    remove_columns = [x for x in ds.column_names]

    ds_chat = ds.map(preprocess_function, 
                     remove_columns=remove_columns,
                     fn_kwargs={"tokenizer": tokenizer,
                                "prompt_template": prompt_template},)

    return ds_chat

def get_max_sequence_length(ds: Dataset, tokenizer: PreTrainedTokenizerBase) -> int:
    """
    Get the max seq length of a ds. Bit inefficient.
    """
    
    def tokenize_and_get_length(example: dict):
        tokenized = tokenizer(
            example['text'], 
            truncation=False, 
            padding=False,
        )
        example['length'] = len(tokenized['input_ids'])
        return example

    dataset_with_lengths = dataset.map(tokenize_and_get_length, batched=False, )    
    max_length = max(dataset_with_lengths['length'])
    
    return max_length

def tokenise_ds(ds: Dataset, tokenizer: PreTrainedTokenizerBase) -> Dataset:
    """
    Takes a ds, applies basic tokenisation with tokenize_fn.
    Return tokenized data.
    """
    

    def tokenize_fn(example: Dict, 
                    tokenizer: reTrainedTokenizerBase, 
                    max_length: int):
        """
        Basic tok function.
        Returns input_ids, am, and labels.
        """
        enc = tokenizer(
            example["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_attention_mask=True,
        )
        enc["labels"] = enc["input_ids"].copy()
        return enc    
    
    max_length = get_max_sequence_length(ds, tokenizer)
    
    # Tokenize data
    ds_tok = ds.map(tokenize_fn,
                    fn_kwargs={
                               "tokenizer": tokenizer,
                               "max_length": train_max_seq_length,
                                },
                          batched=False, 
                          remove_columns=ds.column_names)
    
    return ds_tok