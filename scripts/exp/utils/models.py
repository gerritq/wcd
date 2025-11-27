
import os
import random
from typing import Optional, List

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from datasets import Dataset, concatenate_datasets
from peft import (get_peft_model, 
                  prepare_model_for_kbit_training,
                  LoraConfig, 
                  TaskType
                )
from transformers import (AutoModelForCausalLM, 
                          AutoModelForSequenceClassification, 
                          BitsAndBytesConfig)
from transformers.utils import is_flash_attn_2_available
from transformers import set_seed
set_seed(42)

# --------------------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------------------

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
    "llama3_8b_base": "meta-llama/Llama-3.1-8B",
    "qwen3_06b": "Qwen/Qwen3-0.6B",
    "qwen3_4b": "Qwen/Qwen3-4B-Instruct-2507",
    "qwen3_8b": "Qwen/Qwen3-8B",
    "qwen3_8b_base": "Qwen/Qwen3-8B-Base",
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

BNB_CONFIG = BitsAndBytesConfig(
                        load_in_4bit=True, 
                        bnb_4bit_use_double_quant=True, 
                        bnb_4bit_quant_type="nf4", 
                        bnb_4bit_compute_dtype=torch.bfloat16
                        )

LORA_CONFIG_LM = LoraConfig(
        lora_alpha=32,
        lora_dropout=.05,
        r=16,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type=TaskType.CAUSAL_LM,
)

LORA_CONFIG_CLS = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

BASE_DIR = os.getenv("BASE_WCD")
DATA_DIR = os.path.join(BASE_DIR, "data/sets")
MODEL_DIR = os.path.join(BASE_DIR, "data/exp2")

"""
- we modify the prep data only for atl atm
"""

random.seed(42)

# --------------------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------------------

class LM(ABC):
    """Abstract class for LMs."""
    def __init__(self, 
                 model_type: str,
                 model_name: str, 
                 quantization: bool,
    ):
        
        self.model_type = model_type
        self.model_name = model_name
        self.quantization = quantization
        self.attn_impl = "flash_attention_2" if is_flash_attn_2_available() else "sdpa"

        # Model
        self.model = None
        self.bnb_config = BNB_CONFIG if self.quantization else None
        self.lora_config = LORA_CONFIG_CLS if self.model_type == "classifier" else LORA_CONFIG_LM
        

    def build(self):
        """Standard for LM init"""
        
        # Load model and peft
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=self.bnb_config,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation=self.attn_impl,
        )
        
        if self.quantization:
            model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, self.lora_config)

        # This does the trick to enable check pointing: https://discuss.huggingface.co/t/i-used-to-have-no-problem-with-peft-fine-tuning-after-hundreds-of-trainings-but-now-i-have-encountered-the-error-runtimeerror-element-0-of-tensors-does-not-require-grad-and-does-not-have-a-grad-fn/168829/3
        # Set checkpointing to false in trainer args        
        model.gradient_checkpointing_enable()  # PEFT + checkpointing
        model.enable_input_require_grads()  
        model.config.use_cache = False         # disable cache
        model.print_trainable_parameters()

        self.model = model

        # Print overview
        self._print_setting()

        return self

    def _print_setting(self):
        print("="*20)
        print("BUILT MODEL")
        print("Model type:", self.model_type)
        print("Model name:", self.model_name)
        print("Quantization:", self.quantization)
        print("="*20)

class SLM(LM):
    """
    Vanilla class for SLM.
    """
    def __init__(self, 
                 model_type: str,
                 model_name: str, 
                 quantization: bool,
    ):
        super().__init__(model_type=model_type,
                        model_name=model_name, 
                        quantization=quantization, 
                        
        )

class SLMClassifier(LM):
    """
    Class that uses a Classification Head instead of a LM head.
    """
    def __init__(self, 
                 model_type: str,
                 model_name: str, 
                 quantization: bool,
    ):
        super().__init__(model_type=model_type,
                        model_name=model_name, 
                        quantization=quantization, 
                        
        )
        
    
    def build(self):
        """
        Overwrite the build function to accomodate classification head.
        """
        
        # Load model and replace classification head
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            quantization_config=self.bnb_config,
            num_labels=2,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation=self.attn_impl
        )

        # Replace classification head
        hidden_size = model.config.hidden_size
        num_labels = model.config.num_labels
        model.score = CustomClassificationHead(hidden_size, num_labels)
        
        # Quant
        if self.quantization:
            model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, self.lora_config)

        # This does the trick to enable check pointing: https://discuss.huggingface.co/t/i-used-to-have-no-problem-with-peft-fine-tuning-after-hundreds-of-trainings-but-now-i-have-encountered-the-error-runtimeerror-element-0-of-tensors-does-not-require-grad-and-does-not-have-a-grad-fn/168829/3
        # Set checkpointing to false in trainer args        
        model.gradient_checkpointing_enable()  # PEFT + checkpointing
        model.enable_input_require_grads()  
        model.config.use_cache = False         # disable cache
        model.print_trainable_parameters()

        # Appears to only be corrected for the classifier case
        if not model.config.pad_token_id:
            model.config.pad_token_id = self.tokenizer.pad_token_id

        # Ensure that new head is also on cuda
        model.to(device="cuda", dtype=torch.bfloat16)

        self.model = model

        self._print_setting()

        return self

    def _prepare_data(self):
        # get all data
        train, dev, test = get_all_data_sets(self.data_dir, self.lang)

        # Tokenize function expects a text field
        train = train.rename_column("claim", "text")
        dev = dev.rename_column("claim", "text")
        test = test.rename_column("claim", "text")        

        # Tokenize
        train_tok = tokenize_ds(train, self.tokenizer)
        dev_tok = tokenize_ds(dev, self.tokenizer)
        test_tok = tokenize_ds(test, self.tokenizer)

        # Adjustment: make lables the cd_labels (lables=input_ids in LM)
        train_tok = train_tok.remove_columns("labels").rename_column("cd_labels", "labels")
        dev_tok = dev_tok.remove_columns("labels").rename_column("cd_labels", "labels")
        test_tok = test_tok.remove_columns("labels").rename_column("cd_labels", "labels")

        # Assign to attributes (dev_train_tok just to avoid if conditions in trainer)
        self.train_tok = train_tok
        self.dev_tok = dev_tok
        self.dev_train_tok = self.dev_tok
        self.test_tok = test_tok

        # if self.smoke_test:
        #     self.train_tok = train_tok.select(range(96))
        #     self.dev_tok = dev_tok.select(range(96))
        #     self.dev_train_tok = self.dev_tok
        #     self.test_tok = self.test_tok.select(range(32))


class CustomClassificationHead(nn.Module):
    """
    Custom Classification Head for LMs.

    The default HF LM classification head has no intermediate layer.

    We implement one intermediate layer for models largeer the 1024 hidden dim size.
    """
    def __init__(self, hidden_size, num_labels):
        super().__init__()

        if hidden_size > 1024:
            intermediate1 = hidden_size // 2
            intermediate2 = intermediate1 // 2
            self.net = nn.Sequential(
                nn.Linear(hidden_size, intermediate1),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(intermediate1, intermediate2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(intermediate2, num_labels, bias=False)
            )
        else:
            intermediate_size = hidden_size // 2
            self.net = nn.Sequential(
                nn.Linear(hidden_size, intermediate_size),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(intermediate_size, num_labels, bias=False)
            )

    def forward(self, x):
        return self.net(x)

SLM_REGISTRY = {
    "slm": SLM,
    "classifier": SLMClassifier,
}

def build_slm(model_type: str,
              model_name: str, 
              quantization: bool,):
    cls = SLM_REGISTRY.get(model_type)
    return cls(model_type=model_type,
              model_name=model_name, 
              quantization=quantization, 
              ).build()