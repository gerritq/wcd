
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
                  TaskType,
                  PeftModel
)
from transformers import (AutoModelForCausalLM, 
                          BitsAndBytesConfig,
                          AutoModel,
                          AutoTokenizer,
                          AutoModelForSequenceClassification
)
from transformers.utils import is_flash_attn_2_available

# This is specifically for the classifier
from transformers.modeling_layers import GenericForSequenceClassification
from transformers.models.qwen3.modeling_qwen3 import Qwen3PreTrainedModel
from transformers.models.llama.modeling_llama import LlamaPreTrainedModel

from transformers import PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

# --------------------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------------------
from transformers import set_seed
set_seed(42)

use_bf16 = (
    torch.cuda.is_available()
    and torch.cuda.is_bf16_supported()
)
print("="*20)
print(f'Using bf16 to load the model: {use_bf16}')
print("="*20)

MODEL_MAPPING =  {
    "llama3_1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama3_3b": "meta-llama/Llama-3.2-3B-Instruct",
    "llama3_3b_base": "meta-llama/Llama-3.2-3B",
    "llama3_8b": "meta-llama/Llama-3.1-8B-Instruct",
    "llama3_8b_base": "meta-llama/Llama-3.1-8B",
    "qwen3_06b": "Qwen/Qwen3-0.6B",
    "qwen3_06b_base": "Qwen/Qwen3-0.6B-Base",
    "qwen3_4b": "Qwen/Qwen3-4B-Instruct-2507",
    "qwen3_4b_base": "Qwen/Qwen3-4B",
    "qwen3_8b": "Qwen/Qwen3-8B",
    "qwen3_8b_base": "Qwen/Qwen3-8B-Base",
    "aya_8b": "CohereLabs/aya-expanse-8b",
    "aya_8b_base": "CohereLabs/aya-expanse-8b",
    "mBert": "google-bert/bert-base-multilingual-uncased",
    "xlm-r-b": "FacebookAI/xlm-roberta-base",
    "xlm-r-l": "FacebookAI/xlm-roberta-large",
    "mDeberta-b": "microsoft/mdeberta-v3-base",
    "mDeberta-l": "microsoft/deberta-v3-large",
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

set_seed(42)

# --------------------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------------------

class LM(ABC):
    """Abstract class for LMs."""
    def __init__(self, 
                 model_type: str,
                 model_name: str, 
                 quantization: bool = True,
                 model_dir: str = "",
                 from_checkpoint: bool = False,
    ):
        
        self.model_type = model_type
        self.model_name = model_name
        self.quantization = quantization
        self.model_dir = model_dir
        self.from_checkpoint = from_checkpoint
        self.attn_impl = "flash_attention_2" if is_flash_attn_2_available() else "sdpa"

        # Model
        self.model = None
        self.bnb_config = BNB_CONFIG if self.quantization else None
        self.lora_config = LORA_CONFIG_CLS if self.model_type == "clf" else LORA_CONFIG_LM
        

    def build(self):
        """Standard for LM init"""
        
        # Load model and peft
        
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=self.bnb_config,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation=self.attn_impl,
            dtype=torch.bfloat16 if use_bf16 else "auto",
        )

        if self.quantization:
            base_model = prepare_model_for_kbit_training(base_model)


        if self.from_checkpoint:
            print("="*20)
            print(f"Loading model from checkpoint {self.model_dir}")
            print("="*20)
        
            # FT model config
            model = PeftModel.from_pretrained(model=base_model, 
                                              model_id=self.model_dir,
                                              is_trainable=True
            )
        else:
            model = get_peft_model(base_model, self.lora_config)

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
                 quantization: bool = True,
                 model_dir: str = "",
                 from_checkpoint: bool = False,
    ):
        super().__init__(model_type=model_type,
                        model_name=model_name, 
                        model_dir=model_dir,
                        from_checkpoint=from_checkpoint, 
                        quantization=quantization, 
                        
        )

class PLM(LM):
    """
    Encoder class.
    """
    def __init__(self, 
                 model_type: str,
                 model_name: str, 
                 quantization: bool = True,
                 model_dir: str = "",
                 from_checkpoint: bool = False,
    ):
        super().__init__(
            model_type=model_type,
            model_name=model_name, 
            quantization=quantization,
            model_dir=model_dir,
            from_checkpoint=from_checkpoint,
        )

    def build(self):

        if self.from_checkpoint:
            model_path = self.model_dir
            print("="*20)
            print(f"Loading model from checkpoint {self.model_dir}")
            print("="*20)
        else:
            model_path = self.model_name
            
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=2,
            device_map="auto",
            trust_remote_code=True,
            dtype=torch.bfloat16 if use_bf16 else "auto",
        )
        self.model = model
        self._print_setting()
        return self
    
class SLMClassifier(LM):
    def __init__(self, 
                 model_type: str,
                 model_name: str, 
                 quantization: bool = True,
                 model_dir: str = "",
                 from_checkpoint: bool = False,
    ):
        super().__init__(model_type=model_type,
                        model_name=model_name, 
                        model_dir=model_dir,
                        from_checkpoint=from_checkpoint, 
                        quantization=quantization, 
                        
        )

        self.FamilyPretrainedModel = Qwen3PreTrainedModel if "qwen" in model_name else LlamaPreTrainedModel

    def build(self):
        class CustomGenericForSequenceClassification(GenericForSequenceClassification, self.FamilyPretrainedModel):
            """This is a custom `GenericForSequenceClassification` class whicih replaces the default
            classification head
            """
            def __init__(self, config, base_model: nn.Module):
                super().__init__(config)
                self.num_labels = config.num_labels
                # Similar to `self.model = AutoModel.from_config(config)` but allows to change the base model name if needed in the child class
                
                # We pass the quantised base_model
                # setattr(self, self.base_model_prefix, AutoModel.from_config(config))
                setattr(self, self.base_model_prefix, base_model)
                
                # We replace the self.score default implementation with our classification head
                # self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)
                self.score = CustomClassificationHead(config.hidden_size, self.num_labels)
                
                # Initialize weights and apply final processing
                self.post_init()
                
        base_model = AutoModel.from_pretrained(
                        self.model_name,
                        quantization_config=self.bnb_config,
                        num_labels=2,
                        device_map="auto",
                        trust_remote_code=True,
                        attn_implementation=self.attn_impl,
                        dtype=torch.bfloat16 if use_bf16 else "auto",
        )

        config = base_model.config
        if config.pad_token_id is None:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            config.pad_token_id = tokenizer.pad_token_id

        # Before lora adapters (done incorreclty beofore!)
        if self.quantization:
            base_model = prepare_model_for_kbit_training(base_model)

        # Base clf model
        base_clf_model = CustomGenericForSequenceClassification(
                                                            config=config,
                                                            base_model=base_model,
        )

        if self.from_checkpoint:
            print("="*20)
            print(f"Loading model from checkpoint {self.model_dir}")
            print("="*20)
            # model from checkpoint: lora adapters and clf head!
            model = PeftModel.from_pretrained(
                model=base_clf_model,
                model_id=self.model_dir,
                is_trainable=True,
            )

            # param_names = [name for name, _ in model.named_parameters()]

            # print(param_names)
        else:
            model = get_peft_model(base_clf_model, self.lora_config)


        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        model.config.use_cache = False
        model.print_trainable_parameters()

        self.model = model
        self._print_setting()
        return self

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
    "clf": SLMClassifier,
    "plm": PLM,
}

def build_slm(model_type: str,
                 model_name: str, 
                 quantization: bool = True,
                 model_dir: str = "",
                 from_checkpoint: bool = False,):
    clf_ = SLM_REGISTRY.get(model_type)
    return clf_(model_type=model_type,
                 model_name=model_name, 
                 quantization=quantization,
                 model_dir=model_dir,
                 from_checkpoint=from_checkpoint,
              ).build()