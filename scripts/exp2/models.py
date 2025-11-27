
import os
import random
from typing import Optional, List

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from datasets import Dataset, concatenate_datasets
from peft import get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification
from transformers.utils import is_flash_attn_2_available

from prompts import VANILLA_PROMPTS, RATIONALE_PROMPTS, RATIONALE_LABEL_PROMPTS
from utils import (MODEL_MAPPING,
                   BNB_CONFIG,
                   LORA_CONFIG_LM,
                   LORA_CONFIG_CLS,
                   get_tokenizer,
                   preprocess_function_training,
                   preprocess_function_generation,
                   get_all_data_sets, 
                   ds_apply_chat_templates, 
                   tokenize_ds,
                   tokenize_ds_ATL,
                   check_assitant_token_lables)

BASE_DIR = os.getenv("BASE_WCD")
DATA_DIR = os.path.join(BASE_DIR, "data/sets")
MODEL_DIR = os.path.join(BASE_DIR, "data/exp2")

"""
- we modify the prep data only for atl atm
"""

random.seed(42)

class LM(ABC):
    """Abstract class for LMs."""
    def __init__(self, 
                 model_type: str,
                 model_name: str, 
                 lang: str,
                 quantization: bool,
                 training_size: int,
                 context: int,
                 smoke_test: bool,
                 explanation: str):
        
        self.model_type = model_type
        if model_type not in ['vanilla', 'atl', 'classifier']:
            raise ValueError("Unknown model type: {model_type}")
        self.model_name = MODEL_MAPPING[model_name]
        self.lang = lang
        self.smoke_test = smoke_test
        self.quantization = quantization
        self.context = context
        self.training_size = training_size
        self.explanation = explanation
        self.attn_impl = "flash_attention_2" if is_flash_attn_2_available() else "sdpa"

        # Dirs
        # Order of if statements important for model_dir
        self.model_dir = (os.path.join(MODEL_DIR, self.lang) if not smoke_test else
                          os.path.join(MODEL_DIR, "testing"))
        os.makedirs(self.model_dir, exist_ok=True)

        # Tokenizer and model
        self.tokenizer_train = None
        self.tokenizer_eval = None
        self.model = None
        self.bnb_config = BNB_CONFIG if self.quantization else None
        self.lora_config = LORA_CONFIG_CLS if self.model_type == "classifier" else LORA_CONFIG_LM
        
        # Data
        self.train_tok = None
        self.dev_train_tok = None
        self.dev_test_tok = None
        self.test_tok = None

    @abstractmethod
    def _prepare_data(self):
        pass

    @staticmethod
    def resample_train_data(train_ds: Dataset, total_size: int) -> Dataset:
        """
        Resample a dataset to total_size with 50/50 label balance.
        Returns a new Dataset; does not use or modify `self`.
        """
        assert total_size % 2 == 0, "Total size must be even."
        n_per_label = total_size // 2

        pos_all = [x for x in train_ds if x["label"] == 1]
        neg_all = [x for x in train_ds if x["label"] == 0]

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
    

    def build(self):
        """Standard for LM init"""
        # Load tokenizers
        self.tokenizer_train = get_tokenizer(self.model_type, self.model_name)
        self.tokenizer_eval = get_tokenizer(self.model_type, self.model_name, inference=True) # changes to left padding
        
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

        # preparate data
        self._prepare_data()

        # Print overview
        self._print_setting()

        return self

    def _print_setting(self):
        print("="*20)
        print("BUILT MODEL")
        print("Model type:", self.model_type)
        print("Model name:", self.model_name)
        print("Lang:", self.lang)
        print("Quantization:", self.quantization)
        print("Training data N:", len(self.train_tok))
        print("Smoke test:", self.smoke_test)
        print("="*20)

class SLMLoRA(LM):
    """
    Vanilla class for SLM.
    """
    def __init__(self, 
                 model_type: str,
                 model_name: str,
                 lang: str, 
                 smoke_test: bool, 
                 quantization: bool,
                 context: bool,
                 training_size: int):
        super().__init__(model_type=model_type,
                        model_name=model_name, 
                        lang=lang, 
                        smoke_test=smoke_test, 
                        quantization=quantization, 
                        training_size=training_size,
                        context=context)
    
    def _prepare_data(self):
        # get all data
        train, dev, test = get_all_data_sets(self.data_dir, self.lang)

        # apply chat templates
        train_chat = ds_apply_chat_templates(train, 
                                        self.tokenizer_train,
                                        self.prompt_template,
                                        preprocess_function_training)

        dev_train_chat = ds_apply_chat_templates(dev, 
                                      self.tokenizer_train,
                                      self.prompt_template,
                                      preprocess_function_training)

        dev_test_chat = ds_apply_chat_templates(dev, 
                                      self.tokenizer_eval,
                                      self.prompt_template,
                                      preprocess_function_generation)

        test_chat = ds_apply_chat_templates(test, 
                                      self.tokenizer_eval,
                                      self.prompt_template,
                                      preprocess_function_generation)

        print("="*20)
        print("EXAMPLE TRAINING CHAT TEMPLATES")
        print("Train instance 1:\n", train_chat[0]['text'], "\n")
        print("Train instance 2:\n", train_chat[1]['text'], "\n")
        print("="*20)
        print("EXAMPLE TEST CHAT TEMPLATES")
        print("Test instance 1:\n", test_chat[0]['text'], "\n")
        print("Test instance 2:\n", test_chat[1]['text'], "\n")
        print("="*20)

        # tok data
        train_tok = tokenize_ds(train_chat, self.tokenizer_train)
        
        dev_train_tok = tokenize_ds(dev_train_chat, self.tokenizer_train)
        dev_test_tok = tokenize_ds(dev_test_chat, self.tokenizer_eval)

        test_tok = tokenize_ds(test_chat, self.tokenizer_eval)
        
        # assign to attributes
        self.train_tok = train_tok
        self.dev_train_tok = dev_train_tok
        self.dev_test_tok = dev_test_tok
        self.test_tok = test_tok

        if self.smoke_test:
            self.train_tok = train_tok.select(range(96))
            self.dev_train_tok = dev_train_tok.select(range(96))
            self.dev_test_tok = dev_test_tok.select(range(32))
            self.test_tok = self.test_tok.select(range(32))

class SLMATL(LM):
    """
    Class that uses assistant token loss (ATL).
    """
    def __init__(self, 
                 model_type: str, 
                 model_name: str, 
                 lang: str, 
                 smoke_test: bool, 
                 quantization: bool,
                 context: bool,
                 explanation: str,
                 training_size: int):
        super().__init__(model_type=model_type,
                         model_name=model_name, 
                         lang=lang, 
                         smoke_test=smoke_test, 
                         quantization=quantization, 
                         context=context,
                         explanation=explanation,
                         training_size=training_size)
        
    
    def _prepare_data(self):

        # for comparison we need the exaxt same instances ..... 
        
        if self.explanation == "none":
            
            # Just to make sure we use the same data!! this should normally be main
            
            prompt_template = VANILLA_PROMPTS[self.lang]
            prompt_template['user'] = (prompt_template['user_context'] if self.context 
                                      else prompt_template['user_claim'])
            
            # train and test templates are the same
            self._prepare_data_basic(data_dir, prompt_template, prompt_template)
        
    
        # We have changed the inference prompt to vanilla, we can test this but with
        # the english rationales we shoul also produce them
        if self.explanation == "basic":
            data_dir = os.path.join(DATA_DIR, "annotation")
            prompt_template_train = RATIONALE_LABEL_PROMPTS[self.lang]
            prompt_template_train['user'] = (prompt_template_train['user_context'] if self.context 
                                      else prompt_template_train['user_claim'])

            prompt_template_test = RATIONALE_LABEL_PROMPTS[self.lang]
            prompt_template_test['user'] = (prompt_template_test['user_context'] if self.context 
                                      else prompt_template_test['user_claim'])
            
            
            # train and test differ
            self._prepare_data_basic(data_dir, prompt_template_train, prompt_template_test)
    
        if self.explanation == "mix":
            self._prepare_data_mix()

    def _prepare_data_mix(self):
        data_dir = os.path.join(DATA_DIR, "annotation")

        prompt_template_vanilla = VANILLA_PROMPTS[self.lang]
        prompt_template_vanilla['user'] = (prompt_template_vanilla['user_context'] if self.context 
                                    else prompt_template_vanilla['user_claim'])

        prompt_template_rationale = RATIONALE_PROMPTS[self.lang]
        prompt_template_rationale['user'] = (prompt_template_rationale['user_context'] if self.context 
                                    else prompt_template_rationale['user_claim'])

        train, dev, test = get_all_data_sets(data_dir, self.lang)

        if self.training_size != -1:
            train = self.resample_train_data(train, self.training_size)
            print("="*20)
            assert self.training_size == len(train), f"New training data size does not match target"
            print(f"Training data resampled to {len(train)}")
            print("="*20)
            
        # apply chat templates
        train_chat_label = ds_apply_chat_templates(train, 
                                        self.tokenizer_train,
                                        prompt_template_vanilla,
                                        preprocess_function_training)

        train_chat_rationale = ds_apply_chat_templates(train, 
                                        self.tokenizer_train,
                                        prompt_template_rationale,
                                        preprocess_function_training)

        dev_train_chat = ds_apply_chat_templates(dev, 
                                      self.tokenizer_train,
                                      prompt_template_vanilla,
                                      preprocess_function_training)

        dev_test_chat = ds_apply_chat_templates(dev, 
                                      self.tokenizer_eval,
                                      prompt_template_vanilla,
                                      preprocess_function_generation)

        test_chat = ds_apply_chat_templates(test, 
                                      self.tokenizer_eval,
                                      prompt_template_vanilla,
                                      preprocess_function_generation)

    
        print("="*20)
        print("EXAMPLE TRAINING CHAT TEMPLATES")
        print("Train instance (label) 1:\n", train_chat_label[0]['text'], "\n")
        print("Train instance (rationale) 2:\n", train_chat_rationale[1]['text'], "\n")
        print("="*20)
        print("EXAMPLE TEST CHAT TEMPLATES")
        print("Test instance 1:\n", test_chat[0]['text'], "\n")
        print("Test instance 2:\n", test_chat[1]['text'], "\n")
        print("="*20)

        # Critical: combine label and rationale ds
        train_combined = concatenate_datasets([train_chat_label, train_chat_rationale]).shuffle(seed=42)
        print("="*20)
        print("Original training data set size: ", self.training_size)
        print("Mixed training data set size: ", len(train_combined))
        print("="*20)

        # Tokenize data with the ATL tokenizer
        train_tok = tokenize_ds_ATL(train_combined, self.tokenizer_train)
        dev_train_tok = tokenize_ds_ATL(dev_train_chat, self.tokenizer_train)
        
        # For eval, we do not need ATL
        dev_test_tok = tokenize_ds(dev_test_chat, self.tokenizer_eval)
        test_tok = tokenize_ds(test_chat, self.tokenizer_eval)
        
        print("="*20)
        print("EXAMPLE ASSISTANT TOKEN LOSS --- LABEL")
        check_assitant_token_lables(self.tokenizer_train, train_tok[0])
        check_assitant_token_lables(self.tokenizer_train, train_tok[1])
        check_assitant_token_lables(self.tokenizer_train, train_tok[2])
        print("="*20)

        # assign to attributes
        self.train_tok = train_tok
        self.dev_train_tok = dev_train_tok
        self.dev_test_tok = dev_test_tok
        self.test_tok = test_tok

        if self.smoke_test:
            self.train_tok = train_tok.select(range(96))
            self.dev_train_tok = dev_train_tok.select(range(96))
            self.dev_test_tok = dev_test_tok.select(range(32))
            self.test_tok = self.test_tok.select(range(32))


    def _prepare_data_basic(self, 
                            data_dir: str, 
                            prompt_template_train: str,
                            prompt_template_test: str):
        # get all data
        train, dev, test = get_all_data_sets(data_dir, self.lang)

        if self.training_size != -1:
            train = self.resample_train_data(train, self.training_size)
            print("="*20)
            assert self.training_size == len(train), f"New training data size does not match target"
            print(f"Training data resampled to {len(train)}")
            print("="*20)

        # apply chat templates
        train_chat = ds_apply_chat_templates(train, 
                                        self.tokenizer_train,
                                        prompt_template_train,
                                        preprocess_function_training)

        dev_train_chat = ds_apply_chat_templates(dev, 
                                      self.tokenizer_train,
                                      prompt_template_test,
                                      preprocess_function_training)

        dev_test_chat = ds_apply_chat_templates(dev, 
                                      self.tokenizer_eval,
                                      prompt_template_test,
                                      preprocess_function_generation)

        test_chat = ds_apply_chat_templates(test, 
                                      self.tokenizer_eval,
                                      prompt_template_test,
                                      preprocess_function_generation)

        print("="*20)
        print("EXAMPLE TRAINING CHAT TEMPLATES")
        print("Train instance 1:\n", train_chat[0]['text'], "\n")
        print("Train instance 2:\n", train_chat[1]['text'], "\n")
        print("="*20)
        print("EXAMPLE TEST CHAT TEMPLATES")
        print("Test instance 1:\n", test_chat[0]['text'], "\n")
        print("Test instance 2:\n", test_chat[1]['text'], "\n")
        print("="*20)

        # Tokenize data with the ATL tokenizer
        train_tok = tokenize_ds_ATL(train_chat, self.tokenizer_train)
        dev_train_tok = tokenize_ds_ATL(dev_train_chat, self.tokenizer_train)
        
        # For eval, we do not need ATL
        dev_test_tok = tokenize_ds(dev_test_chat, self.tokenizer_eval)
        test_tok = tokenize_ds(test_chat, self.tokenizer_eval)
        
        print("="*20)
        print("EXAMPLE ASSISTANT TOKEN LOSS")
        check_assitant_token_lables(self.tokenizer_train, train_tok[0])
        print("")
        check_assitant_token_lables(self.tokenizer_train, dev_train_tok[0])
        print("="*20)

        # assign to attributes
        self.train_tok = train_tok
        self.dev_train_tok = dev_train_tok
        self.dev_test_tok = dev_test_tok
        self.test_tok = test_tok

        if self.smoke_test:
            self.train_tok = train_tok.select(range(96))
            self.dev_train_tok = dev_train_tok.select(range(96))
            self.dev_test_tok = dev_test_tok.select(range(32))
            self.test_tok = self.test_tok.select(range(32))

class SLMClassifier(LM):
    """
    Class that uses a Classification Head instead of a LM head.
    """
    def __init__(self, 
                 model_type: str,
                 model_name: str,
                 lang: str,
                 smoke_test: bool,
                 quantization: bool,
                 context: bool,
                 training_size: int):
        super().__init__(model_type=model_type,
                         model_name=model_name, 
                         lang=lang, 
                         smoke_test=smoke_test, 
                         quantization=quantization, 
                         training_size=training_size,
                         context=context
                         )
        
        # Only need one tokenizer and one dev set
        self.tokenizer = None
        self.dev_tok = None
    
    def build(self):
        """
        Overwrite the build function to accomodate classification head.
        """
        
        # Load tokenizers (tokenizer_eval not needed)
        self.tokenizer = get_tokenizer(self.model_type, self.model_name)
        
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

        # preparate data
        self._prepare_data()

        # # add also infor to print statement and to meta
        # if self.training_size != -1:
        #     self._resample_train_data(self.training_size)
        #     print(f"Training data resampled to {len(self.train_tok)}")

        # Print overview
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

        if self.smoke_test:
            self.train_tok = train_tok.select(range(96))
            self.dev_tok = dev_tok.select(range(96))
            self.dev_train_tok = self.dev_tok
            self.test_tok = self.test_tok.select(range(32))


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
    "vanilla": SLMLoRA,
    "atl": SLMATL,
    "classifier": SLMClassifier,
}

def build_slm(model_type: str, 
              model_name: str, 
              lang: str, 
              smoke_test: bool,
              quantization: bool,
              context: bool,
              training_size: int,
              explanation: str):
    cls = SLM_REGISTRY.get(model_type)
    return cls(model_type=model_type,
              model_name=model_name, 
              lang=lang, 
              smoke_test=smoke_test, 
              quantization=quantization, 
              training_size=training_size,
              context=context,
              explanation=explanation
              ).build()