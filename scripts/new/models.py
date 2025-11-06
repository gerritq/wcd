
from typing import Optional, List

from utils import (MODEL_MAPPING,
                   get_tokenizer,
                   preprocess_function_training,
                   preprocess_function_generation,
                   lora_config_lm)
from transformers import (AutoTokenizer,
                          AutoModelForCausalLM)
from peft import (LoraConfig, 
                  get_peft_model,
                  prepare_model_for_kbit_training)

from prompts import SYSTEM_PROMPTS_SLM

BASE_DIR = os.getenv("BASE_WCD")
DATA_DIR = os.path.join(BASE_DIR, "data/sets/main")


"""
1. change the prompt with examples

"""
# if args.pwl:
#     MODEL_DIR = os.path.join(BASE_DIR, "data/models/slm/pwl_hp")
#     TEMP_DIR = os.path.join(MODEL_DIR, ".rnd_pwl_hp")
#     os.makedirs(TEMP_DIR, exist_ok=True)
#     TEMP_DIR = tempfile.mkdtemp(dir=TEMP_DIR)
# else:
#     MODEL_DIR = os.path.join(BASE_DIR, "data/models/slm/van_hp")
#     TEMP_DIR = os.path.join(MODEL_DIR, ".rnd_van_hp")
#     os.makedirs(TEMP_DIR, exist_ok=True)
#     TEMP_DIR = tempfile.mkdtemp(dir=TEMP_DIR)

class SLM:
    """Abstract class for SLMs"""
    def __init__(self, model_name: str, lang: str):
        self.model_name = MODEL_MAPPING[model_name]
        
        self.tokenizer_train = None
        self.tokenizer_eval = None
        self.model = None
        self.prompt_template = SYSTEM_PROMPTS_SLM
        
        self.train_tok = None
        self.dev_tok = None
        self.test_tok = None

    def build(self):
        """Standard for LM init"""
        # Load tokenizers
        self.tokenizer_train = get_tokenizer(self.model_name)
        self.tokenizer_eval = get_tokenizer(self.model_name, inference=True) # changes to left padding
        
        # Load model and peft
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_cfg)
        self.model = model

        # preparate data
        self._prepare_data()
        return self

class SLMQLoRA(SLM):
    def __init__(self, model_name):
        super().__init__(model_name)
    
    def _prepare_data(self):
        # get all data
        train, dev, test = get_all_data_sets()

        # apply chat templates
        train_chat = ds_apply_chat_templates(train, 
                                        self.tokenizer_train,
                                        self.prompt_template,
                                        preprocess_function_training)

        dev_chat = ds_apply_chat_templates(dev, 
                                      self.tokenizer_train,
                                      self.prompt_template,
                                      preprocess_function_training)

        test_chat = ds_apply_chat_templates(test, 
                                      self.tokenizer_eval,
                                      self.prompt_template,
                                      preprocess_function_generation)

        # tok data
        train_tok = tokenise_ds(train_chat, self.tokenizer_train)
        dev_tok = tokenise_ds(dev_chat, self.tokenizer_train)
        test_tok = tokenise_ds(test_chat, self.tokenizer_eval)
        
        self.train_tok = train_tok
        self.dev_tok = dev_tok
        self.test_tok = test_tok


SLM_REGISTRY = {
    "qlora": SLMQLoRA,
    "qlora_pwl": SLMPWL,
    "qlora_cd": SLMClassificationHead,
}

def build_slm(model_type, model_name):
    model = SLM_REGISTRY.get(model_type)
    return model(model_name).build()