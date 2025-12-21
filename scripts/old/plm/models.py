import os
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
)
from utils import (MODEL_MAPPING,
            append_meta_file,
            get_model_number,
            collect_and_save_losses,
            get_all_data_sets,
            tokenize_data)

from abc import ABC, abstractmethod
from peft import (prepare_model_for_kbit_training)
from torch.utils.data import Dataset

"""


"""

BASE_DIR = os.getenv("BASE_WCD")
DATA_DIR = os.path.join(BASE_DIR, "data/sets/main")
MODEL_DIR = os.path.join(BASE_DIR, "data/models/plm")

class PLM(ABC):
    """Abstract class for PLMs"""
    def __init__(self, 
                 model_name: str, 
                 lang: str,
                 training_size: int,
                 context: bool,
                 smoke_test: bool):

        self.model_name = MODEL_MAPPING[model_name]
        self.lang = lang
        self.smoke_test = smoke_test
        self.context = context

        # Dirs
        self.data_dir = DATA_DIR
        self.model_dir = None

        # Tokenizer and model
        self.tokenizer = None
        self.model = None

        # Data
        self.training_size = training_size
        self.train_tok = None
        self.dev_tok = None
        self.test_tok = None

    def _print_setting(self):
        print("="*20)
        print("BUILT MODEL")
        print("Model type:", self.model_type)
        print("Model name:", self.model_name)
        print("Lang:", self.lang)
        print("Training data N:", len(self.train_tok))
        print("Smoke test:", self.smoke_test)
        print("="*20)

    
    @abstractmethod
    def _prepare_data(self):
        pass

    def _resample_train_data(self, total_size: int) -> Dataset:
        """
        Resample self.train_tok to total_size with 50/50 label balance.
        Safely handles cases where one class has fewer samples.
        """
        assert total_size % 2 == 0, "Total size must be even."
        n_per_label = total_size // 2

        pos_all = [x for x in self.train_tok if x["label"] == 1]
        neg_all = [x for x in self.train_tok if x["label"] == 0]

        pos = pos_all[:min(len(pos_all), n_per_label)]
        neg = neg_all[:min(len(neg_all), n_per_label)]

        combined = pos + neg
        random.shuffle(combined)

        new_ds = Dataset.from_list(combined)
        self.train_tok = new_ds
    
    def build(self):
        """Standard build function"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
                                                self.model_name, 
                                                num_labels=2,
                                                device_map="auto"
        )

        model.config.use_cache = False 
        self.model = model

        # preparate data
        self._prepare_data()


        if self.training_size != -1:
            self._resample_train_data(self.training_size)
            print(f"Training data resampled to {len(self.train_tok)}")

            # self.model_dir = os.path.join(MODEL_DIR, "length", self.lang)
            # os.makedirs(self.model_dir, exist_ok=True)

        # Print overview
        self._print_setting()
        return self

class VanillaPLM(PLM):
    def __init__(self, 
                 model_name: str, 
                 lang: str,
                 training_size: int,
                 context: bool,
                 smoke_test: bool):
        super().__init__(model_name=model_name, 
                         lang=lang, 
                         training_size=training_size, 
                         smoke_test=smoke_test,
                         context=context)
        
        self.model_type = "Vanilla"

        self.model_dir = os.path.join(MODEL_DIR, self.lang)
        os.makedirs(self.model_dir, exist_ok=True)

    def _prepare_data(self):
        # get all data
        train, dev, test = get_all_data_sets(self.data_dir, self.lang)

        self.train_tok = tokenize_data(train, self.tokenizer, self.context)
        self.dev_tok = tokenize_data(dev, self.tokenizer, self.context)
        self.test_tok = tokenize_data(test, self.tokenizer, self.context)
        
        if self.smoke_test:
            self.train_tok = self.train_tok.select(range(32))
            self.dev_tok = self.dev_tok.select(range(16))
            self.test_tok = self.test_tok.select(range(16))

    
PLM_REGISTRY = {
    "vanilla": VanillaPLM,
}

def build_plm(model_type: str, 
              model_name: str,
              lang: str,
              training_size: int,
              context: bool,
              smoke_test: bool):
    cls = PLM_REGISTRY.get(model_type)
    return cls(model_name=model_name,
               lang=lang,
               training_size=training_size, 
               smoke_test=smoke_test,
               context=context).build()