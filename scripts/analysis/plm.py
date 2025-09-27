import json
import random
import numpy as np
import evaluate
import torch

from datasets import Dataset, ClassLabel, DatasetDict
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    set_seed, 
    TrainingArguments, 
    Trainer
)


# --- Settings ---
LANG = 'en'
INPUT_PATH = f"../../data/sets/{LANG}_sents.jsonl"
MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 256
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

def generate_dataset():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data_in = [json.loads(line) for line in f]
        data=[]
        for x in data_in:
            x = {k: v for k,v in x.items() if k in ['sentence_clean', 'label_2']}
            x['label'] = x.pop('label_2')
            data.append(x)

    random.shuffle(data)

    positives = [x for x in data if x['label'] == 1][:3000]
    negatives = [x for x in data if x['label'] == 0][:3000]

    # Combine and shuffle again
    data = positives + negatives
    print("N", len(data))
    random.shuffle(data)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(example):
        return tokenizer(
            example["sentence_clean"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
        )

    dataset = Dataset.from_list(data)

    split = dataset.train_test_split(test_size=0.2, seed=SEED)
    val_test = split["test"].train_test_split(test_size=0.5, seed=SEED)
    dataset_dict = DatasetDict({
        "train": split["train"],
        "val": val_test["train"],
        "test": val_test["test"]
    })

    dataset = dataset_dict.map(tokenize, batched=True)
    return dataset

def main():
    dataset = generate_dataset()

    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    def model_init():
        return AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=2
        )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        acc = accuracy_metric.compute(predictions=predictions, references=labels)
        f1 = f1_metric.compute(predictions=predictions, references=labels, average="binary")

        return {
            "accuracy": acc["accuracy"],
            "f1": f1["f1"]
        }

    training_args = TrainingArguments(
        output_dir=None,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        # learning_rate=2e-5,
        eval_strategy="epoch",
        )

    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
        compute_metrics=compute_metrics,
    )

    trainer.train()


if __name__ == "__main__":
    main()