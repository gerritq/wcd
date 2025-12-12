from sentence_transformers import SentenceTransformer
import torch
import numpy as np
import random
from datasets import load_from_disk
import os
from torch.optim import AdamW as TorchAdamW
import transformers
transformers.AdamW = TorchAdamW

BASE_DIR = os.getenv("BASE_WCD", ".")
IN_DIR = os.path.join(BASE_DIR, "data/sents_no_article_split")

def cosine_sim(a, b):
    return float(torch.nn.functional.cosine_similarity(a, b, dim=0))

def compute_centroid_similarity(train_texts, test_texts,
                                model_name="sentence-transformers/all-MiniLM-L6-v2"):

    model = SentenceTransformer(model_name)

    # Encode (normalized embeddings)
    train_emb = model.encode(train_texts, convert_to_tensor=True, normalize_embeddings=True)
    test_emb  = model.encode(test_texts,  convert_to_tensor=True, normalize_embeddings=True)

    # Compute centroids
    train_centroid = train_emb.mean(dim=0)
    test_centroid  = test_emb.mean(dim=0)

    # Normalize centroids
    train_centroid = train_centroid / train_centroid.norm()
    test_centroid  = test_centroid / test_centroid.norm()

    sim = cosine_sim(train_centroid, test_centroid)

    return {
        "centroid_similarity": sim,
        "train_size": len(train_texts),
        "test_size": len(test_texts)
    }


# ----------------------------
# Example usage
# ----------------------------

def main():
    languages = ["en"]
    for lang in languages:
        data = load_from_disk(os.path.join(IN_DIR, lang))
        train = data["train"]
        test  = data["test"]
        train_claims = train["claim"]
        test_claims  = test["claim"]

        stats = compute_centroid_similarity(train_claims, test_claims)
        print(f"Language: {lang}")
        print(stats)