from datasets import load_from_disk
from transformers import AutoTokenizer

# LLaMA tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

data_dir = "/scratch/prj/inf_nlg_ai_detection/wcd/data/sets/main/nl"
ds = load_from_disk(data_dir)

dataset = ds["train"]
prompt =(   "You are a multilingual classifier. "
            "Decide whether the given English claim requires a citation. "
            "Use 1 if the claim needs a citation. Use 0 if the claim does not need a citation.\n\n"
            "Return only JSON in the format: {\"label\": 0} or {\"label\": 1}. "
            "No explanations or extra text.")

TITLE_COL = "title"
SECTION_COL = "section"
CLAIM_COL = "claim"
NEXT_COL = "next_sentence"
PREV_COL = "previous_sentence"

def concat_fields(example):
    prev = example.get(PREV_COL, "") or ""
    claim = example.get(CLAIM_COL, "") or ""
    nxt = example.get(NEXT_COL, "") or ""
    text = " ".join([s for s in [prompt,  prev, claim, nxt] if s.strip()])

    title = example.get(TITLE_COL, "") or ""

    # tokenized length (LLaMA tokenizer)
    toks = tokenizer(text, add_special_tokens=False)
    tok_len = len(toks["input_ids"])

    return {
        "concat_text": text,
        "concat_len": len(text),
        "tok_len": tok_len,
        "title": title,
    }

dataset_with_concat = dataset.map(concat_fields)

# sort by tokenized length instead of raw text length
sorted_indices = sorted(
    range(len(dataset_with_concat)),
    key=lambda i: dataset_with_concat[i]["tok_len"],
    reverse=True,
)

top10_indices = sorted_indices[:10]

for rank, idx in enumerate(top10_indices, start=1):
    row = dataset_with_concat[idx]
    print("=" * 80)
    print(f"#{rank} (index: {idx})")
    print(f"Raw length:       {row['concat_len']}")
    print(f"Tokenized length: {row['tok_len']}")
    print("TITLE:", row["title"])
    print(row["concat_text"])
    print()