import os
import re
import json
import copy
import argparse
from concurrent.futures import ThreadPoolExecutor
from datasets import load_from_disk
from openai import OpenAI

PROMPT = (
    "You are a Wikipedia citation reasoning assistant. "
    "Your task is to determine *why* a given claim does or does not require a citation. "
    "Select the most appropriate category from the list below and provide a concise explanation. "
    "Return your answer as valid JSON in a json block in the format:\n"
    "{{\"category\": your_category, \"explanation\": your_explanation}}\n\n"

    "=== Reasons why citations are needed ===\n"
    "• Quotation – The statement is a direct quotation or close paraphrase of a source.\n"
    "• Statistics – The statement contains statistics or quantitative data.\n"
    "• Controversial – The statement makes surprising or potentially controversial claims (e.g., conspiracy theories).\n"
    "• Opinion – The statement expresses a person’s subjective opinion or belief.\n"
    "• Private Life – The statement contains claims about a person’s private life (e.g., date of birth, relationship status).\n"
    "• Scientific – The statement includes technical or scientific claims.\n"
    "• Historical – The statement makes general or historical claims that are not common knowledge.\n"
    "• Other (Needs Citation) – The statement requires a citation for other reasons (please describe briefly).\n\n"

    "=== Reasons why citations are NOT needed ===\n"
    "• Common Knowledge – The statement only contains well-known or established facts.\n"
    "• Main Section – The statement appears in the lead section and is cited elsewhere in the article.\n"
    "• Plot – The statement describes the plot or characters of a book, film, or similar work that is the article’s subject.\n"
    "• Already Cited – The statement is already supported by a citation elsewhere in the paragraph or article.\n"
    "• Other (No Citation Needed) – The statement does not need a citation for other reasons (please describe briefly).\n\n"

    "Now, analyze the following claim:\n"
    "Claim: {claim}\n"
    "Label: {label}\n\n"
    "Please choose one of the categories above and explain why it applies."
    )

BASE_DIR = os.getenv("BASE_WCD") 
DATA_DIR = os.path.join(BASE_DIR, "data/sets/main")
ANNOTATION_DIR = os.path.join(BASE_DIR, "data/annotation")

def get_dataset(lang):
    """loads train data and returns it as a list of dict"""
    
    ds = load_from_disk(os.path.join(DATA_DIR, lang))
    train = ds['train'].to_list()

    return train

def format_prompt(train):
    """formats the prompt for each claim"""
    for claim in train:
        claim['prompt'] = PROMPT.format(**claim)
    return train

def query(claim, model, client) -> str:

    completion = client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[{"role": "user", "content": claim['prompt']}]
    )

    claim['category'], claim['explanation'] = retrieve_response(completion.choices[0].message.content)
    return claim

def retrieve_response(response: str):
    print(response)
    match = re.search(r"```json\s*(\{.*?\})\s*```", response, re.DOTALL)
    if match:
        json_str = match.group(1)
        print(json_str)
        try:
            data = json.loads(json_str)
            return data["category"], data["explanation"]
        except json.JSONDecodeError:
            return None, None
            print("Failed to decode JSON.")
    else:
        print("No JSON block found in response.")
        return None, None

def run(prompts, model, client):
    def process(claim):
        out = query(claim, model, client)
        return out

    with ThreadPoolExecutor(max_workers=4) as executor:
        annotated_claims = list(executor.map(process, prompts))

    return annotated_claims


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()
    args.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"]
            )

    train = get_dataset(args.lang)
    prompts = format_prompt(train)
    annotated_claims = run(prompts, args.model, args.client)
    
    with open(os.path.join(ANNOTATION_DIR, f"{args.lang}.jsonl"), "w", encoding="utf-8") as f:
        for entry in annotated_claims:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")

if __name__ == "__main__":
    main()

# class LLMAsAJudge:
#     def __init__(self, model: str, max_workers: int = 4):
#         self.model = model
#         self.client = OpenAI(
#             base_url="https://openrouter.ai/api/v1",
#             api_key=os.environ["OPENROUTER_API_KEY"]
#         )
#         self.max_workers = max_workers

#     def _rearrange_subclaims(self, subclaims: List[Subclaim], top_k=3) -> List[Subclaim]:
#         """Create subclaim-evidence pairs"""
#         out = []
#         for subclaim in subclaims:
#             for evidence in subclaim['reranked'][:top_k]:
#                 new_subclaim = copy.deepcopy(subclaim)
#                 new_subclaim['evidence'] = evidence  # this will be a tuple: (text, score, url)
#                 out.append(new_subclaim)
                
#         return out

#     def _format_response(self, response: str) -> List[str]:
#         match = re.search(r"```json\s*(\{.*?\})\s*```", response, re.DOTALL)
#         if match:
#             json_str = match.group(1)
#             try:
#                 data = json.loads(json_str)
#                 return data["label"], data["justification"]
#             except json.JSONDecodeError:
#                 raise ValueError("Failed to decode JSON.")
#         else:
#             raise ValueError("No JSON block found in response.")

#     def _judge(self, claim: Claim) -> Dict[str, str]:
    
#         claim_txt = claim['claim']
#         evidence = claim['evidence'][0] # txt, score, link

#         prompt = LLM_JUDGE_PROMPT.format(
#             claim=claim_txt,
#             evidence=evidence
#         )

#         completion = self.client.chat.completions.create(
#             model=self.model,
#             temperature=0.0,
#             messages=[{"role": "user", "content": prompt}]
#         )

#         result = self._format_response(completion.choices[0].message.content)
#         assert isinstance(result, tuple), f"Judgement is not a tuple: {result}"

#         return result

#     def run(self, subclaims: List[Subclaim]) -> List[Subclaim]:
        
#         print("N of subclaims:", len(subclaims))
#         subclaims = self._rearrange_subclaims(subclaims)
#         print("N of subclaim-evidence pairs (Nx3):", len(subclaims))
        
#         def process(subclaim: Subclaim) -> Subclaim:
#             subclaim['judgement'] = self._judge(subclaim)
#             return subclaim

#         with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
#             subclaims_with_judgements = list(executor.map(process, subclaims))

#         return subclaims_with_judgements