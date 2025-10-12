import os
import json
import re
import random
import argparse
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed

from datasets import load_from_disk
from openai import OpenAI
from prompts import PROMPTS
from sklearn.metrics import accuracy_score, f1_score

class LLM:
    def __init__(self, 
                 args, 
                 max_workers: int = 4):
        self.lang = args.lang
        self.model = args.model
        self.notes = args.notes
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"]
        )
        self.max_workers = max_workers
        self.PROMPT = PROMPTS[self.lang]
        self.base_dir = os.getenv("BASE_WCD")
        self.data_dir = os.path.join(self.base_dir, "data/sets")
        self.metrics_dir = os.path.join(self.base_dir, "data/metrics/llm")

    def _get_model_number(self) -> int:
        model_files = [f for f in os.listdir(self.metrics_dir) if f.endswith(".json")]

        numbers = []
        for name in model_files:
            match = re.search(rf"(\d+)\.json$", name)
            if match:
                numbers.append(int(match.group(1)))

        next_number = max(numbers) + 1 if numbers else 1
        return next_number

    def _prepare_test_data(self) -> List[Dict]:
        ds = load_from_disk(os.path.join(self.data_dir, self.lang))["test"]
        def to_messages(example):
            return {
                "messages": [
                    {"role": "system", "content": "You are a seasoned Wikipedia fact-checker."},
                    {"role": "user", "content": self.PROMPT.replace("{{claim}}", example["claim"])},
                ],
                "label": int(example["label"]),
                "claim": example["claim"],
            }
        rows = [to_messages(ex) for ex in ds][:10]
        random.shuffle(rows)
        self.test_n = len(rows)
        return rows

    def _format_response(self, response: str) -> int:
        match = re.search(r"\s*(\{.*?\})\s*", response, re.DOTALL)
        if match:
            json_str = match.group(1)
            try:
                data = json.loads(json_str)
                return int(data["label"])
            except json.JSONDecodeError:
                raise ValueError("Failed to decode JSON.")
        else:
            raise ValueError("No JSON block found in response.")

    def _query(self, messages: List[Dict]) -> int:
            completion = self.client.chat.completions.create(
                model=self.model,
                temperature=0.0,
                messages=messages,
            )
            content = completion.choices[0].message.content
            return self._format_response(content)

    def _eval(self, results: List[Dict]) -> None:
        valid = [r for r in results if r["pred"] is not None]

        self.valid_n = len(valid)

        y_true = [int(r["label"]) for r in valid]
        y_pred = [int(r["pred"]) for r in valid]

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        res = {'model_number': self.model_number, 
                'model': self.model, 
               'lang': self.lang,
               'test_n': self.test_n,
               'valid_n': self.valid_n,
               'notes': self.notes,
               'accuracy': acc, 
               'f1': f1}
        metrics_path = os.path.join(self.metrics_dir, f"{self.lang}_model_{self.model_number}.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)


    def run(self) -> None:
        
        # load data and get model number
        data = self._prepare_test_data()
        self.model_number = self._get_model_number()

        def worker(example: Dict) -> Dict:
            try:
                pred = self._query(example["messages"])
                return {
                    "claim": example["claim"],
                    "label": example["label"],
                    "pred": pred,
                }
            except Exception as e:
                print(e)
                return {
                    "claim": example["claim"],
                    "label": example["label"],
                    "pred": None,
                    "error": str(e),
                }

        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = [ex.submit(worker, exm) for exm in data]
            for fut in as_completed(futures):
                results.append(fut.result())

        # eval
        self._eval(results)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--notes", type=str)
    args = parser.parse_args()

    llm = LLM(args)
    llm.run()