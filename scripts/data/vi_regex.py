# quick test whether re works with vi

import re


text = "Tiếng Việt là ngôn ngữ thanh điệu mọi âm tiết của tiếng Việt luôn mang 1 thanh điệu nào đó"

# Pattern to match words (\w includes Unicode letters and numbers)
pattern = r"\w+"

matches = re.findall(pattern, text)
print(" ".join(matches) == text)
print(f"Number of tokens: {len(matches)}")