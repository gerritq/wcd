import json
import os
BASE_DIR = os.getenv("BASE_WCD", ".")
INPUT_DIR = os.path.join(BASE_DIR, f"data/raw/parsed") 

languages  = [
    "en",  # English
    "nl",  # Dutch
    "no",  # Norwegian (Bokmål is 'nb', Nynorsk is 'nn', 'no' redirects to Bokmål)
    "it",  # Italian
    "pt",  # Portuguese
    "ro",  # Romanian
    "ru",  # Russian
    "uk",  # Ukrainian
    "bg",  # Bulgarian
    "id",  # Indonesian
    "vi",  # Vietnamese
    "tr"  # Turkish
]

# check textacy
# import textacy.preprocessing as tprep

# x = "Situl„Pădurea Orléans” se întinde din nord-estul aglomerației orléaneze până la porțile orașului Gien, urmând un arc de cerc de aproximativ 60 de kilometri lungime și având o lățime variabilă între aproximativ 2 și 15 km."
# x = tprep.normalize_quotation_marks(x)
# print(x)

print('works')


# Check counts of random
# for lang in languages:
#         in_file = os.path.join(INPUT_DIR,  f"{lang}_parsed.jsonl")
#         with open(in_file, "r", encoding="utf-8") as f:
#             data = [json.loads(line) for line in f]
#             data = [x for x in data if x['source'] == "random"]
#             print(len(data))

# Chekcing htmls 

# for lang in languages:
#     in_file = os.path.join(INPUT_DIR,  f"{lang}_htmls.jsonl")
#     with open(in_file, "r", encoding="utf-8") as f:
#         data = [json.loads(line) for line in f]
#         data = [x for x in data if x['title'] == "Mecanica fluidelor numerică"]

# from bs4 import BeautifulSoup as bs
# soup = bs(data[0]['raw'], "html")                #make BeautifulSoup
# prettyHTML = soup.prettify() 
# print(prettyHTML)
    