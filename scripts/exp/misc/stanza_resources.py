import stanza
import os

def main():
    "see whehter lang is available by trying to load the parser"
    nlp = stanza.Pipeline(lang="zh", 
                            processors="tokenize", 
                            tokenize_pretokenized=False, 
                            dir=os.getenv('HF_HOME'),
                            model_dir=os.getenv('HF_HOME')
                            )
    print("Success")
if __name__ == "__main__":
    main()