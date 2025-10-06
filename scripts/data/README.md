# Order in wich to run the files

1. `api_fa.py`: downloads all `featured` and `good` articles
2. `api_views.py`: downloads teh x most viewed pages in the last 30 days (from today) for each language
3. `aggregate`: combines all three data source in a single file per language (files end with `_all.jsonl`)
4. `raw.py`: downloads all the pages in raw (i.e., html) format
5. `parse.py`: parses the htmls and extracts text by section
6. `sents.py`: retrieves sentences
7. `prepare_ds.py`: for each language prepares the datasets for the models

