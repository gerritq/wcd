# Order in wich to run the files

1. `api_fa.py` - `api_random.py`: downloads all `featured`, `good`, and (750) `random` articles
    - `api_views.py`: downloads the x most viewed pages in the last 30 days (from today) for each language
3. `aggregate`: combines `fa` and `random` in a single file per language (files end with `_all.jsonl`)
4. `raw.py`: downloads all the pages in raw (i.e., html) format
    - Restrict all langs to FA and random articles. There are so many we do not need good articles or views.
5. `parse.py`: parses the htmls and extracts text by section
6. `sents.py`: retrieves sentences
7. `prepare_ds.py`: for each language prepares the datasets for the models

# Miscellanious
- `checks.py`: prints random number of sentences
- `shots.py`: randomly draws shots for in-context learning
- `wiki_langs.py`: scrapes the Wikipedia by languauge page and creates a df
- `counts.py`: count number of articles and labels by source of the final data
- `skip_sections.py`: file from which to load sections to be skipped
- `proc_cn.py`: process citations needed original data
- `print_article.py`: prints an article
- `linguistics.py`: generates a confusion matrix of the elinguistic metric
