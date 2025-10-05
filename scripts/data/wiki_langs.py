import os
import re
import requests
import pandas as pd
import numpy as np

URL = "https://en.wikipedia.org/wiki/List_of_Wikipedias"
BASE_DIR = os.getenv("BASE_WCD")

HTML_FILE = os.path.join(BASE_DIR, "data/languages/list_of_wikipedias.html")
CSV_FILE = os.path.join(BASE_DIR, "data/languages/wikipedias.csv")

def main():
    print("Scrape/load PDF ...")
    if os.path.exists(HTML_FILE):
        print(f"Loading existing file: {HTML_FILE}")
        with open(HTML_FILE, "r", encoding="utf-8") as f:
            html = f.read()
    else:
        print(f"Scraping from {URL}")
        resp = requests.get(URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        resp.raise_for_status()
        html = resp.text
        with open(HTML_FILE, "w", encoding="utf-8") as f:
            f.write(html)


    print("Proc table ...")
    tables = pd.read_html(html) 

    def has_articles_col(df: pd.DataFrame) -> bool:
        cols = [str(c).strip().lower() for c in df.columns]
        return any("article" in c for c in cols)

    candidates = [t for t in tables if has_articles_col(t)]
    
    if not candidates:
        raise RuntimeError("Could not find a table with an 'Articles' column on the page.")

    # if len(candidates) > 1:
    #     raise RuntimeError("More than one table with article columns.")

    df = candidates[0]
    
    df.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in df.columns]


    print("Define tertiles ...")
    # df_sorted = df.sort_values('Articles', ascending=False).head(100).reset_index(drop=True)
    # df_filtered = df[df['Articles'] >= 100_000].copy()
    # df_sorted = df_filtered.sort_values('Articles', ascending=False).reset_index(drop=True)

    df["Share"] = round(df["Articles"] / df["Articles"].sum() *100, 2)

    # sort by share
    df_sorted = df.sort_values("Share", ascending=False).reset_index(drop=True)

    df_sorted["tertile"] = np.where(
    df_sorted["Share"] > 1, "High",
    np.where(
        df_sorted["Share"] > .1, "Medium",
        np.where(
            df_sorted["Share"] > .01, "Low",
            "Extremely-Low"
        )
    )
)

    # labels = ["low", "mid", "high"]
    # df_sorted["tertile"] = pd.qcut(df_sorted["Articles"].astype(int), q=3, labels=labels)

    out = df_sorted[["Language", "Articles", "Share", "tertile"]].rename(
        columns={"Language": "language", "Articles": "articles", "Share": "share"}
    )

    out.to_csv(CSV_FILE, index=False, encoding="utf-8")

if __name__ == "__main__":
    main()