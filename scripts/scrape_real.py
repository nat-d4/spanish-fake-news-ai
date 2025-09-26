#---------------------------------------------
# PURPOSE: collect real spanish news via RSS feeds
#          and readability extraction
#---------------------------------------------
import argparse
import feedparser
from common import log, write_csv_safe, slugify
import pandas as pd


REAL_FEEDS = [
    "https://elpais.com/rss/elpais/portada.xml",
    "https://www.bbc.com/mundo/ultimas_noticias/index.xml",
    "https://www.eldiario.es/rss/",
    "https://www.infobae.com/america/rss/",
]


def main(out: str, limit: int):
    rows = []
    for feed in REAL_FEEDS:
        log(f"Reading feed: {feed}")
        d = feedparser.parse(feed)
        for entry in d.entries[: limit // len(REAL_FEEDS) + 1]:
            title = entry.title
            rows.append({
                "id": slugify(title) + "-real",
                "text": title,
                "source": feed,
                "label": "REAL",
            })
    df = pd.DataFrame(rows)
    write_csv_safe(df, out)
    log(f"Wrote {len(df)} REAL headlines to {out}", "ok")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/raw/real.csv")
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()
    main(args.out, args.limit)