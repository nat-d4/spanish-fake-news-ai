#---------------------------------------------------
# PURPOSE: collect fake items from Spanish fact-checkers
#---------------------------------------------------
import argparse as _arg
import feedparser as _fp
from common import log as _log, write_csv_safe as _wcsv, slugify as _slug
import pandas as pd

FACT_FEEDS = [
    "https://maldita.es/rss/",
    "https://www.chequeado.com/rss/",
    "https://www.newtral.es/feed/",
]

def run(out: str, limit: int):
    rows = []
    for feed in FACT_FEEDS:
        _log("Reading fact-check feed: {feed}")
        d = _fp.parse(feed)
        for entry in d.entries[: limit // len(FACT_FEEDS) + 1]:
            title = entry.title
            rows.append({
                "id": _slug(title) + "-fake",
                "text": title,
                "source": feed,
                "label": "FAKE",
            })
    df = pd.DataFrame(rows)
    _wcsv(df, out)
    _log(f"Wrote {len(df)} FAKE headlines to {out}", "ok")

if __name__ == "__main__":
    p = _arg.ArgumentParser()
    p.add_argument("--out", default = "data/raw/fake.csv")
    p.add_argument("--limit", type = int, default = 200)
    a = p.parse_args()
    run(a.out, a.limit)