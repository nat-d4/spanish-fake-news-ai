import argparse as cl
import pandas as pd
import hashlib
from common import log, write_csv_safe

def main(out: str, reals: str = "data/raw/real.csv", fakes: str = "data/raw/fake.csv", ais: str = "data/raw/ai.csv"):
    parts = []
    for p in [reals, fakes, ais]:
        try:
            parts.append(pd.read_csv(p))
        except Exception:
            log(f"missing or unreadable: {p}", "warn")
    if not parts:
        log("No input CSVs found", "err")
        return
    df = pd.concat(parts, ignore_index = True)
    df = df.drop_duplicates(subset = ["text"], keep = "first")
    write_csv_safe(df, out)
    log(f"Clean dataset size: {len(df)} → {out}", "ok")

if __name__ == "__main__":
    p = cl.ArgumentParser()
    p.add_argument("--out", default = "data/clean/dataset.csv")
    p.add_argument("--real", default = "data/raw/real.csv")
    p.add_argument("--fake", default = "data/raw/fake.csv")
    p.add_argument("--ai", default = "data/raw/ai.csv")
    a = p.parse_args()
    main(a.out, a.real, a.fake, a.ai)