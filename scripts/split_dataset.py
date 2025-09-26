import argparse as sp
import pandas as pd
import numpy as np
from common import log, write_csv_safe

def assign_splits(df: pd.DataFrame, seed: int = 13) -> pd.DataFrame:
    df = df.copy()
    rng = np.random.default_rng(seed)
    rows = df.sample(frac = 1, random_state = seed)
    n = len(rows)
    train_end = int(0.7 * n)
    dev_end = int(0.85 * n)
    rows.loc[:train_end, "split"] = "train"
    rows.loc[train_end:dev_end, "split"] = "dev"
    rows.loc[dev_end:,  "split"] = "test"
    return rows

def main(inp: str, out: str, seed: int):
    df = pd.read_csv(inp)
    df = assign_splits(df, seed)
    write_csv_safe(df, out)
    log(f"Split counts: {df['split'].value_counts().to_dict()}", "ok")

if __name__ == "__main__":
    p = sp.ArgumentParser()
    p.add_argument("--in", dest="inp", default="data/clean/dataset.csv")
    p.add_argument("--out", default="data/clean/dataset_splits.csv")
    p.add_argument("--seed", type=int, default=13)
    a = p.parse_args()
    main(a.inp, a.out, a.seed)