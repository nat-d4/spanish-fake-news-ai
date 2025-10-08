import argparse as sp
import pandas as pd
import numpy as np
from common import log, write_csv_safe

def assign_splits(df: pd.DataFrame, seed: int = 13) -> pd.DataFrame:
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)  # shuffle + reset index
    n = len(df)
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)
    df.loc[:train_end-1, "split"] = "train"
    df.loc[train_end:val_end-1, "split"] = "val"
    df.loc[val_end:, "split"] = "test"
    return df

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