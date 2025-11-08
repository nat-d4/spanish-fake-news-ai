import sys, pandas as pd
from scripts.predict import predict_headlines

def main(model_dir="models/beto-targeted", golden_csv="reports/golden_set.csv", min_acc=0.9):
    gold = pd.read_csv(golden_csv)
    df, _ = predict_headlines(gold["headline"].tolist(), model_dir=model_dir, max_len=256)
    merged = gold.join(df[["label"]].rename(columns={"label":"label_pred"}))
    acc = (merged["label"] == merged["label_pred"]).mean()
    print(merged.to_string(index=False))
    print(f"\nAccuracy on golden set: {acc:.3f}")
    if acc < min_acc:
        sys.exit(1)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", default="models/beto-targeted")
    p.add_argument("--golden_csv", default="reports/golden_set.csv")
    p.add_argument("--min_acc", type=float, default=0.9)
    a = p.parse_args()
    main(a.model_dir, a.golden_csv, a.min_acc)
