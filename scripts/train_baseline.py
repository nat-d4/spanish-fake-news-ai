import argparse as tb
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib
from common import log, ensure_dir

def main(inp: str, out: str):
    df = pd.read_csv(inp)
    train = df[df.split == "train"]
    dev = df[df.split == "dev"]

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1,2), sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=200, n_jobs=-1)),
    ])
    pipe.fit(train.text, train.label)

    preds = pipe.predict(dev.text)
    rep = classification_report(dev.label, preds, digits = 3)
    log("Dev report: \n" + rep)

    ensure_dir(out.rsplit("/", 1)[0])
    joblib.dump(pipe, out)
    log(f"Saved baseline model → {out}", "ok")

    if __name__ == "__main__":
        p = tb.ArgumentParser()
        p.add_argument("--in", dest = "inp", default = "data/clean/dataset_splits.csv")
        p.add_argument("--out", default = "models/baseline.joblib")
        a = p.parse_args()
        main(a.inp, a.out)