import argparse as ev
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from common import log

def eval_baseline(inp: str, model_path: str):
    import joblib
    df = pd.read_csv(inp)
    test = df[df.split == "test"]
    pipe = joblib.load(model_path)
    preds = pipe.predict(test.text)
    print(classification_report(test.label, preds, digits = 3))
    print("Confusion:\n", confusion_matrix(test.label, preds, labels = ["REAL", "FAKE", "AI_GENERATED"]))

def eval_transformer(inp: str, model_dir: str):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    df = pd.read_csv(inp)
    test = df[df.split == "test"]
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    labels = ["REAL", "FAKE", "AI-GENERATED"]
    y_pred_lbl = []
    for t in test.text.tolist():
        enc = tok(t, return_tensors = "pt", truncation = True, padding = True, max_length = 32)
        with torch.no_grad():
            out = model(**enc).logits
        idx = int(out.argmax(dim = 1).item())
        y_pred_lbl.append(labels[idx])
    print(classification_report(test.label, y_pred_lbl, digits = 3))

if __name__ == "__main__":
    p = ev.ArgumentParser()
    p.add_argument("--in", dest="inp", default="data/clean/dataset_splits.csv",
                   help="Path to the dataset with train/dev/test splits")
    p.add_argument("--baseline", default="",
                   help="Path to a saved baseline model .joblib")
    p.add_argument("--transformer", default="",
                   help="Path to a fine-tuned transformer model directory")
    args = p.parse_args()

    if args.baseline:
        eval_baseline(args.inp, args.baseline)
    elif args.transformer:
        eval_transformer(args.inp, args.transformer)
    else:
        log("Provide --baseline or --transformer", "err")