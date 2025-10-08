# scripts/evaluate.py
import argparse
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from common import log

LABELS = ["REAL", "FAKE", "AI_GENERATED"]

def save_confusion_png(y_true, y_pred, labels, out_png):
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    plt.figure()
    disp.plot(xticks_rotation=45, values_format='d', colorbar=False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def eval_baseline(inp: str, model_path: str):
    import joblib
    df = pd.read_csv(inp)
    test = df[df.split == "test"]
    pipe = joblib.load(model_path)
    preds = pipe.predict(test.text)
    save_confusion_png(test.label, preds, LABELS, "reports/confusion_baseline.png")
    print(classification_report(test.label, preds, labels=LABELS, digits=3))
    print("Confusion:\n", confusion_matrix(test.label, preds, labels=LABELS))

def eval_transformer(inp: str, model_dir: str):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    import numpy as np
    import pandas as pd

    df = pd.read_csv(inp)
    test = df[df.split == "test"].reset_index(drop=True)

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    enc = tok(list(test.text), padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        logits = model(**enc).logits
    preds_idx = logits.argmax(dim=1).cpu().numpy()

    # map ids to labels
    id2label = model.config.id2label if hasattr(model.config, "id2label") else {i: l for i, l in enumerate(LABELS)}
    preds = [id2label[int(i)] for i in preds_idx]

    save_confusion_png(test.label, preds, LABELS, "reports/confusion_transformer.png")

    import pandas as pd, torch
    probs = torch.softmax(logits, dim=1).max(dim=1).values.cpu().numpy()
    pd.DataFrame({
        "text": test.text,
        "gold": test.label,
        "pred": preds,
        "confidence": probs
    }).query("gold != pred").to_csv("reports/transformer_errors.csv", index=False)

    import pandas as pd, torch
    probs = torch.softmax(logits, dim=1).max(dim=1).values.cpu().numpy()
    pd.DataFrame({"text": test.text, "gold": test.label, "pred": preds, "confidence": probs}) \
    .query("gold != pred").to_csv("reports/transformer_errors.csv", index=False)

    print(classification_report(test.label, preds, labels=LABELS, digits=3))
    print("Confusion:\n", confusion_matrix(test.label, preds, labels=LABELS))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", action="store_true")
    p.add_argument("--transformer", action="store_true")
    p.add_argument("--inp", required=True)
    p.add_argument("--model_path")
    p.add_argument("--model_dir")
    args = p.parse_args()

    if args.baseline:
        eval_baseline(args.inp, args.model_path)
    if args.transformer:
        eval_transformer(args.inp, args.model_dir)