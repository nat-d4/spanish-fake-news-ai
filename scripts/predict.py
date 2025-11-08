# scripts/predict.py
import argparse
import sys
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

LABELS = ["REAL", "FAKE", "AI_GENERATED"]

def load_model(model_dir: str):
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    return tok, model

def predict_headlines(headlines, model_dir: str, max_len: int = 256):
    """
    Returns (df, probs_np)
      - df: DataFrame with columns [headline, label, confidence]
      - probs_np: numpy array of shape (N, num_labels) with per-class probabilities
    """
    if not headlines:
        raise ValueError("No headlines provided. Use --headline \"...\" [\"...\"]")

    tok, model = load_model(model_dir)
    enc = tok(headlines, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=1)

    pred_ids = probs.argmax(dim=1)
    rows = []
    for h, pid, pr in zip(headlines, pred_ids, probs):
        pid = int(pid)
        conf = float(pr[pid].item())
        rows.append({"headline": h, "label": LABELS[pid], "confidence": conf})

    df = pd.DataFrame(rows)
    probs_np = probs.cpu().numpy()
    return df, probs_np

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--headline", nargs="+", help="One or more Spanish headlines to classify")
    p.add_argument("--model_dir", default="models/beto-targeted")
    p.add_argument("--max_len", type=int, default=256)
    a = p.parse_args()

    try:
        df, probs_np = predict_headlines(a.headline, a.model_dir, a.max_len)

        # Pretty table
        pd.set_option("display.max_colwidth", 200)
        print(df.to_string(index=False))

        # Top-2 per-class breakdown for each headline
        for h, row in zip(a.headline, probs_np):
            by_label = sorted(zip(LABELS, row), key=lambda x: x[1], reverse=True)
            top2 = ", ".join([f"{lbl}={p:.3f}" for lbl, p in by_label[:2]])
            all_probs = " ".join(f"{lbl}={p:.3f}" for lbl, p in by_label)
            print(f"- {h}\n  → top-2: {top2} | all: {all_probs}")

        sys.exit(0)
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
