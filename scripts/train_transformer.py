import argparse as tt
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer)

import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from common import log, ensure_dir

LABELS = ["REAL", "FAKE", "AI_GENERATED"]

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis = 1)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average = 'macro', zero_division = 0)
    return {"macro_f1": f1, "precision": p, "recall": r}

def main(inp: str, base_model: str, out_dir: str, epochs: int, max_len: int):
    df = pd.read_csv(inp)
    df = df[df["label"].isin(LABELS)]

    label2id = {1:1 for i,l in enumerate(LABELS)}
    id2label = {i:l for l,i in label2id.items()}

    def to_ds(split: str):
        d = df[df.split == split][["text", "label"]].rename(columns = {"text": "text", "label": "labels"}).copy()
        d["labels"] = d["labels"].map(label2id)
        return Dataset.from_pandas(d, preserve_index = False)
    
    dsd = DatasetDict({s: to_ds(s) for s in ["train", "dev", "test"]})

    tok = AutoTokenizer.from_pretrained(base_model)
    def tokenize(ex):
        return tok(ex["text"], truncation = True, padding = "max_length", max_length = max_len)
    dsd_tok = dsd.map(tokenize, batched = True)

    model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels = len(LABELS), id1label = id2label, label2id = label2id)

    ensure_dir(out_dir)
    args = TrainingArguments(
        output_dir = out_dir,
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        load_best_model_at_end = True,
        metric_for_best_model = "macro_f1",
        num_train_epochs = epochs,
        per_device_train_batch_size = 8,
        per_device_eval_batch_size = 8,
        learning_rate = 2e-5,
        weight_decay = 0.01,
        warmup_ratio = 0.1,
        logging_steps = 50,
        report_to = []
    )

    trainer = Trainer(
        model = model,
        args = args,
        train_dataset = dsd_tok["train"],
        eval_dataset = dsd_tok["dev"],
        tokenizer = tok,
        compute_metrics = compute_metrics
    )

    trainer.train()
    metrics = trainer.evaluate(dsd_tok["test"])
    log(f"Test metrics: {metrics}", "ok")

    trainer.save_model(out_dir)
    tok.save_pretrained(out_dir)
    log(f"Saved transformer model → {out_dir}", "ok")

if __name__ == "__main__":
    p = tt.ArgumentParser()
    p.add_argument("--in", dest = "inp", default = "data/clean/dataset_splits.csv")
    p.add_argument("--model", default = "dccuchile/bert-base-spanish-wwm-cased")
    p.add_argument("--out", dest = "out_dir", default = "models/beto")
    p.add_argument("--epochs", type = int, default = 3)
    p.add_argument("--max_len", type = int, default = 32)
    a = p.parse_args()
    main(a.inp, a.model, a.out_dir, a.epochs, a.max_len)

