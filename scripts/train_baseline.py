import argparse as tt
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import precision_recall_fscore_support
from common import log, ensure_dir

LABELS = ["REAL", "FAKE", "AI_GENERATED"]

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # HF sometimes returns (logits,) — handle both
    if isinstance(logits, (list, tuple)):
        logits = logits[0]
    preds = np.argmax(logits, axis=1)
    p, r, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    return {"macro_f1": f1, "precision": p, "recall": r}

def main(inp: str, base_model: str, out_dir: str, epochs: int, max_len: int):
    df = pd.read_csv(inp)
    df = df[df["label"].isin(LABELS)].reset_index(drop=True)

    # Correct label maps
    label2id = {l: i for i, l in enumerate(LABELS)}
    id2label = {i: l for l, i in label2id.items()}

    # Build HF datasets, handling "val" or "dev"
    def to_ds(split: str):
        sub = df[df.split == split][["text", "label"]].copy()
        if sub.empty:
            return None
        sub = sub.rename(columns={"label": "labels"})
        sub["labels"] = sub["labels"].map(label2id)
        return Dataset.from_pandas(sub, preserve_index=False)

    train_ds = to_ds("train")
    val_ds = to_ds("val") or to_ds("dev")
    test_ds = to_ds("test")
    if train_ds is None or val_ds is None or test_ds is None:
        raise ValueError("Expected splits train/val(test) (or dev) missing in dataset.csv")

    dsd = DatasetDict({"train": train_ds, "val": val_ds, "test": test_ds})

    tok = AutoTokenizer.from_pretrained(base_model)

    def tokenize(batch):
        return tok(batch["text"], truncation=True, padding="max_length", max_length=max_len)

    dsd_tok = dsd.map(tokenize, batched=True, remove_columns=["text"])

    # Model with correct label mapping
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=len(LABELS),
        id2label=id2label,
        label2id=label2id,
    )

    ensure_dir(out_dir)

    args = TrainingArguments(
        output_dir=out_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=50,
        report_to=[],  # no W&B, etc.
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dsd_tok["train"],
        eval_dataset=dsd_tok["val"],
        tokenizer=tok,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate(dsd_tok["test"])
    log(f"Test metrics: {metrics}", "ok")

    # Save everything for local eval later
    model.config.id2label = id2label
    model.config.label2id = label2id
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    log(f"Saved transformer model → {out_dir}", "ok")

if __name__ == "__main__":
    p = tt.ArgumentParser()
    p.add_argument("--inp", required=True)
    p.add_argument("--base_model", default="dccuchile/bert-base-spanish-wwm-cased")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--max_len", type=int, default=256)
    a = p.parse_args()
    main(a.inp, a.base_model, a.out_dir, a.epochs, a.max_len)
