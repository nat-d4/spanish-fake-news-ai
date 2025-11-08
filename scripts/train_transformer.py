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

import torch
from torch.nn import CrossEntropyLoss

LABELS = ["REAL", "FAKE", "AI_GENERATED"]


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, (list, tuple)):
        logits = logits[0]
    preds = np.argmax(logits, axis=1)
    p, r, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    return {"macro_f1": f1, "precision": p, "recall": r}


class WeightedTrainer(Trainer):
    """Trainer with class-weighted CrossEntropyLoss (HF v4/v5 compatible)."""
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    # accept extra kwargs like num_items_in_batch from newer Trainer
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(
            logits.view(-1, self.model.config.num_labels),
            labels.view(-1),
        )
        return (loss, outputs) if return_outputs else loss


def main(inp: str, base_model: str, out_dir: str, epochs: int, max_len: int):
    # --- Load & basic filter ---
    df = pd.read_csv(inp)
    df = df[df["label"].isin(LABELS)].reset_index(drop=True)

    # --- Current TRAIN slice ---
    train_df = df[df["split"] == "train"]

    # --- Targeted FAKE supplements (train-only) ---
    FAKE_SUPP = [
        "El mundo termina mañana, científicos lo confirman",
        "Cura milagrosa elimina todas las enfermedades en 24 horas",
        "Gobierno admite que la gravedad fue abolida por decreto",
        "Beber agua con limón cura el cáncer según 'expertos'",
        "Vacuna convierte a las personas en antenas 5G",
        "Descubren ciudad perdida en Marte con vida humana",
        "El Sol se apagará la próxima semana, recomienda abrigarse",
        "Nuevo impuesto por respirar entra en vigor este lunes",
        "Terremoto de magnitud 15 previsto con precisión exacta",
        "Comer chocolate adelgaza 10 kilos en tres días",
        "Predicen que un asteroide chocará mañana a las 9:00",
        "Autoridades confirman que la Tierra es plana oficialmente",
        "Bebida mágica garantiza felicidad eterna, 100% real",
        "Millonario secreto regala 1.000€ a quien comparta esta noticia",
        "No habrá noche durante un mes por alineación de planetas",
        "Nueva ley prohíbe pensar en negativo y multa a ciudadanos",
        "NASA oculta pruebas de civilizaciones en el centro de la Tierra",
        "El océano Atlántico se evaporará por completo este año",
        "Robot desarrolla sentimientos y se postula a presidente",
        "Los gatos hablan español si les das leche de almendras",
        "El mundo termina mañana",
    ]
    fake_supp_df = pd.DataFrame({"text": FAKE_SUPP, "label": "FAKE", "split": "train"})
    train_df = pd.concat([train_df, fake_supp_df], ignore_index=True)

    # --- Replace original TRAIN with augmented TRAIN ---
    df = pd.concat([train_df, df[df["split"] != "train"]], ignore_index=True)

    # --- Strong oversampling in TRAIN ONLY to fight imbalance ---
    train_df = df[df["split"] == "train"]
    fake_df = train_df[train_df["label"] == "FAKE"]
    real_df = train_df[train_df["label"] == "REAL"]
    df = pd.concat([df, *([fake_df] * 6), *([real_df] * 2)], ignore_index=True)

    print("New train distribution:")
    print(df[df["split"] == "train"]["label"].value_counts())

    # --- Label maps ---
    label2id = {l: i for i, l in enumerate(LABELS)}
    id2label = {i: l for l, i in label2id.items()}

    # --- Build HF datasets (supports 'val' or 'dev') ---
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
        raise ValueError("Expected splits train/val/test (or dev) missing in dataset.")

    dsd = DatasetDict({"train": train_ds, "val": val_ds, "test": test_ds})

    tok = AutoTokenizer.from_pretrained(base_model)

    def tokenize(batch):
        return tok(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_len,
        )

    dsd_tok = dsd.map(tokenize, batched=True, remove_columns=["text"])

    # --- Model ---
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=len(LABELS),
        id2label=id2label,
        label2id=label2id,
    )

    ensure_dir(out_dir)

    # --- TrainingArguments with fallback for older transformers versions ---
    base_kwargs = dict(
        output_dir=out_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=50,
        dataloader_pin_memory=False,  # silence CPU warning
    )

    try:
        args = TrainingArguments(
            **base_kwargs,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            greater_is_better=True,
            warmup_ratio=0.1,
            report_to=[],          # disable W&B, etc.
            save_total_limit=2,
        )
    except TypeError:
        args = TrainingArguments(**base_kwargs)

    # --- Manual class weights (order: REAL, FAKE, AI_GENERATED) ---
    class_weights = torch.tensor([1.0, 6.0, 1.5], dtype=torch.float)
    print("Using manual class weights (REAL, FAKE, AI_GENERATED):", class_weights.tolist())

    # --- Trainer (weighted) with tokenizer/processing compatibility ---
    try:
        trainer = WeightedTrainer(
            model=model,
            args=args,
            train_dataset=dsd_tok["train"],
            eval_dataset=dsd_tok["val"],
            processing_class=tok,   # new name in v5
            compute_metrics=compute_metrics,
            class_weights=class_weights,
        )
    except TypeError:
        trainer = WeightedTrainer(
            model=model,
            args=args,
            train_dataset=dsd_tok["train"],
            eval_dataset=dsd_tok["val"],
            tokenizer=tok,          # older versions expect `tokenizer`
            compute_metrics=compute_metrics,
            class_weights=class_weights,
        )

    # --- Train & evaluate ---
    trainer.train()
    metrics = trainer.evaluate(dsd_tok["test"])
    log(f"Test metrics: {metrics}", "ok")

    # --- Save for inference ---
    model.config.id2label = id2label
    model.config.label2id = label2id
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    log(f"Saved transformer model and tokenizer → {out_dir}", "ok")


if __name__ == "__main__":
    p = tt.ArgumentParser()
    p.add_argument("--inp", required=True)
    p.add_argument("--base_model", default="dccuchile/bert-base-spanish-wwm-cased")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--epochs", type=int, default=4)      # bump to 4 as discussed
    p.add_argument("--max_len", type=int, default=256)
    a = p.parse_args()
    main(a.inp, a.base_model, a.out_dir, a.epochs, a.max_len)