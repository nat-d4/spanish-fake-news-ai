import argparse as ga
import os
import pandas as pd
import random
import uuid
from common import write_csv_safe, log, slugify

SEEDS = [
    "Gobierno anuncia nueva política económica",
    "Se registra apagón en varias provincias",
    "Investigan presunto caso de corrupción",
]

VARIANTS = [
    "Última hora: {base} según fuentes anónimas.",
    "En redes circula la noticia: {base}.",
    "Expertos cuestionan que {base} tenga fundamento.",
]

def gen_variant(title: str) -> str:
    return random.choice(VARIANTS).format(base=title)

def main(out: str, total: int = 600, seed: int | None = 42, label_style: str = "AI_GENERATED"):
    if seed is not None:
        random.seed(seed)

    # Normalize label spelling once here (keep consistent across your project!)
    if label_style not in {"AI_GENERATED", "AI-GENERATED"}:
        raise ValueError("label_style must be 'AI_GENERATED' or 'AI-GENERATED'.")

    rows = []
    i = 0
    n_seeds = len(SEEDS)
    while len(rows) < total:
        base = SEEDS[i % n_seeds]
        txt = gen_variant(base)
        rows.append({
            "id": f"{slugify(base)}-ai-{uuid.uuid4().hex[:8]}",
            "text": txt,
            "source": "synthetic",
            "label": label_style,
        })
        i += 1

    df_out = pd.DataFrame(rows)
    write_csv_safe(df_out, out)
    log(f"Wrote {len(df_out)} AI-GENERATED headlines to {out}", "ok")

if __name__ == "__main__":
    p = ga.ArgumentParser()
    p.add_argument("--out", default="data/raw/ai_generated.csv")
    # allow either name; both map to total number of rows
    p.add_argument("--limit", "--n", type=int, default=600, dest="total",
                   help="Total number of AI headlines to generate")
    p.add_argument("--seed", type=int, default=42, help="Random seed (set None to disable)")
    p.add_argument("--label_style", choices=["AI_GENERATED", "AI-GENERATED"], default="AI_GENERATED")
    a = p.parse_args()
    main(a.out, a.total, a.seed, a.label_style)