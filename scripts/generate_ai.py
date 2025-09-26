import argparse as ga
import os
import pandas as pd
import random
from common import write_csv_safe, log, slugify

SEEDS = [
    "Gobierno anuncia nueva política económica",
    "Se registra apagón en varias provincias",
    "Investigan presunto caso de corrupción",
]

VARIANTS = [
    "Última hora: {base} segun fuentes anónimas.",
    "En redes circula la noticia: {base}.",
    "Expertos cuestionan que {base} tenga fundamento.",
]

def gen_variants(title: str, n: int = 2) -> list[str]:
    out = []
    for _ in range(n):
        out.append(random.choice(VARIANTS).format(base = title))
    return out

def main(out: str, per_item: int = 2, limit: int = 600):
    titles = SEEDS[: limit // per_item]
    rows = []
    for t in titles:
        for txt in gen_variants(t, n = per_item):
            rows.append({
                "id": slugify(t) + f"-ai-{random.randint(1000, 9999)}",
                "text": txt,
                "source": "synthetic",
                "label": "AI-GENERATED",
            })
    
    df_out = pd.DataFrame(rows)
    write_csv_safe(df_out, out)
    log(f"Wrote {len(df_out)} AI-GENERATED healines to {out}", "ok")

if __name__ == "__main__":
    p = ga.ArgumentParser()
    p.add_argument("--out", default = "data/raw/ai.csv")
    p.add_argument("--per_item", type = int, default = 2)
    p.add_argument("--limit", type = int, default = 600)
    a = p.parse_args()
    main(a.out, a.per_item, a.limit)