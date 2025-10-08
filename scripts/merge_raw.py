import pandas as pd
import argparse

p = argparse.ArgumentParser()
p.add_argument("--real")
p.add_argument("--fake")
p.add_argument("--ai")
p.add_argument("--out", required=True)
a = p.parse_args()

dfs = []
if a.real: dfs.append(pd.read_csv(a.real).assign(label="REAL"))
if a.fake: dfs.append(pd.read_csv(a.fake).assign(label="FAKE"))
if a.ai:   dfs.append(pd.read_csv(a.ai).assign(label="AI_GENERATED"))

df = pd.concat(dfs, ignore_index=True)
df.to_csv(a.out, index=False)
print(f"[OK] Wrote {len(df)} rows to {a.out}")