# spanish-fake-news-ai
Capstone project: detecting AI-generated and fake news in Spanish media


# Spanish Headline Classifier — REAL / FAKE / AI_GENERATED

**Pipeline:** collect → merge → split → train (baseline + transformer) → evaluate → report.

## Reproduce (quickstart)
```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python scripts/train_baseline.py --inp data/processed/dataset.csv --out models/baseline.joblib
python scripts/train_transformer.py --inp data/processed/dataset.csv --base_model dccuchile/bert-base-spanish-wwm-cased --epochs 3 --max_len 256 --out_dir models/beto-256-e3
python scripts/evaluate.py --baseline --inp data/processed/dataset.csv --model_path models/baseline.joblib
python scripts/evaluate.py --transformer --inp data/processed/dataset.csv --model_dir models/beto-256-e3

## Results (test)

| Model                  | Macro-F1 | REAL F1 | FAKE F1 | AI_GEN F1 |
|------------------------|---------:|--------:|--------:|----------:|
| Baseline (TF-IDF + LR) | 0.887    | 0.933   | 0.727   | 1.000     |
| BETO L=256 E=3         | 0.970    | 0.976   | 0.933   | 1.000     |
