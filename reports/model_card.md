```markdown
# Model Card — BETO headline classifier

**Intended use:** Classify Spanish headlines as REAL / FAKE / AI_GENERATED.

**Training data:** data/processed/dataset.csv (70/15/15 split). Fake headline sources: ["https://maldita.es/rss/", "https://www.chequeado.com/rss/", "https://www.newtral.es/feed/"], Real headline sources: ["https://elpais.com/rss/elpais/portada.xml", "https://www.bbc.com/mundo/ultimas_noticias/index.xml", "https://www.eldiario.es/rss/", "https://www.infobae.com/america/rss/"], synthetic AI rows via `scripts/generate_ai.py`.

**Labels & mapping:** REAL=0, FAKE=1, AI_GENERATED=2 (stored in model config).

**Model:** `dccuchile/bert-base-spanish-wwm-cased` fine-tuned, max_len=256, epochs=3, lr=2e-5, batch=8, weight_decay=0.01, warmup_ratio=0.1.

**Metrics (test):** Macro-F1 = 0.970. See `reports/metrics_transformer.txt` and confusion PNG.

**Limitations:** Short text; domain shift across sources; synthetic AI text may not reflect real LLM output; Spanish regional variation.

**Risks & mitigations:** Don’t deploy as sole fact-checking; include human review; monitor performance by source/time; retrain periodically.