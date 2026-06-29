"""
scripts/model_extraction_test.py — Teste de extração de modelo via Knockoff (genérico)
Avalia se o modelo é vulnerável a ser reconstituído por um adversário via queries.

Um atacante que consulta o modelo repetidamente pode treinar um modelo "knockoff"
com alta fidelidade, efetivamente roubando as propriedades aprendidas do modelo
sem acesso ao código ou dados de treino.

MITRE ATLAS: AML.T0044 Full ML Model Access → Extract ML Model
AISP Module 2: Adversarial Robustness Testing
OWASP MLSVS V5 — Model robustness

Métricas:
  - Fidelity   : acordo entre knockoff e original (ideal < 0.90 para defender)
  - Accuracy   : acurácia do knockoff nos dados de teste (ideal < acc_original)
  - Info leakage: quantos bits o modelo vaza por query (baseado na fidelidade)

Uso:
    python scripts/model_extraction_test.py \
        --data dataset.csv --target label \
        --model model/rf_model.pkl \
        --n-queries 5000 \
        --max-fidelity 0.90
"""

import argparse
import json
import logging
import pathlib
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def load_meta(meta_path: str | None) -> dict | None:
    if not meta_path or not pathlib.Path(meta_path).exists():
        return None
    with open(meta_path) as f:
        return json.load(f)


def build_features(df: pd.DataFrame, target: str, meta: dict | None) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.preprocessing import LabelEncoder
    df = df.copy()
    if meta:
        for col, classes in meta.get("encoder_classes", {}).items():
            if col in df.columns:
                le = LabelEncoder()
                le.classes_ = np.array(classes)
                df[col] = df[col].fillna("UNKNOWN").astype(str)
                known = set(le.classes_)
                df[col] = df[col].apply(lambda x: x if x in known else le.classes_[0])
                df[col] = le.transform(df[col])
        for col in meta.get("num_features", []):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        feature_cols = [c for c in meta["feature_names"] if c in df.columns]
    else:
        for col in df.select_dtypes(include="object").columns:
            if col != target:
                df[col] = df[col].fillna("UNKNOWN").astype("category").cat.codes
        for col in df.columns:
            if col != target:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        feature_cols = [c for c in df.columns if c != target and df[c].dtype != object]
    X = df[feature_cols].values.astype(np.float32)
    y = pd.to_numeric(df[target], errors="coerce").fillna(0).astype(int).values
    return X, y


def generate_query_set(X_pool: np.ndarray, n_queries: int,
                        seed: int, strategy: str = "random") -> np.ndarray:
    """
    Gera o conjunto de queries para o ataque de extração.
    strategy='random'   : amostras aleatórias do pool
    strategy='uniform'  : perturbações uniformes no espaço de features
    """
    rng = np.random.RandomState(seed)
    if strategy == "random":
        idx = rng.choice(len(X_pool), size=min(n_queries, len(X_pool)), replace=False)
        return X_pool[idx]
    else:
        # Uniform sampling no range de cada feature
        x_min = X_pool.min(axis=0)
        x_max = X_pool.max(axis=0)
        return rng.uniform(x_min, x_max, size=(n_queries, X_pool.shape[1])).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Model extraction vulnerability test via Knockoff (AML.T0044)"
    )
    parser.add_argument("--data",          required=True)
    parser.add_argument("--target",        required=True)
    parser.add_argument("--model",         required=True)
    parser.add_argument("--meta",          default=None)
    parser.add_argument("--n-queries",     type=int, default=5000,
                        help="Número de queries para treinar o knockoff (default: 5000)")
    parser.add_argument("--max-fidelity",  type=float, default=0.90,
                        help="Fidelidade máxima tolerada (default: 0.90). Acima = vulnerável.")
    parser.add_argument("--query-strategy", default="random", choices=["random", "uniform"],
                        help="Estratégia de query: random (usa pool) ou uniform (espaço completo)")
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument("--sample",        type=int, default=50_000)
    parser.add_argument("--output-dir",    default="results")
    args = parser.parse_args()

    for p in [args.model, args.data]:
        if not pathlib.Path(p).exists():
            log.error(f"File not found: {p}")
            sys.exit(1)

    log.info(f"Loading target model (oracle): {args.model}")
    oracle = joblib.load(args.model)
    meta = load_meta(args.meta)

    log.info(f"Loading data ({args.sample:,} rows)…")
    p = pathlib.Path(args.data)
    df = pd.read_csv(p, low_memory=False, nrows=args.sample) \
         if p.suffix == ".csv" else pd.read_parquet(p)

    X_all, y_all = build_features(df, args.target, meta)
    X_train_pool, X_test, y_train_pool, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=args.seed, stratify=y_all
    )

    # ── 1. Oracle accuracy (baseline) ────────────────────────────────────
    acc_oracle = (oracle.predict(X_test) == y_test).mean()
    log.info(f"  Oracle accuracy on test set: {acc_oracle:.4f}")

    # ── 2. Geração do query set ───────────────────────────────────────────
    log.info(f"\n  Generating {args.n_queries} queries (strategy={args.query_strategy})…")
    X_query = generate_query_set(X_train_pool, args.n_queries, args.seed, args.query_strategy)

    # ── 3. Oracle labels via predição (simula acesso black-box) ──────────
    log.info("  Querying oracle (simulating black-box access)…")
    y_query = oracle.predict(X_query)
    log.info(f"  Queries returned: {len(y_query)} labels (positive rate: {y_query.mean():.4f})")

    # ── 4. Treino do knockoff model ───────────────────────────────────────
    log.info("  Training knockoff model on oracle responses…")
    knockoff = RandomForestClassifier(
        n_estimators=100, max_depth=15, class_weight="balanced",
        random_state=args.seed, n_jobs=-1
    )
    knockoff.fit(X_query, y_query)
    log.info("  Knockoff model trained.")

    # ── 5. Avaliação ──────────────────────────────────────────────────────
    oracle_preds   = oracle.predict(X_test)
    knockoff_preds = knockoff.predict(X_test)

    fidelity    = float((oracle_preds == knockoff_preds).mean())
    acc_knockoff = float((knockoff_preds == y_test).mean())

    # Information leakage proxy: quanto o knockoff capturou do oracle
    # Fidelidade de 1.0 = modelo completamente extraído
    # Fidelidade de 0.5 = aleatório (nenhuma extração)
    info_leakage = max(0.0, (fidelity - 0.5) / 0.5)  # normalizado [0, 1]

    log.info(f"\n=== Model Extraction Gate (AML.T0044) ===")
    log.info(f"  Queries used             : {args.n_queries}")
    log.info(f"  Oracle accuracy          : {acc_oracle:.4f}")
    log.info(f"  Knockoff accuracy        : {acc_knockoff:.4f}")
    log.info(f"  Fidelity (agreement)     : {fidelity:.4f}  (threshold: {args.max_fidelity})")
    log.info(f"  Info leakage proxy       : {info_leakage:.4f}  (0=safe, 1=fully extracted)")
    log.info(f"  NOTE: threshold empírico — calibre com baseline do domínio.")
    log.info(f"  MITRE ATLAS AML.T0044")

    if fidelity > args.max_fidelity:
        log.warning(f"  HIGH FIDELITY — modelo vulnerável a extração!")
        log.warning(f"  Mitigações: prediction throttling, output rounding, confidence truncation,")
        log.warning(f"  differential privacy, rate limiting na API de inferência.")

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "passed": fidelity <= args.max_fidelity,
        "n_queries": args.n_queries,
        "query_strategy": args.query_strategy,
        "acc_oracle": float(acc_oracle),
        "acc_knockoff": float(acc_knockoff),
        "fidelity": fidelity,
        "info_leakage_proxy": info_leakage,
        "max_fidelity_threshold": args.max_fidelity,
        "atlas": "AML.T0044",
    }
    with open(out_dir / "model_extraction_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # Gate: apenas WARNING por padrão (RF local = não há endpoint exposto)
    # Torne bloqueante em pipelines com serving de modelo via API
    if not report["passed"]:
        log.warning("  Model Extraction Gate: WARNING (fidelidade alta)")
        log.warning("  Em ambientes de serving: considere este gate como BLOQUEANTE.")
    else:
        log.info("  Model Extraction Gate: PASSED")


if __name__ == "__main__":
    main()
