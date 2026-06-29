"""
scripts/label_noise_check.py — Detecção de label noise via Cleanlab (genérico)
Funciona com qualquer dataset tabular de classificação (binário ou multiclasse).
Auto-detecta features numéricas e categóricas.

MITRE ATLAS: AML.T0020 Label Flipping / Clean-label Poisoning

Uso:
    python scripts/label_noise_check.py \
        --data dataset.csv --target label --threshold 0.05

    # Features explícitas:
    python scripts/label_noise_check.py \
        --data dataset.csv --target label \
        --cat-features severity confidence --threshold 0.05
"""

import argparse
import json
import logging
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from cleanlab.filter import find_label_issues

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def auto_encode(df: pd.DataFrame, target: str,
                cat_features: list[str] | None) -> tuple[np.ndarray, np.ndarray]:
    """Encoda features automaticamente e retorna (X, y)."""
    exclude = {target}
    X_df = df.drop(columns=list(exclude), errors="ignore").copy()

    # Se cat_features não fornecido, auto-detecta
    if cat_features is None:
        cat_features = [
            c for c in X_df.columns
            if X_df[c].dtype == object and X_df[c].nunique() < 5000
        ]

    for col in cat_features:
        if col in X_df.columns:
            X_df[col] = X_df[col].fillna("UNKNOWN").astype("category").cat.codes

    for col in X_df.columns:
        X_df[col] = pd.to_numeric(X_df[col], errors="coerce").fillna(0)

    # Exclui colunas com variância zero
    X_df = X_df.loc[:, X_df.std() > 0]

    # Encoda target para inteiros contíguos
    y_raw = df[target].fillna("UNKNOWN").astype(str)
    classes, y = np.unique(y_raw.values, return_inverse=True)
    log.info(f"  Classes: {classes.tolist()}")

    return X_df.values.astype(np.float32), y


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cleanlab label noise detection (generic classification)"
    )
    parser.add_argument("--data",         required=True)
    parser.add_argument("--target",       required=True, help="Target column name")
    parser.add_argument("--threshold",    type=float, default=0.05,
                        help="Max label noise rate (default: 0.05). Calibre com baseline do domínio.")
    parser.add_argument("--sample",       type=int, default=50_000)
    parser.add_argument("--cv-folds",     type=int, default=3)
    parser.add_argument("--cat-features", nargs="*", default=None,
                        help="Categorical feature names (auto-detected if omitted)")
    parser.add_argument("--output-dir",   default="results")
    parser.add_argument("--fail-on-noise", action="store_true",
                        help="Exit 1 if noise rate exceeds threshold (default: warn only)")
    args = parser.parse_args()

    path = pathlib.Path(args.data)
    if not path.exists():
        log.error(f"Dataset not found: {path}")
        sys.exit(1)

    log.info(f"Loading {args.sample:,} rows from {path}")
    df = pd.read_csv(path, low_memory=False, nrows=args.sample) \
         if path.suffix == ".csv" else pd.read_parquet(path)
    df = df.dropna(subset=[args.target]).reset_index(drop=True)

    if args.target not in df.columns:
        log.error(f"Target '{args.target}' not found. Available: {list(df.columns)}")
        sys.exit(1)

    log.info(f"  Shape: {df.shape} | target='{args.target}'")
    X, y = auto_encode(df, args.target, args.cat_features)
    log.info(f"  Features: {X.shape[1]} | Samples: {len(y)}")

    log.info(f"  Computing {args.cv_folds}-fold OOF probabilities…")
    clf = RandomForestClassifier(n_estimators=100, max_depth=10,
                                  class_weight="balanced", random_state=42, n_jobs=-1)
    pred_probs = cross_val_predict(clf, X, y, cv=args.cv_folds, method="predict_proba")

    log.info("  Running Cleanlab find_label_issues…")
    issues_idx = find_label_issues(
        labels=y,
        pred_probs=pred_probs,
        return_indices_ranked_by="self_confidence",
    )

    noise_rate = len(issues_idx) / len(y)
    log.info(f"\n=== Label Noise Gate ===")
    log.info(f"  Suspicious issues : {len(issues_idx):,}")
    log.info(f"  Noise rate        : {noise_rate:.4f} ({noise_rate*100:.2f}%)")
    log.info(f"  Threshold         : {args.threshold:.4f} ({args.threshold*100:.1f}%)")
    log.info(f"  NOTE: threshold empírico — calibre com baseline do domínio antes de tornar bloqueante.")

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "label_noise_report.json", "w") as f:
        json.dump({
            "passed": noise_rate <= args.threshold,
            "noise_rate": noise_rate,
            "n_issues": len(issues_idx),
            "threshold": args.threshold,
        }, f, indent=2)

    if noise_rate > args.threshold:
        log.error(f"  Label Noise Gate FAILED — {noise_rate*100:.2f}% > {args.threshold*100:.1f}%")
        log.info("  NOTE: thresholds empíricos — calibre com baseline do domínio antes de tornar bloqueante.")
        log.info("  MITRE ATLAS AML.T0020 · Clean-label poisoning")
        if args.fail_on_noise:
            sys.exit(1)
        else:
            log.warning("  Continuando (--fail-on-noise não ativo). Adicione --fail-on-noise para bloquear.")
        return

    log.info("  Label Noise Gate: PASSED")


if __name__ == "__main__":
    main()
