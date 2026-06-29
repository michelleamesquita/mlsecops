"""
scripts/membership_inference.py — Membership Inference Attack via IBM ART (genérico)
Avalia vazamento de privacidade em QUALQUER classificador sklearn serializado.

MITRE ATLAS: AML.T0056 Membership Inference
OWASP MLSVS: privacidade dos dados de treinamento

Uso:
    python scripts/membership_inference.py \
        --data dataset.csv --target label \
        --model model/rf_model.pkl \
        --max-advantage 0.10
"""

import argparse
import json
import logging
import pathlib
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def load_meta(meta_path: str | None) -> dict | None:
    if not meta_path or not pathlib.Path(meta_path).exists():
        return None
    with open(meta_path) as f:
        return json.load(f)


def auto_build_features(df: pd.DataFrame, target: str, meta: dict | None) -> tuple[np.ndarray, np.ndarray]:
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
        exclude = {target}
        for col in df.select_dtypes(include="object").columns:
            if col not in exclude:
                df[col] = df[col].fillna("UNKNOWN").astype("category").cat.codes
        for col in df.columns:
            if col != target:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        feature_cols = [c for c in df.columns if c != target and df[c].dtype != object]

    X = df[feature_cols].values.astype(np.float32)
    y = pd.to_numeric(df[target], errors="coerce").fillna(0).astype(int).values
    return X, y


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Membership inference attack via IBM ART (generic sklearn)"
    )
    parser.add_argument("--data",          required=True)
    parser.add_argument("--target",        required=True)
    parser.add_argument("--model",         required=True)
    parser.add_argument("--meta",          default=None)
    parser.add_argument("--max-advantage", type=float, default=0.10,
                        help="Max MI advantage (default: 0.10 = 10%%). Calibre com baseline do domínio.")
    parser.add_argument("--sample",        type=int, default=60_000)
    parser.add_argument("--n-attack",      type=int, default=1000)
    parser.add_argument("--output-dir",    default="results")
    args = parser.parse_args()

    try:
        from art.estimators.classification import SklearnClassifier
        from art.attacks.inference.membership_inference import MembershipInferenceBlackBox
    except ImportError:
        log.error("adversarial-robustness-toolbox not installed.")
        sys.exit(1)

    for p in [args.model, args.data]:
        if not pathlib.Path(p).exists():
            log.error(f"File not found: {p}")
            sys.exit(1)

    clf = joblib.load(args.model)
    meta = load_meta(args.meta)

    log.info(f"Loading data ({args.sample:,} rows)…")
    df = pd.read_csv(args.data, low_memory=False, nrows=args.sample) \
         if pathlib.Path(args.data).suffix == ".csv" else pd.read_parquet(args.data)

    X_all, y_all = auto_build_features(df, args.target, meta)

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.3, random_state=42, stratify=y_all
    )
    n = min(args.n_attack, len(X_train), len(X_test))
    half = n // 2

    x_min, x_max = X_all.min(axis=0), X_all.max(axis=0)
    art_clf = SklearnClassifier(model=clf, clip_values=(x_min, x_max))

    log.info("Running MembershipInferenceBlackBox…")
    mia = MembershipInferenceBlackBox(art_clf, attack_model_type="rf")
    mia.fit(X_train[:half], y_train[:half], X_test[:half], y_test[:half])

    inferred_train = mia.infer(X_train[half:n], y_train[half:n])
    inferred_test  = mia.infer(X_test[half:n],  y_test[half:n])

    attack_acc = (inferred_train.sum() + (1 - inferred_test).sum()) / (len(inferred_train) + len(inferred_test))
    advantage  = float(attack_acc - 0.5)

    log.info(f"\n=== Membership Inference Gate ===")
    log.info(f"  TPR (train → member) : {inferred_train.mean():.4f}")
    log.info(f"  FPR (test  → member) : {inferred_test.mean():.4f}")
    log.info(f"  Attack accuracy      : {attack_acc:.4f}")
    log.info(f"  Advantage            : {advantage:.4f} ({advantage*100:.1f}%)")
    log.info(f"  Max advantage        : {args.max_advantage:.4f} ({args.max_advantage*100:.1f}%)")
    log.info(f"  NOTE: threshold empírico — calibre com baseline do domínio.")
    log.info(f"  MITRE ATLAS AML.T0056 · OWASP MLSVS")

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "membership_inference.json", "w") as f:
        json.dump({
            "tpr": float(inferred_train.mean()),
            "fpr": float(inferred_test.mean()),
            "attack_accuracy": float(attack_acc),
            "advantage": advantage,
            "max_advantage_threshold": args.max_advantage,
            "passed": advantage <= args.max_advantage,
        }, f, indent=2)

    if advantage > args.max_advantage:
        log.error(f"MI Gate FAILED — advantage={advantage:.4f} > {args.max_advantage:.4f}")
        log.error("Mitigações: differential privacy (DP-SGD), regularização, data minimization.")
        sys.exit(1)

    log.info("Membership Inference Gate: PASSED")


if __name__ == "__main__":
    main()
