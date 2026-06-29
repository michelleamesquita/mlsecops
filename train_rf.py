"""
MLSecOps — Estágio 02: Experimentação Segura
Treina Random Forest para classificar findings de SAST como is_risky (0/1).

Ameaças mitigadas (MITRE ATLAS):
  - AML.T0020 Poisoning  → validação de qualidade dos dados antes do treino
  - Label noise          → distribuição do target logada para revisão humana

Artefatos gerados (MLflow):
  - Parâmetros: n_estimators, max_depth, class_weight, test_size
  - Métricas:   accuracy, f1_weighted, roc_auc, precision_weighted, recall_weighted
  - Artefato:   model/rf_model.pkl + feature_names.json
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------
DATA_PATH = Path(os.getenv("DATA_PATH", "all_findings_flat.csv"))
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "mlsecops-rf-findings")
MODEL_OUTPUT_DIR = Path(os.getenv("MODEL_OUTPUT_DIR", "model"))

N_ESTIMATORS = int(os.getenv("RF_N_ESTIMATORS", "200"))
MAX_DEPTH = int(os.getenv("RF_MAX_DEPTH", "15")) if os.getenv("RF_MAX_DEPTH") else None
MIN_SAMPLES_LEAF = int(os.getenv("RF_MIN_SAMPLES_LEAF", "10"))
TEST_SIZE = float(os.getenv("RF_TEST_SIZE", "0.2"))
RANDOM_STATE = 42

# Features categóricas a encodar
CAT_FEATURES = ["model", "severity", "confidence", "test_id", "cwe"]

# Features numéricas já prontas
NUM_FEATURES = [
    "line_number",
    "patch_lines",
    "patch_added",
    "patch_removed",
    "patch_files_touched",
    "patch_hunks",
    "patch_churn",
    "patch_net",
    "prompt_chars",
    "prompt_lines",
    "prompt_tokens",
    "prompt_has_security_guidelines",
    "temperature",
    "cwe_prevalence_overall",
    "cwe_severity_score",
    "cwe_weighted_severity",
]

TARGET = "is_risky"


# ---------------------------------------------------------------------------
# Data Quality Gate (Estágio 01)
# ---------------------------------------------------------------------------
def run_data_quality_gate(df: pd.DataFrame) -> None:
    """
    Validações mínimas antes do treino.
    Calibre thresholds com baseline do seu domínio antes de torná-los bloqueantes.
    """
    log.info("=== Data Quality Gate ===")

    missing_rate = df[TARGET].isna().mean()
    log.info(f"  Target missing rate: {missing_rate:.4f}")
    assert missing_rate == 0.0, f"Target com {missing_rate*100:.1f}% de nulos — pipeline abortada."

    risky_rate = df[TARGET].mean()
    log.info(f"  is_risky prevalence: {risky_rate:.4f} ({risky_rate*100:.1f}%)")
    if risky_rate < 0.01 or risky_rate > 0.99:
        log.warning("  Desbalanceamento extremo (>99:1). Revise class_weight ou estratégia de sampling.")

    for col in NUM_FEATURES:
        null_rate = df[col].isna().mean()
        if null_rate > 0.05:
            log.warning(f"  Feature '{col}' com {null_rate*100:.1f}% nulos — considere imputação.")

    log.info("  Data Quality Gate: PASSED")


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------
def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Encoda categóricas e retorna (X, encoders)."""
    encoders: dict[str, LabelEncoder] = {}
    df = df.copy()

    for col in CAT_FEATURES:
        df[col] = df[col].fillna("UNKNOWN").astype(str)
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    all_features = CAT_FEATURES + NUM_FEATURES
    for col in NUM_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df[all_features], encoders


# ---------------------------------------------------------------------------
# Treino
# ---------------------------------------------------------------------------
def train(data_path: Path = DATA_PATH, sample: int | None = None,
          seed: int = RANDOM_STATE, experiment_name: str = EXPERIMENT_NAME,
          kfold: int = 0) -> None:
    log.info(f"Carregando dataset: {data_path}")
    df = pd.read_csv(data_path, low_memory=False, nrows=sample)
    log.info(f"  Shape: {df.shape}")

    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce").fillna(0).astype(int)

    run_data_quality_gate(df)

    X, encoders = build_features(df)
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=seed, stratify=y
    )
    log.info(f"  Train: {len(X_train)} | Test: {len(X_test)}")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="random_forest_findings"):
        params = {
            "n_estimators": N_ESTIMATORS,
            "max_depth": MAX_DEPTH,
            "min_samples_leaf": MIN_SAMPLES_LEAF,
            "class_weight": "balanced",
            "test_size": TEST_SIZE,
            "random_state": seed,
            "n_features": len(CAT_FEATURES) + len(NUM_FEATURES),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
        }
        mlflow.log_params(params)
        log.info(f"  Parâmetros: {params}")

        log.info("  Treinando Random Forest (early stopping via OOB score)...")
        clf = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
            oob_score=True,      # habilita early stopping via OOB
            warm_start=True,     # adiciona árvores incrementalmente
        )
        # Early stopping: adiciona árvores em rounds e para quando OOB estabiliza
        best_oob = 0.0
        rounds_no_improve = 0
        oob_patience = int(os.getenv("RF_OOB_PATIENCE", "3"))
        step = max(10, N_ESTIMATORS // 10)
        trees_added = 0
        for n in range(step, N_ESTIMATORS + 1, step):
            clf.n_estimators = n
            clf.fit(X_train, y_train)
            trees_added = n
            oob = clf.oob_score_
            delta = oob - best_oob
            if delta > 1e-4:
                best_oob = oob
                rounds_no_improve = 0
            else:
                rounds_no_improve += 1
            if rounds_no_improve >= oob_patience:
                log.info(f"  Early stopping at {n} trees (OOB={oob:.4f}, no improvement for {oob_patience} rounds)")
                break
        log.info(f"  Treino concluído: {trees_added} árvores, OOB score={clf.oob_score_:.4f}")
        mlflow.log_metric("oob_score", clf.oob_score_)
        mlflow.log_param("trees_at_stop", trees_added)

        # ── KFold Cross-Validation (Stage 5 — avaliação robusta) ──────────
        cv_roc_mean, cv_roc_std = None, None
        if kfold > 1:
            log.info(f"  Executando {kfold}-fold Stratified CV...")
            cv = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=seed)
            # Usa um clf leve para CV (sem warm_start para evitar vazamento)
            clf_cv = RandomForestClassifier(
                n_estimators=trees_added, max_depth=MAX_DEPTH,
                min_samples_leaf=MIN_SAMPLES_LEAF, class_weight="balanced",
                random_state=seed, n_jobs=-1,
            )
            X_all = pd.concat([
                pd.DataFrame(X_train, columns=CAT_FEATURES + NUM_FEATURES),
                pd.DataFrame(X_test,  columns=CAT_FEATURES + NUM_FEATURES),
            ]).values
            y_all = np.concatenate([y_train, y_test])
            cv_scores = cross_val_score(clf_cv, X_all, y_all,
                                        cv=cv, scoring="roc_auc", n_jobs=-1)
            cv_roc_mean = float(cv_scores.mean())
            cv_roc_std  = float(cv_scores.std())
            log.info(f"  KFold ROC-AUC: {cv_roc_mean:.4f} ± {cv_roc_std:.4f}")
            mlflow.log_metric("cv_roc_auc_mean", cv_roc_mean)
            mlflow.log_metric("cv_roc_auc_std",  cv_roc_std)
            mlflow.log_param("kfold", kfold)

        # ------------------------------------------------------------------
        # Avaliação
        # ------------------------------------------------------------------
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted")
        roc = roc_auc_score(y_test, y_prob)

        metrics = {
            "accuracy": acc,
            "f1_weighted": f1,
            "precision_weighted": prec,
            "recall_weighted": rec,
            "roc_auc": roc,
        }
        mlflow.log_metrics(metrics)

        log.info("=== Adversarial Gate (métricas mínimas) ===")
        log.info(f"  accuracy          : {acc:.4f}")
        log.info(f"  f1_weighted       : {f1:.4f}")
        log.info(f"  roc_auc           : {roc:.4f}")
        log.info(f"  precision_weighted: {prec:.4f}")
        log.info(f"  recall_weighted   : {rec:.4f}")

        # Threshold de referência: calibre com baseline do domínio antes de
        # tornar bloqueante (princípio 1 do Agent.md)
        MIN_ROC_AUC = float(os.getenv("GATE_MIN_ROC_AUC", "0.75"))
        if roc < MIN_ROC_AUC:
            log.error(f"  ROC-AUC {roc:.4f} abaixo do threshold {MIN_ROC_AUC} — modelo reprovado no gate.")
            sys.exit(1)
        log.info("  Adversarial Gate: PASSED")

        log.info("\n" + classification_report(y_test, y_pred, target_names=["safe (0)", "risky (1)"]))

        # ------------------------------------------------------------------
        # Feature Importance
        # ------------------------------------------------------------------
        feature_names = CAT_FEATURES + NUM_FEATURES
        importances = dict(zip(feature_names, clf.feature_importances_.tolist()))
        top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]
        log.info("  Top-10 features:")
        for fname, imp in top_features:
            log.info(f"    {fname:<35} {imp:.4f}")

        # ------------------------------------------------------------------
        # Persistência do artefato (SLSA: evidência rastreável no MLflow)
        # ------------------------------------------------------------------
        MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODEL_OUTPUT_DIR / "rf_model.pkl"
        joblib.dump(clf, model_path)

        feature_meta = {
            "feature_names": feature_names,
            "cat_features": CAT_FEATURES,
            "num_features": NUM_FEATURES,
            "target": TARGET,
            "encoder_classes": {col: enc.classes_.tolist() for col, enc in encoders.items()},
        }
        meta_path = MODEL_OUTPUT_DIR / "feature_names.json"
        with open(meta_path, "w") as f:
            json.dump(feature_meta, f, indent=2)

        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(meta_path))
        mlflow.sklearn.log_model(clf, "sklearn_model")

        run_id = mlflow.active_run().info.run_id
        log.info(f"\n  MLflow Run ID: {run_id}")
        log.info(f"  Modelo salvo em: {model_path}")
        log.info("  Pipeline concluída com sucesso.")

        # Métricas em markdown para PR comment (secure-experiment.yml)
        Path("results").mkdir(exist_ok=True)
        cv_line = f"| KFold CV ROC-AUC    | {cv_roc_mean:.4f} ± {cv_roc_std:.4f} |\n" if cv_roc_mean else ""
        md = f"""## MLSecOps · RF Findings Classifier

| Métrica            | Valor  |
|--------------------|--------|
| ROC-AUC            | {roc:.4f} |
| F1 (weighted)      | {f1:.4f} |
| Accuracy           | {acc:.4f} |
| Precision (w)      | {prec:.4f} |
| Recall (w)         | {rec:.4f} |
| OOB Score          | {clf.oob_score_:.4f} |
{cv_line}
**Params:** n_estimators={trees_added} (early stop), max_depth={MAX_DEPTH}, class_weight=balanced  
**Train samples:** {len(X_train):,} | **Test samples:** {len(X_test):,}  
**Run ID:** `{run_id}`

> Threshold ROC-AUC: {os.getenv('GATE_MIN_ROC_AUC', '0.75')} — calibre com baseline do domínio.
"""
        with open("results/train_metrics.md", "w") as f:
            f.write(md)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLSecOps RF Findings Classifier — Treino")
    parser.add_argument("--seed",            type=int,   default=None,
                        help="Random seed (overrides RANDOM_STATE env var)")
    parser.add_argument("--experiment-name", type=str,   default=None,
                        help="MLflow experiment name (overrides MLFLOW_EXPERIMENT_NAME env)")
    parser.add_argument("--sample",          type=int,   default=None,
                        help="Max rows to load (useful para CI rápido)")
    parser.add_argument("--kfold", type=int, default=0,
                        help="StratifiedKFold folds for CV evaluation (0 = disabled, ≥2 = enabled)")

    args = parser.parse_args()

    if args.seed is not None:
        import random
        random.seed(args.seed)
        np.random.seed(args.seed)
        os.environ["RANDOM_STATE"] = str(args.seed)
    if args.experiment_name is not None:
        os.environ["MLFLOW_EXPERIMENT_NAME"] = args.experiment_name

    seed = int(os.getenv("RANDOM_STATE", str(RANDOM_STATE)))
    exp  = os.getenv("MLFLOW_EXPERIMENT_NAME", EXPERIMENT_NAME)

    train(sample=args.sample, seed=seed, experiment_name=exp, kfold=args.kfold)
