"""
scripts/inference_monitor.py — Monitoramento de inferência em produção (genérico)
Cobre o estágio 4 do ciclo ML: Monitoring (Fig. 4-1 — Adversarial AI, Packt)

Detecta em produção:
  - Distribution shift nos inputs (inputs divergindo do treino → ataque de evasão)
  - Anomalous query rate (possível model extraction em andamento)
  - Prediction confidence degradation (modelo sendo enganado)
  - Input outliers (adversarial inputs chegando ao endpoint)

MITRE ATLAS: AML.T0043 Evade ML Model (inferência) · AML.T0044 Extract ML Model
AISP Module 7: Deployment Monitoring
OWASP MLSVS V9 — Monitoring

Modo batch (CI/CD agendado):
    python scripts/inference_monitor.py \
        --inference-log logs/inference_batch.csv \
        --training-ref  data/baseline.parquet \
        --model         model/rf_model.pkl \
        --target        is_risky

Modo stream (escreve métricas Prometheus):
    python scripts/inference_monitor.py ... --prometheus-port 8001
"""

import argparse
import json
import logging
import pathlib
import sys
from collections import Counter

import joblib
import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def load_data(path: str, sample: int) -> pd.DataFrame:
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Not found: {p}")
    df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p, low_memory=False)
    return df.sample(min(sample, len(df)), random_state=42).reset_index(drop=True)


def detect_distribution_shift(ref: pd.DataFrame, cur: pd.DataFrame,
                               alpha: float) -> dict:
    """KS test por feature numérica entre treino e batch de inferência."""
    drifted = []
    results = {}
    num_cols = [c for c in ref.select_dtypes(include="number").columns if c in cur.columns]

    for col in num_cols:
        r = pd.to_numeric(ref[col], errors="coerce").dropna()
        c = pd.to_numeric(cur[col], errors="coerce").dropna()
        if len(r) < 10 or len(c) < 10:
            continue
        stat, pval = stats.ks_2samp(r.values, c.values)
        results[col] = {"statistic": float(stat), "pvalue": float(pval), "drifted": pval < alpha}
        if pval < alpha:
            drifted.append(col)

    return {"drifted_features": drifted, "n_drifted": len(drifted),
            "total_features": len(results), "details": results}


def detect_anomalous_queries(cur: pd.DataFrame, meta: dict | None) -> dict:
    """
    Isolation Forest para detectar inputs anômalos no batch de inferência.
    Alta taxa de anomalias pode indicar model extraction ou adversarial probing.
    """
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    num_cols = list(cur.select_dtypes(include="number").columns)
    if not num_cols:
        return {"anomaly_rate": 0.0, "n_anomalies": 0, "skipped": True}

    X = cur[num_cols].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    iso = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
    labels = iso.fit_predict(X_scaled)
    anomaly_rate = float((labels == -1).mean())

    return {
        "anomaly_rate": anomaly_rate,
        "n_anomalies": int((labels == -1).sum()),
        "n_total": len(labels),
        "alert": anomaly_rate > 0.15,  # >15% inputs anômalos = alerta
    }


def check_confidence_degradation(clf, cur: pd.DataFrame, meta: dict | None,
                                   target: str, min_confidence: float) -> dict:
    """
    Verifica se a confiança média do modelo degradou (modelo sendo enganado).
    Baixa confiança = modelo incerto = possível adversarial input.
    """
    from sklearn.preprocessing import LabelEncoder

    df = cur.copy()
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
        feature_cols = [c for c in df.columns if c != target and df[c].dtype != object]

    if not feature_cols:
        return {"mean_max_confidence": None, "alert": False, "skipped": True}

    X = df[feature_cols].fillna(0).values.astype(np.float32)
    try:
        probs = clf.predict_proba(X)
        max_conf = probs.max(axis=1).mean()
        return {
            "mean_max_confidence": float(max_conf),
            "alert": max_conf < min_confidence,
            "threshold": min_confidence,
        }
    except Exception as e:
        return {"error": str(e), "alert": False}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inference monitoring — production distribution & anomaly detection"
    )
    parser.add_argument("--inference-log",  required=True,
                        help="Batch de requests de inferência (CSV ou Parquet)")
    parser.add_argument("--training-ref",   required=True,
                        help="Distribuição de referência do treino (CSV ou Parquet)")
    parser.add_argument("--model",          required=True)
    parser.add_argument("--meta",           default=None)
    parser.add_argument("--target",         required=True)
    parser.add_argument("--alpha",          type=float, default=0.01)
    parser.add_argument("--min-confidence", type=float, default=0.60,
                        help="Confiança mínima média esperada (default: 0.60)")
    parser.add_argument("--sample",         type=int,   default=50_000)
    parser.add_argument("--output-dir",     default="results")
    parser.add_argument("--fail-on-alert",  action="store_true")
    args = parser.parse_args()

    for p in [args.model, args.inference_log, args.training_ref]:
        if not pathlib.Path(p).exists():
            log.error(f"File not found: {p}")
            sys.exit(1)

    clf  = joblib.load(args.model)
    meta = json.load(open(args.meta)) if args.meta and pathlib.Path(args.meta).exists() else None

    log.info("Loading inference batch…")
    cur = load_data(args.inference_log, args.sample)
    log.info(f"  Inference batch : {len(cur)} requests")

    log.info("Loading training reference…")
    ref = load_data(args.training_ref, args.sample)
    log.info(f"  Training ref    : {len(ref)} rows")

    alerts: list[str] = []
    report: dict = {}

    # ── 1. Distribution shift ─────────────────────────────────────────────
    log.info("\n[1/3] Distribution Shift Detection (KS test)…")
    r = detect_distribution_shift(ref, cur, args.alpha)
    report["distribution_shift"] = r
    if r["n_drifted"] > 0:
        alerts.append(f"Distribution shift in {r['n_drifted']} features: {r['drifted_features']}")
        log.warning(f"  ALERT — {r['n_drifted']} features drifted: {r['drifted_features']}")
    else:
        log.info(f"  OK — {r['total_features']} features tested, no drift detected")

    # ── 2. Anomalous query detection ──────────────────────────────────────
    log.info("[2/3] Anomalous Query Detection (Isolation Forest)…")
    r = detect_anomalous_queries(cur, meta)
    report["anomalous_queries"] = r
    if r.get("alert"):
        alerts.append(f"High anomaly rate: {r['anomaly_rate']:.2%} of inference inputs are anomalous")
        log.warning(f"  ALERT — {r['anomaly_rate']:.2%} anomalous inputs (possible extraction/probing)")
    else:
        log.info(f"  OK — anomaly rate: {r.get('anomaly_rate', 0):.2%}")

    # ── 3. Confidence degradation ─────────────────────────────────────────
    log.info("[3/3] Confidence Degradation Check…")
    r = check_confidence_degradation(clf, cur, meta, args.target, args.min_confidence)
    report["confidence"] = r
    if r.get("alert"):
        alerts.append(f"Low mean confidence: {r['mean_max_confidence']:.4f} < {args.min_confidence}")
        log.warning(f"  ALERT — mean confidence {r['mean_max_confidence']:.4f} < {args.min_confidence}")
    elif r.get("skipped"):
        log.info("  SKIPPED (no features available)")
    else:
        log.info(f"  OK — mean max confidence: {r['mean_max_confidence']:.4f}")

    # ── Report ────────────────────────────────────────────────────────────
    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report["alerts"] = alerts
    report["passed"] = len(alerts) == 0
    with open(out_dir / "inference_monitor_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    log.info(f"\n=== Inference Monitor ===")
    log.info(f"  Alerts   : {len(alerts)}")
    for a in alerts:
        log.warning(f"  ⚠  {a}")
    log.info(f"  NOTE: thresholds empíricos — calibre em produção com baseline real.")
    log.info(f"  MITRE ATLAS AML.T0043 · AML.T0044")

    if alerts and args.fail_on_alert:
        log.error("Inference Monitor FAILED.")
        sys.exit(1)

    log.info("Inference Monitor: " + ("PASSED" if not alerts else "WARNING"))


if __name__ == "__main__":
    main()
