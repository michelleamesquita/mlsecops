"""
scripts/model_behavioral_baseline.py — Fingerprinting comportamental do modelo (genérico)

Salva as predições do modelo sobre um conjunto de referência fixo ("digital fingerprint").
Em todo re-treino compara o comportamento atual com o baseline para detectar:
  - Poisoning silencioso (predictions mudaram sem mudança explícita de código)
  - Backdoor por trigger (predição muda para entradas específicas)
  - Regressão de segurança (propriedades adversariais degradaram)
  - Adulteração pós-treino do artefato

MITRE ATLAS: AML.T0018 Backdoor ML Model · AML.T0020 Poisoning
AISP Module 6: Supply Chain Integrity

Modos de operação:
  --update  : Salva baseline (deve rodar na primeira vez ou após treino aprovado)
  (default) : Verifica comportamento atual vs. baseline

Uso:
    # Primeira vez (salvar baseline):
    python scripts/model_behavioral_baseline.py \
        --data dataset.csv --target label \
        --model model/rf_model.pkl \
        --update

    # Verificar (CI — todo re-treino):
    python scripts/model_behavioral_baseline.py \
        --data dataset.csv --target label \
        --model model/rf_model.pkl \
        --baseline model/behavioral_baseline.json
"""

import argparse
import json
import logging
import pathlib
import sys

import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def load_meta(meta_path: str | None) -> dict | None:
    if not meta_path or not pathlib.Path(meta_path).exists():
        return None
    with open(meta_path) as f:
        return json.load(f)


def build_reference_set(df: pd.DataFrame, target: str, meta: dict | None,
                        n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Seleciona conjunto de referência fixo (determinístico via seed)."""
    from sklearn.preprocessing import LabelEncoder

    df = df.sample(min(n, len(df)), random_state=seed).reset_index(drop=True)
    df_copy = df.copy()

    if meta:
        for col, classes in meta.get("encoder_classes", {}).items():
            if col in df_copy.columns:
                le = LabelEncoder()
                le.classes_ = np.array(classes)
                df_copy[col] = df_copy[col].fillna("UNKNOWN").astype(str)
                known = set(le.classes_)
                df_copy[col] = df_copy[col].apply(lambda x: x if x in known else le.classes_[0])
                df_copy[col] = le.transform(df_copy[col])
        for col in meta.get("num_features", []):
            if col in df_copy.columns:
                df_copy[col] = pd.to_numeric(df_copy[col], errors="coerce").fillna(0)
        feature_cols = [c for c in meta["feature_names"] if c in df_copy.columns]
    else:
        for col in df_copy.select_dtypes(include="object").columns:
            if col != target:
                df_copy[col] = df_copy[col].fillna("UNKNOWN").astype("category").cat.codes
        for col in df_copy.columns:
            if col != target:
                df_copy[col] = pd.to_numeric(df_copy[col], errors="coerce").fillna(0)
        feature_cols = [c for c in df_copy.columns if c != target and df_copy[c].dtype != object]

    X = df_copy[feature_cols].values.astype(np.float32)
    y = pd.to_numeric(df[target], errors="coerce").fillna(0).astype(int).values
    return X, y


def compute_fingerprint(clf, X: np.ndarray) -> dict:
    """Gera o fingerprint comportamental: predições + probabilidades."""
    preds = clf.predict(X).tolist()
    probs = clf.predict_proba(X).tolist()
    classes = clf.classes_.tolist()
    return {
        "n_samples": len(X),
        "classes": [int(c) for c in classes],
        "predictions": preds,
        "mean_probabilities": np.array(probs).mean(axis=0).tolist(),
        "positive_rate": float(np.mean(preds)),
    }


def compare_fingerprints(baseline: dict, current: dict,
                         max_jsd: float, max_disagreement: float) -> tuple[bool, list[str]]:
    """Compara dois fingerprints e retorna (passed, issues)."""
    issues: list[str] = []

    # JSD nas probabilidades médias (sensível a mudanças distribucionais)
    b_probs = np.array(baseline["mean_probabilities"]) + 1e-10
    c_probs = np.array(current["mean_probabilities"]) + 1e-10
    b_probs /= b_probs.sum()
    c_probs /= c_probs.sum()
    jsd = float(jensenshannon(b_probs, c_probs))

    log.info(f"  JS Divergence (mean probs)    : {jsd:.6f}  (threshold: {max_jsd})")

    if jsd > max_jsd:
        issues.append(f"JSD={jsd:.4f} > {max_jsd} — distribuição de predições mudou significativamente")

    # Taxa de desacordo nas predições (amostras que mudaram de classe)
    b_preds = np.array(baseline["predictions"])
    c_preds = np.array(current["predictions"])

    if len(b_preds) == len(c_preds):
        disagreement = float((b_preds != c_preds).mean())
        log.info(f"  Disagreement rate             : {disagreement:.4f}  (threshold: {max_disagreement})")
        if disagreement > max_disagreement:
            issues.append(f"Disagreement={disagreement:.4f} > {max_disagreement} — {int(disagreement*len(b_preds))} predições mudaram")
    else:
        log.warning(f"  Reference set sizes differ: baseline={len(b_preds)} vs current={len(c_preds)}")

    # Taxa de positivos
    b_rate = baseline["positive_rate"]
    c_rate = current["positive_rate"]
    rate_delta = abs(c_rate - b_rate)
    log.info(f"  Positive rate: baseline={b_rate:.4f}  current={c_rate:.4f}  Δ={rate_delta:.4f}")
    if rate_delta > 0.15:
        issues.append(f"Positive rate shifted by {rate_delta:.4f} — possível label flipping no treino")

    return len(issues) == 0, issues


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Model behavioral baseline — fingerprinting e drift detection"
    )
    parser.add_argument("--data",          required=True)
    parser.add_argument("--target",        required=True)
    parser.add_argument("--model",         required=True)
    parser.add_argument("--meta",          default=None, help="feature_names.json (opcional)")
    parser.add_argument("--baseline",      default="model/behavioral_baseline.json",
                        help="Arquivo de baseline (default: model/behavioral_baseline.json)")
    parser.add_argument("--update",        action="store_true",
                        help="Salvar/atualizar baseline (em vez de verificar)")
    parser.add_argument("--n-reference",   type=int, default=1000,
                        help="Tamanho do conjunto de referência fixo (default: 1000)")
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument("--max-jsd",       type=float, default=0.05,
                        help="JSD máximo tolerado (default: 0.05). Calibre com baseline do domínio.")
    parser.add_argument("--max-disagreement", type=float, default=0.05,
                        help="Taxa máxima de predições que mudaram (default: 0.05 = 5%%)")
    parser.add_argument("--sample",        type=int, default=50_000)
    parser.add_argument("--output-dir",    default="results")
    args = parser.parse_args()

    for p in [args.model, args.data]:
        if not pathlib.Path(p).exists():
            log.error(f"File not found: {p}")
            sys.exit(1)

    log.info(f"Loading model: {args.model}")
    clf = joblib.load(args.model)
    meta = load_meta(args.meta)

    log.info(f"Loading data ({args.sample:,} rows)…")
    p = pathlib.Path(args.data)
    df = pd.read_csv(p, low_memory=False, nrows=args.sample) \
         if p.suffix == ".csv" else pd.read_parquet(p)

    log.info(f"  Building fixed reference set (n={args.n_reference}, seed={args.seed})…")
    X_ref, y_ref = build_reference_set(df, args.target, meta, args.n_reference, args.seed)

    log.info("  Computing behavioral fingerprint…")
    fingerprint = compute_fingerprint(clf, X_ref)
    fingerprint["seed"]     = args.seed
    fingerprint["model"]    = args.model
    fingerprint["data"]     = args.data

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = pathlib.Path(args.baseline)

    if args.update:
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        with open(baseline_path, "w") as f:
            json.dump(fingerprint, f, indent=2)
        log.info(f"\n  Baseline salvo: {baseline_path}")
        log.info(f"  n_samples={fingerprint['n_samples']}, positive_rate={fingerprint['positive_rate']:.4f}")
        log.info("  Modelo aprovado — use este baseline para verificar re-treinamentos futuros.")
        return

    # Modo de verificação
    if not baseline_path.exists():
        log.warning(f"  Baseline não encontrado: {baseline_path}")
        log.warning("  Execute com --update após o primeiro treino aprovado para criar o baseline.")
        log.warning("  Skipping behavioral check (first run).")
        with open(out_dir / "behavioral_baseline_report.json", "w") as f:
            json.dump({"passed": True, "skipped": True,
                       "reason": "no baseline yet"}, f, indent=2)
        return

    with open(baseline_path) as f:
        baseline = json.load(f)

    log.info(f"\n=== Model Behavioral Baseline Gate ===")
    log.info(f"  Baseline: {baseline_path}  (n={baseline['n_samples']})")
    log.info(f"  Current : n={fingerprint['n_samples']}")

    passed, issues = compare_fingerprints(
        baseline, fingerprint, args.max_jsd, args.max_disagreement
    )

    report = {
        "passed": passed,
        "issues": issues,
        "baseline": {
            "positive_rate": baseline["positive_rate"],
            "mean_probabilities": baseline["mean_probabilities"],
        },
        "current": {
            "positive_rate": fingerprint["positive_rate"],
            "mean_probabilities": fingerprint["mean_probabilities"],
        },
        "thresholds": {"max_jsd": args.max_jsd, "max_disagreement": args.max_disagreement},
    }
    with open(out_dir / "behavioral_baseline_report.json", "w") as f:
        json.dump(report, f, indent=2)

    if issues:
        log.warning(f"\n  Issues detectados:")
        for issue in issues:
            log.warning(f"    ⚠  {issue}")
        log.warning("  NOTE: thresholds empíricos — calibre com baseline do domínio.")
        log.warning("  MITRE ATLAS AML.T0018 · AML.T0020")

    if not passed:
        log.error("  Behavioral Baseline Gate FAILED — comportamento do modelo mudou.")
        log.error("  Revise: dados de treino, label encoding, features selecionadas.")
        sys.exit(1)

    log.info("  Behavioral Baseline Gate: PASSED")


if __name__ == "__main__":
    main()
