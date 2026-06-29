"""
scripts/poison_detection.py — Detecção estatística de envenenamento (genérico)
Aplica uma bateria de testes estatísticos para detectar adulteração dos dados de treino.

Testes executados:
  1. KS test (Kolmogorov-Smirnov)   — shift em features numéricas contínuas
  2. Chi-squared test               — shift em features categóricas
  3. Jensen-Shannon divergence      — adulteração na distribuição do target (label flipping)
  4. Isolation Forest               — amostras anômalas / adversariais no espaço de features
  5. IQR outlier rate               — taxa de outliers por feature (sem referência)
  6. Chaff Detection                — near-duplicates injetados para diluir o sinal de treino

Genérico: funciona com qualquer CSV/Parquet, qualquer coluna target (binário ou multiclasse).
Com --reference: compara baseline vs. atual (modo CI).
Sem --reference: análise standalone da distribuição interna (modo exploratório).

MITRE ATLAS: AML.T0020 Poisoning · AML.T0018 Backdoor ML Model
             AML.T0021ai Spamming with Chaff Data (chaff detection)
Ref: Sotiropoulos cap. 4 (Fig. 4-1) · OpenSSF MLSecOps Whitepaper 2025

Uso:
    # Com referência (CI recomendado):
    python scripts/poison_detection.py \
        --data data/train.csv --reference data/baseline.csv \
        --target is_risky --alpha 0.01

    # Sem referência (análise standalone):
    python scripts/poison_detection.py \
        --data all_findings_flat.csv --target is_risky
"""

import argparse
import json
import logging
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------

def load(path: str, sample: int) -> pd.DataFrame:
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {p}")
    df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p, low_memory=False)
    if sample and len(df) > sample:
        df = df.sample(sample, random_state=42)
    return df.reset_index(drop=True)


def auto_detect_features(df: pd.DataFrame, target: str) -> tuple[list[str], list[str]]:
    """Auto-detecta features numéricas e categóricas (exclui target e colunas livres)."""
    num_cols, cat_cols = [], []
    for col in df.columns:
        if col == target:
            continue
        # Colunas com muitos valores únicos em object = identificadores, excluir
        if df[col].dtype == object:
            nuniq = df[col].nunique()
            if nuniq < 0.5 * len(df) and nuniq < 5000:
                cat_cols.append(col)
        else:
            try:
                pd.to_numeric(df[col])
                num_cols.append(col)
            except (ValueError, TypeError):
                pass
    return num_cols, cat_cols


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence entre duas distribuições de probabilidade."""
    p = p + 1e-10
    q = q + 1e-10
    p /= p.sum()
    q /= q.sum()
    m = (p + q) / 2
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))


# ---------------------------------------------------------------------------
# Testes individuais
# ---------------------------------------------------------------------------

def test_ks(ref: pd.Series, cur: pd.Series, alpha: float) -> dict:
    r = pd.to_numeric(ref, errors="coerce").dropna()
    c = pd.to_numeric(cur, errors="coerce").dropna()
    if len(r) < 10 or len(c) < 10:
        return {"test": "ks", "skipped": True}
    stat, pvalue = stats.ks_2samp(r.values, c.values)
    return {"test": "ks", "statistic": float(stat), "pvalue": float(pvalue),
            "drifted": bool(pvalue < alpha)}


def test_chi2(ref: pd.Series, cur: pd.Series, alpha: float) -> dict:
    ref_counts = ref.fillna("__NULL__").value_counts()
    cur_counts = cur.fillna("__NULL__").value_counts()
    all_cats = set(ref_counts.index) | set(cur_counts.index)
    r = np.array([ref_counts.get(c, 0) for c in all_cats], dtype=float)
    c = np.array([cur_counts.get(c, 0) for c in all_cats], dtype=float)
    # Normalize current to same total as reference
    if c.sum() > 0 and r.sum() > 0:
        c = c / c.sum() * r.sum()
    c = np.maximum(c, 0.1)
    r = np.maximum(r, 0.1)
    stat, pvalue = stats.chisquare(c, f_exp=r)
    return {"test": "chi2", "statistic": float(stat), "pvalue": float(pvalue),
            "drifted": bool(pvalue < alpha)}


def test_label_jsd(ref: pd.Series, cur: pd.Series) -> dict:
    """JS Divergence na distribuição do target. JSD > 0.1 = sinal de label flipping."""
    classes = sorted(set(ref.dropna().astype(str).unique()) | set(cur.dropna().astype(str).unique()))
    r = ref.astype(str).value_counts(normalize=True)
    c = cur.astype(str).value_counts(normalize=True)
    p = np.array([r.get(cl, 0.0) for cl in classes])
    q = np.array([c.get(cl, 0.0) for cl in classes])
    jsd = js_divergence(p, q)
    return {
        "test": "js_divergence_target",
        "jsd": float(jsd),
        "ref_dist": {cl: float(r.get(cl, 0)) for cl in classes},
        "cur_dist": {cl: float(c.get(cl, 0)) for cl in classes},
        "drifted": bool(jsd > 0.1),
    }


def test_isolation_forest(df: pd.DataFrame, num_cols: list[str],
                          contamination: float = 0.05) -> dict:
    """Isolation Forest para detectar amostras anômalas / adversariais."""
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    if not num_cols:
        return {"test": "isolation_forest", "skipped": True, "reason": "no numeric features"}

    X = df[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    iso = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
    labels = iso.fit_predict(X_scaled)
    anomaly_rate = (labels == -1).mean()

    return {
        "test": "isolation_forest",
        "anomaly_count": int((labels == -1).sum()),
        "anomaly_rate": float(anomaly_rate),
        "contamination_threshold": contamination,
        "drifted": bool(anomaly_rate > contamination * 2),
    }


def test_iqr_outlier_rate(df: pd.DataFrame, num_cols: list[str],
                           max_outlier_rate: float = 0.10) -> dict:
    """Taxa de outliers via IQR. Sem necessidade de referência."""
    results = {}
    for col in num_cols:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) < 10:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        outliers = ((s < q1 - 3 * iqr) | (s > q3 + 3 * iqr)).mean()
        results[col] = float(outliers)
    worst = max(results.values(), default=0.0)
    return {
        "test": "iqr_outlier_rate",
        "per_feature": results,
        "max_outlier_rate": float(worst),
        "drifted": bool(worst > max_outlier_rate),
    }


def detect_chaff(df: pd.DataFrame, num_cols: list[str], threshold: float = 0.02) -> dict:
    """
    Chaff Detection — MITRE ATLAS AML.T0021ai "Spamming with Chaff Data".

    Detects near-duplicate rows (chaff) injected to dilute the training signal
    or mask a poisoning pattern.  Strategy:
      1. Hash each row (numeric columns, rounded to 4 dp) → count collisions.
      2. Flag if the near-duplicate rate exceeds `threshold`.

    Uses only numeric features (robust to label-encoding differences).
    With large datasets, runs on a stratified sample for speed.
    """
    import hashlib as _hl

    try:
        cols = [c for c in num_cols if c in df.columns]
        if not cols:
            return {"test": "chaff_detection", "skipped": True, "reason": "no numeric cols"}

        # Round to 4 decimal places to catch near-duplicates with float noise
        sample_df = df[cols].copy()
        for c in cols:
            sample_df[c] = pd.to_numeric(sample_df[c], errors="coerce").round(4)

        # Hash each row as a fixed-width string
        row_hashes = sample_df.apply(
            lambda row: _hl.md5(row.to_csv(index=False, header=False).encode()).hexdigest(),
            axis=1,
        )
        counts = row_hashes.value_counts()
        n_total = len(df)
        n_duplicated = int((counts[counts > 1] - 1).sum())  # extra copies
        chaff_rate = n_duplicated / n_total if n_total > 0 else 0.0
        drifted = chaff_rate > threshold

        return {
            "test": "chaff_detection",
            "n_total": n_total,
            "n_near_duplicates": n_duplicated,
            "chaff_rate": round(chaff_rate, 6),
            "threshold": threshold,
            "drifted": bool(drifted),
        }
    except Exception as exc:
        return {"test": "chaff_detection", "skipped": True, "reason": str(exc)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Statistical data poisoning detection (generic, any dataset)"
    )
    parser.add_argument("--data",      required=True, help="Current dataset (CSV or Parquet)")
    parser.add_argument("--target",    required=True, help="Target column name (e.g. is_risky)")
    parser.add_argument("--reference", default=None,  help="Baseline dataset for comparison (optional)")
    parser.add_argument("--alpha",     type=float, default=0.01,
                        help="Significance level for KS/Chi2 tests (default: 0.01)")
    parser.add_argument("--contamination", type=float, default=0.05,
                        help="Expected anomaly rate for Isolation Forest (default: 0.05 = 5%%)")
    parser.add_argument("--sample",    type=int, default=100_000)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--fail-on-poison", action="store_true",
                        help="Exit 1 if any poisoning signal detected")
    parser.add_argument("--chaff-threshold", type=float, default=0.02,
                        help="Max allowed near-duplicate rate for chaff detection "
                             "(AML.T0021ai, default: 0.02 = 2%%)")
    args = parser.parse_args()

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Loading current dataset: {args.data}")
    cur = load(args.data, args.sample)
    log.info(f"  Shape: {cur.shape} | target='{args.target}'")

    if args.target not in cur.columns:
        log.error(f"Target column '{args.target}' not found. Available: {list(cur.columns)}")
        sys.exit(1)

    num_cols, cat_cols = auto_detect_features(cur, args.target)
    log.info(f"  Numeric features   : {len(num_cols)}")
    log.info(f"  Categorical features: {len(cat_cols)}")

    all_results: dict = {}
    poison_signals: list[str] = []

    # ── Teste 1: IQR outlier rate (standalone, sem referência) ────────────
    log.info("\n[1/6] IQR Outlier Rate…")
    r = test_iqr_outlier_rate(cur, num_cols)
    all_results["iqr_outlier_rate"] = r
    if r["drifted"]:
        poison_signals.append(f"iqr_outlier_rate: max={r['max_outlier_rate']:.3f}")
        log.warning(f"  SIGNAL — outlier rate {r['max_outlier_rate']:.3f} > threshold")
    else:
        log.info(f"  OK — max outlier rate: {r['max_outlier_rate']:.3f}")

    # ── Teste 2: Isolation Forest (standalone) ────────────────────────────
    log.info("[2/6] Isolation Forest (anomaly detection)…")
    r = test_isolation_forest(cur, num_cols, args.contamination)
    all_results["isolation_forest"] = r
    if r.get("drifted"):
        poison_signals.append(f"isolation_forest: anomaly_rate={r['anomaly_rate']:.3f}")
        log.warning(f"  SIGNAL — anomaly rate {r['anomaly_rate']:.3f} > 2x contamination")
    elif r.get("skipped"):
        log.info("  SKIPPED (no numeric features)")
    else:
        log.info(f"  OK — anomaly rate: {r['anomaly_rate']:.3f} ({r['anomaly_count']} samples)")

    # ── Teste 3: Chaff Detection (near-duplicates) ────────────────────────
    log.info("[3/6] Chaff Detection (near-duplicate rows — AML.T0021ai)…")
    r = detect_chaff(cur, num_cols, threshold=args.chaff_threshold)
    all_results["chaff_detection"] = r
    if r.get("skipped"):
        log.info(f"  SKIPPED: {r.get('reason', '')}")
    elif r["drifted"]:
        poison_signals.append(f"chaff_detection: rate={r['chaff_rate']:.4f} > {args.chaff_threshold}")
        log.warning(f"  SIGNAL — near-duplicate rate {r['chaff_rate']:.4f} ({r['n_near_duplicates']} rows)")
        log.warning("  MITRE ATLAS AML.T0021ai: possible chaff injection detected.")
    else:
        log.info(f"  OK — near-duplicate rate: {r['chaff_rate']:.4f} ({r['n_near_duplicates']} rows)")

    if args.reference:
        log.info(f"\nLoading reference dataset: {args.reference}")
        ref = load(args.reference, args.sample)

        # ── Teste 4: KS test por feature numérica ─────────────────────────
        log.info("[4/6] KS Test (numerical features)…")
        ks_results, ks_drifted = {}, []
        for col in num_cols:
            if col not in ref.columns:
                continue
            r = test_ks(ref[col], cur[col], args.alpha)
            ks_results[col] = r
            if r.get("drifted"):
                ks_drifted.append(col)
                log.warning(f"  DRIFT — {col}: p={r['pvalue']:.4f} < α={args.alpha}")
            else:
                log.info(f"  OK    — {col}: p={r.get('pvalue', 'N/A')}")
        all_results["ks_test"] = ks_results
        if ks_drifted:
            poison_signals.append(f"ks_test: drifted={ks_drifted}")

        # ── Teste 5: Chi-squared por feature categórica ───────────────────
        log.info("[5/6] Chi-squared Test (categorical features)…")
        chi2_results, chi2_drifted = {}, []
        for col in cat_cols:
            if col not in ref.columns:
                continue
            r = test_chi2(ref[col], cur[col], args.alpha)
            chi2_results[col] = r
            if r.get("drifted"):
                chi2_drifted.append(col)
                log.warning(f"  DRIFT — {col}: p={r['pvalue']:.4f}")
        all_results["chi2_test"] = chi2_results
        if chi2_drifted:
            poison_signals.append(f"chi2_test: drifted={chi2_drifted}")

        # ── Teste 6: JS divergence no target (label flipping) ─────────────
        log.info("[6/6] JS Divergence on target distribution (label flipping)…")
        r = test_label_jsd(ref[args.target], cur[args.target])
        all_results["label_jsd"] = r
        if r["drifted"]:
            poison_signals.append(f"label_jsd: JSD={r['jsd']:.4f} > 0.1")
            log.warning(f"  SIGNAL — JSD={r['jsd']:.4f} — possível label flipping!")
            log.warning(f"  Ref dist: {r['ref_dist']}")
            log.warning(f"  Cur dist: {r['cur_dist']}")
        else:
            log.info(f"  OK — JSD={r['jsd']:.4f}")
    else:
        log.info("[4-6/6] KS/Chi2/JSD skipped (no --reference provided — use for CI with baseline)")

    # ── Sumário ─────────────────────────────────────────────────────────
    passed = len(poison_signals) == 0
    report = {
        "passed": passed,
        "poison_signals": poison_signals,
        "tests": all_results,
        "config": {"alpha": args.alpha, "contamination": args.contamination,
                   "chaff_threshold": args.chaff_threshold},
    }
    with open(out_dir / "poison_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    log.info(f"\n=== Poison Detection Gate ===")
    log.info(f"  Signals detected : {len(poison_signals)}")
    if poison_signals:
        for s in poison_signals:
            log.warning(f"  ⚠  {s}")
        log.info("  NOTE: thresholds empíricos — calibre com baseline do domínio antes de tornar bloqueantes.")
        log.info("  MITRE ATLAS AML.T0020 · AML.T0018 · AML.T0021ai")

    if not passed and args.fail_on_poison:
        log.error("Poison Detection Gate FAILED.")
        sys.exit(1)

    log.info("Poison Detection Gate: PASSED" if passed else "Poison Detection Gate: WARNING (--fail-on-poison não ativo)")


if __name__ == "__main__":
    main()
