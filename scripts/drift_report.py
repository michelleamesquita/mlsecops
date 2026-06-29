"""
scripts/drift_report.py — Data drift via Evidently AI (Estágio 01)
Ameaça mitigada: AML.T0020 Poisoning / distribuição de dados adulterada

Compara um dataset de referência (baseline) contra o atual.
Gera relatório HTML em results/drift_report.html e falha se drift for detectado.

Uso:
    python scripts/drift_report.py \
        --reference data/baseline.parquet \
        --current   data/train.parquet \
        --fail-on-drift
"""

import argparse
import sys
import logging
import pathlib
import json

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

NUM_FEATURES = [
    "line_number", "patch_lines", "patch_added", "patch_removed",
    "patch_files_touched", "patch_hunks", "patch_churn", "patch_net",
    "prompt_chars", "prompt_lines", "prompt_tokens",
    "prompt_has_security_guidelines", "temperature",
    "cwe_prevalence_overall", "cwe_severity_score", "cwe_weighted_severity",
    "is_risky",
]


def load_data(path: str, sample: int = 50_000) -> pd.DataFrame:
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {p}")
    if p.suffix == ".parquet":
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p, low_memory=False)
    # Keep only numeric columns for drift analysis
    cols = [c for c in NUM_FEATURES if c in df.columns]
    df = df[cols]
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sample(min(sample, len(df)), random_state=42).reset_index(drop=True)


def compute_psi(expected: pd.Series, actual: pd.Series, buckets: int = 10) -> float:
    """Population Stability Index (PSI). PSI > 0.2 → significant drift."""
    expected_clean = expected.dropna()
    actual_clean = actual.dropna()
    if expected_clean.empty or actual_clean.empty:
        return 0.0

    breakpoints = pd.qcut(expected_clean, q=buckets, duplicates="drop", retbins=True)[1]
    breakpoints[0] = -float("inf")
    breakpoints[-1] = float("inf")

    def bucket_pct(series):
        counts = pd.cut(series, bins=breakpoints).value_counts(normalize=True).sort_index()
        return counts.clip(lower=1e-6)

    exp_pct = bucket_pct(expected_clean)
    act_pct = bucket_pct(actual_clean)
    psi = ((act_pct - exp_pct) * (act_pct / exp_pct).apply(lambda x: x if x > 0 else 1e-6).apply(
        __import__("math").log
    )).sum()
    return abs(psi)


def main() -> None:
    parser = argparse.ArgumentParser(description="Data drift report via Evidently AI")
    parser.add_argument("--reference", required=True, help="Reference (baseline) dataset path")
    parser.add_argument("--current",   required=True, help="Current dataset path")
    parser.add_argument("--fail-on-drift", action="store_true",
                        help="Exit with code 1 if significant drift detected")
    parser.add_argument("--psi-threshold", type=float, default=0.2,
                        help="PSI threshold for drift (default: 0.2). Calibre com baseline do domínio.")
    parser.add_argument("--sample", type=int, default=50_000)
    args = parser.parse_args()

    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset
        USE_EVIDENTLY = True
    except ImportError:
        log.warning("evidently not installed — falling back to PSI-based drift check")
        USE_EVIDENTLY = False

    log.info(f"Loading reference: {args.reference}")
    ref = load_data(args.reference, args.sample)
    log.info(f"Loading current:   {args.current}")
    cur = load_data(args.current, args.sample)

    pathlib.Path("results").mkdir(exist_ok=True)
    drift_detected = False

    if USE_EVIDENTLY:
        log.info("Running Evidently DataDriftPreset…")
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=ref, current_data=cur)
        report.save_html("results/drift_report.html")

        result_dict = report.as_dict()
        drift_share = result_dict.get("metrics", [{}])[0].get(
            "result", {}
        ).get("share_of_drifted_columns", 0)
        drift_detected = drift_share > 0.3  # >30% colunas com drift = sinal de alerta

        log.info(f"\n=== Evidently Drift Report ===")
        log.info(f"  Share of drifted columns: {drift_share:.2%}")
        log.info(f"  Report: results/drift_report.html")
    else:
        # Fallback: PSI por coluna
        log.info("=== PSI Drift Check ===")
        psi_results = {}
        for col in ref.columns:
            if col in cur.columns:
                psi = compute_psi(ref[col], cur[col])
                psi_results[col] = psi
                status = "DRIFT" if psi > args.psi_threshold else "OK"
                log.info(f"  {col:<35} PSI={psi:.4f}  [{status}]")
                if psi > args.psi_threshold:
                    drift_detected = True

        with open("results/drift_report.json", "w") as f:
            json.dump(psi_results, f, indent=2)

        # Minimal HTML
        rows = "".join(
            f"<tr><td>{c}</td><td>{v:.4f}</td>"
            f"<td style='color:{'red' if v>args.psi_threshold else 'green'}'>"
            f"{'DRIFT' if v>args.psi_threshold else 'OK'}</td></tr>"
            for c, v in psi_results.items()
        )
        html = f"""<html><body><h2>PSI Drift Report</h2>
        <p>Threshold: {args.psi_threshold} (PSI >0.2 = significant drift)</p>
        <table border='1'><tr><th>Feature</th><th>PSI</th><th>Status</th></tr>
        {rows}</table></body></html>"""
        with open("results/drift_report.html", "w") as f:
            f.write(html)

    if drift_detected and args.fail_on_drift:
        log.error("Drift Gate FAILED — distribuição do dataset mudou significativamente.")
        log.error("Revise os dados antes de treinar. Calibre o threshold com baseline do domínio.")
        sys.exit(1)

    log.info("Drift Gate: PASSED")


if __name__ == "__main__":
    main()
