"""
scripts/lineage.py — Lineage Manifest (provenance unificada)
Captura toda a cadeia de custódia do artefato ML: dados → treinamento → modelo → avaliação.

Gera um único JSON que conecta:
  • SHA256 de cada artefato de entrada (dados, config, código)
  • Parâmetros de treinamento (lidos do MLflow run ou argparse)
  • SHA256 do modelo de saída
  • Métricas de avaliação e resultados de segurança
  • Identidade do executor (git commit, CI run id)

Compatível com SLSA Provenance Level 2 e CycloneDX ML BOM.

MITRE ATLAS: AML.T0010 ML Supply Chain Compromise · AML.T0020 Poison Training Data
SLSA:        https://slsa.dev/provenance/v0.2
OpenSSF MLSecOps Whitepaper 2025: §4 "Artifact Provenance"

Uso:
    # Após o treinamento:
    python scripts/lineage.py \\
        --data all_findings_flat.csv \\
        --model model/rf_model.pkl \\
        --mlflow-run <RUN_ID> \\
        --output results/lineage_manifest.json

    # Verificar contra um manifest anterior (CI):
    python scripts/lineage.py \\
        --verify results/lineage_manifest.json \\
        --model model/rf_model.pkl
"""

import argparse
import hashlib
import json
import logging
import os
import pathlib
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

MANIFEST_VERSION = "1.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def git_info() -> dict[str, str]:
    """Returns current git commit hash and author if inside a repo."""
    info: dict[str, str] = {}
    try:
        info["commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        info["branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        info["author"] = subprocess.check_output(
            ["git", "log", "-1", "--format=%ae"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        pass
    return info


def ci_info() -> dict[str, str]:
    """Captures standard GitHub Actions / GitLab CI environment variables."""
    keys = [
        "GITHUB_RUN_ID", "GITHUB_RUN_NUMBER", "GITHUB_WORKFLOW",
        "GITHUB_ACTOR", "GITHUB_SHA", "GITHUB_REF",
        "CI_PIPELINE_ID", "CI_JOB_URL",
    ]
    return {k: os.environ[k] for k in keys if k in os.environ}


def mlflow_run_params(run_id: str | None) -> dict[str, Any]:
    """Fetches params and metrics from an MLflow run (if mlflow is installed)."""
    if not run_id:
        return {}
    try:
        import mlflow
        client = mlflow.MlflowClient()
        run = client.get_run(run_id)
        return {
            "run_id": run_id,
            "params": dict(run.data.params),
            "metrics": {k: float(v) for k, v in run.data.metrics.items()},
            "tags": {k: v for k, v in run.data.tags.items()
                     if not k.startswith("mlflow.")},
            "artifact_uri": run.info.artifact_uri,
            "status": run.info.status,
        }
    except Exception as exc:
        log.warning(f"  MLflow lookup failed: {exc}")
        return {"run_id": run_id, "error": str(exc)}


def results_snapshot(results_dir: pathlib.Path) -> dict[str, Any]:
    """
    Reads all JSON report files from the results/ directory and embeds their
    pass/fail status into the manifest for full traceability.
    """
    snapshot: dict[str, Any] = {}
    if not results_dir.exists():
        return snapshot
    for p in sorted(results_dir.glob("*.json")):
        try:
            with open(p) as f:
                data = json.load(f)
            snapshot[p.name] = {
                "sha256": sha256_file(p),
                "passed": data.get("passed"),
            }
        except Exception:
            pass
    return snapshot


# ---------------------------------------------------------------------------
# Build manifest
# ---------------------------------------------------------------------------

def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "_type": "https://in-toto.io/Statement/v0.1",
        "predicateType": "https://slsa.dev/provenance/v0.2",
        "manifest_version": MANIFEST_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "builder": {
            "id": "MLSecOps-lineage-tracker",
            "version": "1.0",
        },
        "materials": {},      # inputs
        "invocation": {},     # how the build was triggered
        "byproducts": {},     # outputs + security results
    }

    # ── Materials: hash each input artifact ─────────────────────────────────
    log.info("=== Hashing input materials ===")
    materials: dict[str, Any] = {}

    data_files = args.data if args.data else []
    for dp in data_files:
        p = pathlib.Path(dp)
        if p.exists():
            materials[str(p)] = {"sha256": sha256_file(p), "role": "training_data"}
            log.info(f"  data     {p.name}: {materials[str(p)]['sha256'][:16]}…")
        else:
            log.warning(f"  [MISSING] {p}")

    config_files = args.config if args.config else []
    for cp in config_files:
        p = pathlib.Path(cp)
        if p.exists():
            materials[str(p)] = {"sha256": sha256_file(p), "role": "config"}
            log.info(f"  config   {p.name}: {materials[str(p)]['sha256'][:16]}…")

    script_files = args.scripts if args.scripts else []
    for sp in script_files:
        p = pathlib.Path(sp)
        if p.exists():
            materials[str(p)] = {"sha256": sha256_file(p), "role": "training_script"}
            log.info(f"  script   {p.name}: {materials[str(p)]['sha256'][:16]}…")

    manifest["materials"] = materials

    # ── Invocation: git + CI context ────────────────────────────────────────
    log.info("=== Capturing invocation context ===")
    invocation: dict[str, Any] = {
        "git": git_info(),
        "ci": ci_info(),
    }
    if args.mlflow_run:
        log.info(f"  Fetching MLflow run: {args.mlflow_run}")
        invocation["mlflow"] = mlflow_run_params(args.mlflow_run)
    manifest["invocation"] = invocation

    # ── Byproducts: model artifact + security reports ───────────────────────
    log.info("=== Hashing model artifacts ===")
    byproducts: dict[str, Any] = {}

    model_files = args.model if args.model else []
    for mp in model_files:
        p = pathlib.Path(mp)
        if p.exists():
            byproducts[str(p)] = {"sha256": sha256_file(p), "role": "model"}
            log.info(f"  model    {p.name}: {byproducts[str(p)]['sha256'][:16]}…")
        else:
            log.warning(f"  [MISSING] {p}")

    results_dir = pathlib.Path(args.results_dir)
    log.info(f"=== Snapshotting security results in {results_dir}/ ===")
    byproducts["security_reports"] = results_snapshot(results_dir)
    for name, meta in byproducts.get("security_reports", {}).items():
        status = "PASS" if meta.get("passed") else ("FAIL" if meta.get("passed") is False else "N/A")
        log.info(f"  [{status}] {name}")

    manifest["byproducts"] = byproducts

    # ── Top-level integrity digest (manifest self-hash placeholder) ──────────
    manifest_json = json.dumps(manifest, sort_keys=True, default=str)
    manifest["manifest_sha256"] = hashlib.sha256(manifest_json.encode()).hexdigest()

    return manifest


# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------

def verify_manifest(manifest_path: pathlib.Path, model_files: list[str]) -> None:
    """
    Re-hashes current artifacts and compares against the stored manifest.
    Fails if any model or data hash has drifted — indicates supply-chain tampering.
    """
    log.info(f"=== Verifying against manifest: {manifest_path} ===")
    with open(manifest_path) as f:
        stored = json.load(f)

    all_artifacts = {**stored.get("materials", {}), **stored.get("byproducts", {})}
    tampered: list[str] = []
    missing:  list[str] = []

    for fp_str, meta in all_artifacts.items():
        if not isinstance(meta, dict) or "sha256" not in meta:
            continue
        p = pathlib.Path(fp_str)
        if not p.exists():
            missing.append(fp_str)
            log.warning(f"  [MISSING]  {fp_str}")
            continue
        current_hash = sha256_file(p)
        if current_hash != meta["sha256"]:
            tampered.append(fp_str)
            log.error(f"  [TAMPERED] {fp_str}")
            log.error(f"    stored : {meta['sha256']}")
            log.error(f"    current: {current_hash}")
        else:
            log.info(f"  [OK]       {fp_str}")

    if model_files:
        for mp in model_files:
            if mp not in all_artifacts and pathlib.Path(mp).exists():
                log.warning(f"  [NEW-MODEL] {mp} — not in manifest, was it replaced?")

    if tampered or missing:
        log.error(f"\nLineage Verify FAILED — tampered={len(tampered)}, missing={len(missing)}")
        log.error("MITRE ATLAS AML.T0010: possible supply-chain compromise or artifact substitution.")
        sys.exit(1)

    log.info("\nLineage Verify: PASSED — all artifacts match stored manifest.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lineage manifest: capture full ML artifact provenance chain"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── build sub-command ──────────────────────────────────────────────────
    build_p = subparsers.add_parser("build", help="Generate a new lineage manifest")
    build_p.add_argument("--data", nargs="+", default=[],
                         help="Input dataset files (CSV, Parquet, etc.)")
    build_p.add_argument("--model", nargs="+", default=[],
                         help="Output model artifact files (pkl, joblib, onnx, …)")
    build_p.add_argument("--config", nargs="+", default=[],
                         help="Config files to include (policy.yml, requirements.txt, …)")
    build_p.add_argument("--scripts", nargs="+", default=[],
                         help="Training/evaluation script files")
    build_p.add_argument("--mlflow-run", default=None,
                         help="MLflow run ID to embed params/metrics")
    build_p.add_argument("--results-dir", default="results",
                         help="Directory of security JSON reports to snapshot (default: results/)")
    build_p.add_argument("--output", default="results/lineage_manifest.json",
                         help="Output manifest path (default: results/lineage_manifest.json)")

    # ── verify sub-command ─────────────────────────────────────────────────
    verify_p = subparsers.add_parser("verify", help="Verify artifacts against a stored manifest")
    verify_p.add_argument("manifest", help="Path to the lineage manifest JSON")
    verify_p.add_argument("--model", nargs="+", default=[],
                          help="Current model files to cross-check")

    args = parser.parse_args()

    if args.command == "build":
        manifest = build_manifest(args)
        out = pathlib.Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(manifest, f, indent=2, default=str)
        log.info(f"\nLineage manifest written: {out}")
        log.info(f"  Manifest SHA256: {manifest['manifest_sha256'][:16]}…")

    elif args.command == "verify":
        verify_manifest(pathlib.Path(args.manifest), args.model)


if __name__ == "__main__":
    main()
