"""
scripts/generate_mlbom.py — CycloneDX ML BOM (Software Bill of Materials for ML)
Gera um SBOM completo combinando:
  1. Inventário de dependências Python do ambiente (packages)
  2. ML Model Card (CycloneDX 1.6 machine-learning-model component)
     — algoritmo, features, métricas de performance, dataset provenance,
       considerações éticas e limitações conhecidas

Compatível com CycloneDX spec 1.6 (machine-learning-model + modelCard).
OWASP CycloneDX ML BOM: https://cyclonedx.org/capabilities/mlbom/
ISO/IEC 42001 — AI Management System (rastreabilidade de modelos)

MITRE ATLAS: AML.T0010 Supply Chain Compromise (artifact provenance)

Uso:
    python scripts/generate_mlbom.py \\
        --model model/rf_model.pkl \\
        --meta  model/feature_names.json \\
        --metrics results/train_metrics_latest.json \\
        --output results/sbom_ml.cdx.json
"""

import argparse
import hashlib
import json
import logging
import pathlib
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def get_installed_packages() -> list[dict[str, str]]:
    """Retorna lista de pacotes instalados via pip list --format=json."""
    try:
        out = subprocess.check_output(
            [sys.executable, "-m", "pip", "list", "--format=json"],
            stderr=subprocess.DEVNULL,
        )
        return json.loads(out)
    except Exception as exc:
        log.warning(f"  pip list falhou: {exc}")
        return []


def load_json_safe(path: pathlib.Path | None) -> dict:
    if not path or not path.exists():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def load_metrics(metrics_path: pathlib.Path | None) -> dict[str, float]:
    """
    Lê métricas do modelo. Aceita:
      - JSON com chaves diretas: {"roc_auc": 0.89, "f1_weighted": 0.85}
      - JSON aninhado de MLflow artifacts
    """
    raw = load_json_safe(metrics_path)
    if not raw:
        return {}
    # Extrai apenas valores numéricos no primeiro nível
    return {k: float(v) for k, v in raw.items() if isinstance(v, (int, float))}


# ---------------------------------------------------------------------------
# Package components (environment inventory)
# ---------------------------------------------------------------------------

def build_package_components(packages: list[dict]) -> list[dict[str, Any]]:
    components = []
    for pkg in packages:
        name = pkg.get("name", "")
        version = pkg.get("version", "")
        components.append({
            "type": "library",
            "bom-ref": f"pkg:pypi/{name.lower()}@{version}",
            "name": name,
            "version": version,
            "purl": f"pkg:pypi/{name.lower()}@{version}",
        })
    return components


# ---------------------------------------------------------------------------
# ML Model Card component
# ---------------------------------------------------------------------------

def build_model_component(args: argparse.Namespace,
                          metrics: dict[str, float],
                          meta: dict) -> dict[str, Any]:
    """
    Constrói o componente CycloneDX 1.6 'machine-learning-model' com model card.
    """
    model_path = pathlib.Path(args.model) if args.model else None
    model_hash = sha256_file(model_path) if model_path and model_path.exists() else None

    # ── Features ─────────────────────────────────────────────────────────────
    cat_features = meta.get("cat_features", [])
    num_features = meta.get("num_features", [])
    all_features = meta.get("feature_names", cat_features + num_features)
    target = meta.get("target", "is_risky")

    inputs = [
        {
            "format": "tabular",
            "properties": [
                {"name": "feature_count", "value": str(len(all_features))},
                {"name": "categorical_features", "value": ", ".join(cat_features)},
                {"name": "numerical_features",   "value": ", ".join(num_features)},
            ],
        }
    ]
    outputs = [
        {
            "format": "prediction",
            "properties": [
                {"name": "task",        "value": "binary-classification"},
                {"name": "target",      "value": target},
                {"name": "output_type", "value": "probability + label"},
            ],
        }
    ]

    # ── Performance metrics ───────────────────────────────────────────────────
    METRIC_MAP = {
        "roc_auc":           "roc-auc",
        "f1_weighted":       "f1",
        "accuracy":          "accuracy",
        "precision_weighted": "precision",
        "recall_weighted":   "recall",
        "cv_roc_auc_mean":   "cross-val-roc-auc-mean",
        "cv_roc_auc_std":    "cross-val-roc-auc-std",
        "oob_score":         "oob-score",
    }
    performance_metrics = []
    for key, cdx_type in METRIC_MAP.items():
        if key in metrics:
            performance_metrics.append({
                "type":  cdx_type,
                "value": f"{metrics[key]:.4f}",
            })

    # ── Model parameters ──────────────────────────────────────────────────────
    rf_params = {k: v for k, v in meta.items()
                 if k.startswith("n_estimators") or k in
                 {"max_depth", "min_samples_leaf", "class_weight", "random_state",
                  "n_estimators", "kfold"}}

    component: dict[str, Any] = {
        "type": "machine-learning-model",
        "bom-ref": "rf-findings-classifier",
        "name": args.name,
        "version": args.model_version,
        "description": args.description,
        "modelCard": {
            "modelParameters": {
                "task":                 "classification",
                "architectureFamily":   "tree-based-ensemble",
                "modelArchitecture":    "RandomForestClassifier (scikit-learn)",
                "datasets": [
                    {
                        "type": "training",
                        "name": args.dataset_name,
                        "description": args.dataset_description,
                        "governance": {
                            "owners": [{"contact": {"name": "MLSecOps Pipeline"}}],
                        },
                    }
                ],
                "inputs":  inputs,
                "outputs": outputs,
            },
            "quantitativeAnalysis": {
                "performanceMetrics": performance_metrics,
            },
            "considerations": {
                "users": [args.intended_users],
                "useCases": [args.use_case],
                "technicalLimitations": [
                    "Trained on synthetic CI data when real dataset is unavailable; "
                    "re-train with production data before deployment.",
                    "Random Forest does not generalize to distribution shifts "
                    "beyond the training domain.",
                    "Adversarial robustness evaluated via black-box attacks only "
                    "(ZooAttack / HopSkipJump); gradient-based bounds not available.",
                ],
                "performanceTradeoffs": [
                    "class_weight=balanced compensates for label imbalance "
                    "but may reduce precision on majority class.",
                ],
                "ethicalConsiderations": [
                    {
                        "name": "Bias in SAST findings",
                        "description": (
                            "Model may reflect biases in historical annotation. "
                            "Regularly audit predictions across severity/confidence strata."
                        ),
                    }
                ],
            },
        },
    }

    # Adiciona hash do arquivo do modelo se disponível
    if model_hash:
        component["hashes"] = [{"alg": "SHA-256", "content": model_hash}]

    # Parâmetros do RF como properties
    if rf_params:
        component["properties"] = [
            {"name": k, "value": str(v)} for k, v in rf_params.items()
        ]

    return component


# ---------------------------------------------------------------------------
# Assemble full BOM
# ---------------------------------------------------------------------------

def build_bom(args: argparse.Namespace) -> dict[str, Any]:
    log.info("=== Generating CycloneDX ML BOM ===")

    # Load supporting data
    meta    = load_json_safe(pathlib.Path(args.meta) if args.meta else None)
    metrics = load_metrics(pathlib.Path(args.metrics) if args.metrics else None)
    if metrics:
        log.info(f"  Metrics loaded: {list(metrics.keys())}")
    else:
        log.warning("  No metrics file found — model card will have empty performance section.")

    # Environment packages
    log.info("  Scanning installed packages…")
    packages = get_installed_packages()
    log.info(f"  Found {len(packages)} packages")

    pkg_components   = build_package_components(packages)
    model_component  = build_model_component(args, metrics, meta)

    bom: dict[str, Any] = {
        "bomFormat":    "CycloneDX",
        "specVersion":  "1.6",
        "serialNumber": f"urn:uuid:{uuid.uuid4()}",
        "version":      1,
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tools": [
                {"vendor": "MLSecOps", "name": "generate_mlbom.py", "version": "1.0"},
            ],
            "component": {
                "type":    "application",
                "name":    "mlsecops-pipeline",
                "version": "1.0",
            },
        },
        "components": [model_component] + pkg_components,
    }
    return bom


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate CycloneDX ML BOM with model card (spec 1.6)"
    )
    parser.add_argument("--model",       default=None,
                        help="Trained model file (.pkl, .joblib, …)")
    parser.add_argument("--meta",        default=None,
                        help="feature_names.json from training")
    parser.add_argument("--metrics",     default=None,
                        help="JSON file with model metrics (roc_auc, f1_weighted, …)")
    parser.add_argument("--output",      default="results/sbom_ml.cdx.json",
                        help="Output BOM path (default: results/sbom_ml.cdx.json)")
    parser.add_argument("--name",        default="rf_findings_classifier",
                        help="Model component name")
    parser.add_argument("--model-version", default="1.0.0")
    parser.add_argument("--description",
                        default="Random Forest classifier for SAST findings risk classification")
    parser.add_argument("--dataset-name",
                        default="all_findings_flat.csv")
    parser.add_argument("--dataset-description",
                        default="SAST findings generated by Bandit on Python repositories; "
                                "binary target is_risky annotated by security team.")
    parser.add_argument("--intended-users", default="security-analysts")
    parser.add_argument("--use-case",
                        default="Classify SAST findings as risky or not-risky to prioritise "
                                "manual review effort in a CI/CD security pipeline.")
    args = parser.parse_args()

    bom = build_bom(args)

    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(bom, f, indent=2)

    n_pkgs = len(bom["components"]) - 1  # minus the model component
    log.info(f"\nML BOM written: {out}")
    log.info(f"  Components  : 1 ML model + {n_pkgs} packages")
    log.info(f"  Spec version: {bom['specVersion']}")
    log.info(f"  Serial      : {bom['serialNumber']}")


if __name__ == "__main__":
    main()
