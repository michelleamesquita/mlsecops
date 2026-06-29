"""
scripts/generate_ci_data.py — Gera dataset sintético para CI/CD
Cria um CSV com o mesmo schema do dataset real mas com dados fictícios.
Usado quando o dataset real não está disponível (ex: runners do GitHub Actions).

Uso:
    python scripts/generate_ci_data.py \
        --output all_findings_flat.csv \
        --rows 5000 \
        --seed 42

Para datasets com schema diferente, passe --schema com um JSON descrevendo
as colunas e seus tipos:
    python scripts/generate_ci_data.py \
        --schema schema.json --output meu_dataset.csv
"""

import argparse
import json
import logging
import pathlib
import sys

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Schema padrão: all_findings_flat.csv (SAST Bandit) ─────────────────────
# Reflecte a estrutura real do dataset para que os scripts de validação
# encontrem as colunas esperadas.
DEFAULT_SCHEMA = {
    "target": "is_risky",
    "columns": {
        "severity":           {"type": "int",   "min": 0, "max": 2},
        "confidence":         {"type": "int",   "min": 0, "max": 2},
        "line_number":        {"type": "int",   "min": 1, "max": 5000},
        "col_offset":         {"type": "int",   "min": 0, "max": 200},
        "end_col_offset":     {"type": "int",   "min": 0, "max": 200},
        "issue_cwe_id":       {"type": "int",   "min": 0, "max": 999},
        "test_id_encoded":    {"type": "int",   "min": 0, "max": 50},
        "filename_length":    {"type": "int",   "min": 5, "max": 200},
        "issue_text_length":  {"type": "int",   "min": 10, "max": 500},
        "is_risky":           {"type": "binary", "pos_rate": 0.30},
    },
}


def load_schema(schema_path: str | None) -> dict:
    if not schema_path:
        return DEFAULT_SCHEMA
    p = pathlib.Path(schema_path)
    if not p.exists():
        log.error(f"Schema file not found: {p}")
        sys.exit(1)
    with open(p) as f:
        return json.load(f)


def generate(schema: dict, n_rows: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data: dict[str, np.ndarray] = {}
    for col, spec in schema["columns"].items():
        dtype = spec.get("type", "float")
        if dtype == "binary":
            rate = spec.get("pos_rate", 0.5)
            data[col] = rng.choice([0, 1], size=n_rows, p=[1 - rate, rate])
        elif dtype == "int":
            lo, hi = spec.get("min", 0), spec.get("max", 100)
            data[col] = rng.integers(lo, hi + 1, size=n_rows)
        elif dtype == "float":
            lo, hi = spec.get("min", 0.0), spec.get("max", 1.0)
            data[col] = rng.uniform(lo, hi, size=n_rows)
        elif dtype == "categorical":
            cats = spec.get("categories", ["A", "B", "C"])
            weights = spec.get("weights", None)
            if weights:
                weights = np.array(weights, dtype=float)
                weights /= weights.sum()
            data[col] = rng.choice(cats, size=n_rows, p=weights)
        else:
            data[col] = np.zeros(n_rows)
    return pd.DataFrame(data)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic CI dataset with a given schema"
    )
    parser.add_argument("--output", required=True,
                        help="Output CSV path (e.g. all_findings_flat.csv)")
    parser.add_argument("--rows", type=int, default=5_000,
                        help="Number of rows to generate (default: 5000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--schema", default=None,
                        help="JSON schema file (optional; defaults to SAST Bandit schema)")
    args = parser.parse_args()

    schema = load_schema(args.schema)
    log.info(f"Generating synthetic dataset: {args.rows} rows, seed={args.seed}")
    df = generate(schema, args.rows, args.seed)

    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    target = schema.get("target", "label")
    log.info(f"  Written: {out}  shape={df.shape}")
    if target in df.columns:
        dist = df[target].value_counts(normalize=True).to_dict()
        log.info(f"  Target '{target}' distribution: { {k: f'{v:.1%}' for k,v in dist.items()} }")


if __name__ == "__main__":
    main()
