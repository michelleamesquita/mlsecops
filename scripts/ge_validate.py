"""
scripts/ge_validate.py — Schema + distribuição via Great Expectations (genérico)
Funciona com qualquer dataset tabular. Auto-detecta tipos de coluna se --schema não fornecido.

Usa contexto efêmero (sem projeto GE pré-existente) — compatível com qualquer CI.

MITRE ATLAS: AML.T0020 Data Poisoning

Uso:
    # Auto-detect schema:
    python scripts/ge_validate.py --data dataset.csv --target label

    # Com schema explícito (YAML/JSON):
    python scripts/ge_validate.py --data dataset.csv --target label --schema schema.yml
"""

import argparse
import json
import logging
import pathlib
import sys
from typing import Any

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def load_schema(schema_path: str | None) -> dict | None:
    if not schema_path:
        return None
    p = pathlib.Path(schema_path)
    if not p.exists():
        log.warning(f"Schema file not found: {p}")
        return None
    import yaml  # optional
    with open(p) as f:
        return yaml.safe_load(f) if p.suffix in (".yml", ".yaml") else json.load(f)


def auto_detect_schema(df: pd.DataFrame, target: str) -> dict:
    """Gera um schema mínimo a partir do próprio dataframe."""
    schema: dict[str, Any] = {
        "required_columns": list(df.columns),
        "target": target,
        "dtypes": {},
        "set_columns": {},   # colunas com cardinalidade baixa → conjunto esperado de valores
    }
    for col in df.columns:
        schema["dtypes"][col] = str(df[col].dtype)
        if df[col].dtype == object and df[col].nunique() < 50:
            schema["set_columns"][col] = sorted(df[col].dropna().unique().tolist())
    return schema


def run_expectations(df: pd.DataFrame, target: str, schema: dict) -> list[dict]:
    results: list[dict] = []

    def check(name: str, passed: bool, msg: str) -> dict:
        icon = "PASSED" if passed else "FAILED"
        log.info(f"  [{icon}] {name}: {msg}")
        return {"expectation": name, "passed": passed, "message": msg}

    # Schema mínimo: colunas obrigatórias
    missing = [c for c in schema.get("required_columns", []) if c not in df.columns]
    results.append(check("expect_columns_to_exist",
                         len(missing) == 0,
                         f"missing: {missing}" if missing else "all columns present"))

    # Row count
    results.append(check("expect_table_row_count_to_be_between",
                         len(df) >= 100,
                         f"rows={len(df):,}"))

    # Target: sem nulos
    if target in df.columns:
        nulls = int(df[target].isna().sum())
        results.append(check(f"expect_column_values_to_not_be_null [{target}]",
                             nulls == 0,
                             f"null count={nulls}"))

        # Target: prevalência razoável (não completamente degenerado)
        target_numeric = pd.to_numeric(df[target], errors="coerce")
        if target_numeric.notna().all():
            rate = float(target_numeric.mean())
            results.append(check(f"expect_column_mean_to_be_between [{target}]",
                                 0.001 <= rate <= 0.999,
                                 f"mean={rate:.4f}"))

    # Set expectations para colunas de baixa cardinalidade
    for col, expected_vals in schema.get("set_columns", {}).items():
        if col not in df.columns:
            continue
        actual_vals = set(df[col].dropna().astype(str).unique())
        expected_set = set(str(v) for v in expected_vals)
        unexpected = actual_vals - expected_set
        results.append(check(f"expect_column_values_to_be_in_set [{col}]",
                             len(unexpected) == 0,
                             f"unexpected values: {unexpected}" if unexpected else f"all in {expected_set}"))

    # Numéricas: sem negativos em colunas de contagem (patch_lines, tokens, etc.)
    count_keywords = ["count", "lines", "tokens", "chars", "size", "len", "num"]
    for col in df.select_dtypes(include="number").columns:
        if any(kw in col.lower() for kw in count_keywords):
            neg_count = int((pd.to_numeric(df[col], errors="coerce") < 0).sum())
            if neg_count > 0:
                results.append(check(f"expect_column_values_to_be_between [{col} >= 0]",
                                     False, f"negative values: {neg_count}"))

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GE-style schema + distribution validation (generic)"
    )
    parser.add_argument("--data",    required=True)
    parser.add_argument("--target",  required=True, help="Target column name")
    parser.add_argument("--schema",  default=None,  help="Schema YAML/JSON (auto-detect if omitted)")
    parser.add_argument("--sample",  type=int, default=100_000)
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    path = pathlib.Path(args.data)
    if not path.exists():
        log.error(f"Dataset not found: {path}")
        sys.exit(1)

    log.info(f"Loading: {path} (max {args.sample:,} rows)")
    df = pd.read_csv(path, low_memory=False, nrows=args.sample) \
         if path.suffix == ".csv" else pd.read_parquet(path)
    log.info(f"  Shape: {df.shape}")

    schema = load_schema(args.schema) or auto_detect_schema(df, args.target)

    log.info("=== Great Expectations — Schema & Distribution Gate ===")
    results = run_expectations(df, args.target, schema)

    failed = [r for r in results if not r["passed"]]
    passed = [r for r in results if r["passed"]]
    log.info(f"\nSummary: {len(passed)} passed / {len(failed)} failed")

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "ge_report.json", "w") as f:
        json.dump({"passed": len(failed) == 0, "results": results}, f, indent=2)

    if failed:
        log.error("GE Gate FAILED")
        sys.exit(1)
    log.info("GE Gate: PASSED")


if __name__ == "__main__":
    main()
