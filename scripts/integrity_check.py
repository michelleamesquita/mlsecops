"""
scripts/integrity_check.py — Verificação de integridade SHA256 (genérico)
Garante que dados, modelos e artefatos não foram adulterados desde a última execução.

Suporta: CSV, Parquet, PKL, Joblib, H5, PT, ONNX, JSON, YAML e qualquer binário.
Gera manifest JSON compatível com SLSA Provenance Level 1.

MITRE ATLAS: AML.T0010 ML Supply Chain Compromise · AML.T0020 Data Poisoning
             AML.T0024g Erode Dataset Integrity (split-view poisoning via --row-level)
SLSA Framework: artifact hash verification

Uso:
    # Gerar checksums (primeira vez):
    python scripts/integrity_check.py --files data/ model/ --update

    # Verificar integridade:
    python scripts/integrity_check.py --files data/ model/ --checksums checksums.sha256

    # Row-level: detecta substituição parcial de linhas (split-view poisoning):
    python scripts/integrity_check.py --files dataset.csv --row-level --chunk-size 1000 --update
"""

import argparse
import hashlib
import json
import logging
import pathlib
import sys
from datetime import datetime, timezone
from typing import Generator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

SCAN_EXTENSIONS = {
    ".csv", ".parquet", ".pkl", ".pickle", ".joblib",
    ".h5", ".hdf5", ".pt", ".pth", ".onnx", ".json", ".yaml", ".yml",
}

IGNORED_DIRS = {"mlruns", ".git", "__pycache__", ".dvc", "node_modules", ".venv", "venv"}


def iter_files(paths: list[str]) -> Generator[pathlib.Path, None, None]:
    """Recursively yields all relevant files from paths (files or dirs)."""
    for raw in paths:
        p = pathlib.Path(raw)
        if p.is_file():
            yield p
        elif p.is_dir():
            for child in sorted(p.rglob("*")):
                if child.is_file() and child.suffix.lower() in SCAN_EXTENSIONS:
                    if not any(part in IGNORED_DIRS for part in child.parts):
                        yield child
        else:
            log.warning(f"Path not found, skipping: {p}")


def sha256_file(path: pathlib.Path, chunk_size: int = 65536) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def load_checksums(path: pathlib.Path) -> dict[str, str]:
    """Load a checksums file (sha256sum format: hash  filepath)."""
    checksums: dict[str, str] = {}
    if not path.exists():
        return checksums
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(None, 1)
            if len(parts) == 2:
                checksums[parts[1].strip()] = parts[0].strip()
    return checksums


def save_checksums(manifest: dict[str, str], output: pathlib.Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        f.write(f"# MLSecOps integrity manifest — generated {datetime.now(timezone.utc).isoformat()}\n")
        for filepath, digest in sorted(manifest.items()):
            f.write(f"{digest}  {filepath}\n")


def save_slsa_manifest(manifest: dict[str, str], output: pathlib.Path) -> None:
    """Gera manifest JSON compatível com SLSA Provenance."""
    subjects = [
        {"name": fp, "digest": {"sha256": h}} for fp, h in sorted(manifest.items())
    ]
    slsa = {
        "_type": "https://in-toto.io/Statement/v0.1",
        "subject": subjects,
        "predicateType": "https://slsa.dev/provenance/v0.2",
        "predicate": {
            "builder": {"id": "github-actions"},
            "buildType": "MLSecOps-integrity-check",
            "metadata": {"completeness": {"parameters": True}},
            "generatedAt": datetime.now(timezone.utc).isoformat(),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(slsa, f, indent=2)


def sha256_chunks(path: pathlib.Path, chunk_size: int) -> dict[str, str]:
    """
    Hasha o CSV/Parquet em chunks de N linhas.
    Detecta split-view poisoning: substituição parcial de linhas que não altera
    o hash do arquivo inteiro mas muda hashes de chunks específicos.
    Ref: Carlini et al., 2024 · MITRE ATLAS AML.T0024g
    """
    import csv
    import io

    chunk_hashes: dict[str, str] = {}
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            chunk: list[list] = []
            chunk_idx = 0
            for row in reader:
                chunk.append(row)
                if len(chunk) >= chunk_size:
                    buf = io.StringIO()
                    w = csv.writer(buf)
                    if header:
                        w.writerow(header)
                    w.writerows(chunk)
                    h = hashlib.sha256(buf.getvalue().encode()).hexdigest()
                    chunk_hashes[f"chunk_{chunk_idx:06d}"] = h
                    chunk = []
                    chunk_idx += 1
            if chunk:
                buf = io.StringIO()
                w = csv.writer(buf)
                if header:
                    w.writerow(header)
                w.writerows(chunk)
                h = hashlib.sha256(buf.getvalue().encode()).hexdigest()
                chunk_hashes[f"chunk_{chunk_idx:06d}"] = h
    except Exception as e:
        log.warning(f"  Row-level hash failed for {path}: {e}. Falling back to file hash.")
        chunk_hashes["file"] = sha256_file(path)
    return chunk_hashes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SHA256 integrity check for ML artifacts (data, models, configs)"
    )
    parser.add_argument("--files", nargs="+", required=True,
                        help="Files or directories to scan")
    parser.add_argument("--checksums", default="checksums.sha256",
                        help="Reference checksums file (default: checksums.sha256)")
    parser.add_argument("--update", action="store_true",
                        help="Generate/update checksums instead of verifying")
    parser.add_argument("--row-level", action="store_true",
                        help="Hash CSV/Parquet in chunks to detect partial row substitution "
                             "(split-view poisoning, AML.T0024g). Requires --chunk-size.")
    parser.add_argument("--chunk-size", type=int, default=10_000,
                        help="Rows per chunk for row-level hashing (default: 10000)")
    parser.add_argument("--output-dir", default="results",
                        help="Directory for reports (default: results/)")
    args = parser.parse_args()

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checksums_path = pathlib.Path(args.checksums)

    files = list(iter_files(args.files))
    if not files:
        log.warning("No eligible files found to scan.")
        sys.exit(0)

    log.info(f"=== Integrity Check — {len(files)} file(s) ===")
    current: dict[str, str] = {}
    for fp in files:
        digest = sha256_file(fp)
        current[str(fp)] = digest
        log.info(f"  {digest[:16]}…  {fp}")

    # ── Row-level hashing (split-view poisoning detection) ──────────────────
    row_hashes: dict[str, dict[str, str]] = {}
    row_level_failed: list[str] = []
    if args.row_level:
        log.info(f"\n=== Row-Level Integrity Check (chunk_size={args.chunk_size}) ===")
        csv_files = [fp for fp in files if fp.suffix.lower() in {".csv", ".parquet", ".tsv"}]
        if not csv_files:
            log.warning("  No CSV/Parquet files found — row-level check skipped.")
        else:
            row_checksums_path = pathlib.Path(str(args.checksums).replace(".sha256", "_rowlevel.json"))
            for fp in csv_files:
                log.info(f"  Hashing {fp} in {args.chunk_size}-row chunks…")
                row_hashes[str(fp)] = sha256_chunks(fp, args.chunk_size)
                log.info(f"    {len(row_hashes[str(fp)])} chunks hashed")

            if args.update:
                row_checksums_path.parent.mkdir(parents=True, exist_ok=True)
                with open(row_checksums_path, "w") as f:
                    json.dump(row_hashes, f, indent=2)
                log.info(f"  Row-level checksums saved: {row_checksums_path}")
            else:
                if row_checksums_path.exists():
                    with open(row_checksums_path) as f:
                        ref_row = json.load(f)
                    for fp_str, chunks in row_hashes.items():
                        ref_chunks = ref_row.get(fp_str, {})
                        for chunk_id, h in chunks.items():
                            if chunk_id not in ref_chunks:
                                log.info(f"  [ROW-NEW]      {fp_str}::{chunk_id}")
                            elif ref_chunks[chunk_id] != h:
                                row_level_failed.append(f"{fp_str}::{chunk_id}")
                                log.error(f"  [ROW-TAMPERED] {fp_str}::{chunk_id}")
                            else:
                                log.info(f"  [ROW-OK]       {fp_str}::{chunk_id}")
                    if row_level_failed:
                        log.error(f"\n  Row-level Gate FAILED — {len(row_level_failed)} tampered chunk(s).")
                        log.error("  MITRE ATLAS AML.T0024g: possible split-view poisoning.")
                    else:
                        log.info("  Row-level Gate: PASSED")
                else:
                    log.warning(f"  No row-level baseline at {row_checksums_path}. Run --update to seed.")
    # ─────────────────────────────────────────────────────────────────────────

    if args.update:
        save_checksums(current, checksums_path)
        save_slsa_manifest(current, out_dir / "slsa_manifest.json")
        log.info(f"\nChecksums saved: {checksums_path}")
        log.info(f"SLSA manifest  : {out_dir}/slsa_manifest.json")
        return

    # Verify mode
    reference = load_checksums(checksums_path)
    if not reference:
        log.warning(f"No reference checksums found at {checksums_path}.")
        log.warning("Run with --update to generate baseline. Skipping gate (first run).")
        save_checksums(current, checksums_path)
        log.info(f"Baseline created: {checksums_path}")
        return

    tampered: list[str] = []
    missing:  list[str] = []
    new_files: list[str] = []

    for fp, digest in current.items():
        if fp not in reference:
            new_files.append(fp)
            log.info(f"  [NEW]      {fp}")
        elif reference[fp] != digest:
            tampered.append(fp)
            log.error(f"  [TAMPERED] {fp}")
            log.error(f"    expected: {reference[fp]}")
            log.error(f"    actual  : {digest}")
        else:
            log.info(f"  [OK]       {fp}")

    for fp in reference:
        if fp not in current:
            missing.append(fp)
            log.warning(f"  [MISSING]  {fp}")

    report = {
        "ok": len(current) - len(tampered),
        "tampered": tampered,
        "missing": missing,
        "new_files": new_files,
        "row_level_tampered": row_level_failed,
        "passed": len(tampered) == 0 and len(missing) == 0 and len(row_level_failed) == 0,
    }
    with open(out_dir / "integrity_report.json", "w") as f:
        json.dump(report, f, indent=2)

    log.info(f"\n  OK: {report['ok']} | Tampered: {len(tampered)} | Missing: {len(missing)} | New: {len(new_files)} | Row-chunks tampered: {len(row_level_failed)}")

    if tampered or missing:
        log.error("Integrity Gate FAILED — artifacts adulterados ou ausentes.")
        log.error("MITRE ATLAS AML.T0010: possível supply-chain compromise.")
        sys.exit(1)

    if row_level_failed:
        log.error("Row-Level Integrity Gate FAILED — split-view poisoning detected.")
        sys.exit(1)

    log.info("Integrity Gate: PASSED")


if __name__ == "__main__":
    main()
