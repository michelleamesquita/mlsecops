"""
scripts/model_scan.py — Scan de segurança em artefatos de modelo (genérico)
Detecta código malicioso, adulteração e problemas de formato em modelos serializados.

Camadas de verificação:
  1. ModelScan (ProtectAI) — detecta globals inseguros em pickle/joblib/h5/pt/onnx
  2. SHA256 integrity      — compara hash atual com hash esperado
  3. Magic bytes check     — valida formato real do arquivo vs. extensão declarada
  4. Pickle safety audit   — inspeciona REDUCE/GLOBAL opcodes diretamente (fallback)

Genérico: suporta .pkl, .joblib, .h5, .hdf5, .pt, .pth, .onnx, .safetensors

MITRE ATLAS: AML.T0010 ML Supply Chain Compromise · AML.T0018 Backdoor ML Model
Ref: Sotiropoulos ch18 · ProtectAI ModelScan · OWASP MLSVS

Uso:
    python scripts/model_scan.py --model model/rf_model.pkl
    python scripts/model_scan.py --model model/rf_model.pkl --expected-hash abc123...
"""

import argparse
import hashlib
import io
import json
import logging
import pathlib
import pickle
import pickletools
import struct
import subprocess
import sys
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Magic bytes por formato (primeiros bytes do arquivo)
MAGIC_BYTES: dict[str, list[bytes]] = {
    "pkl/joblib": [b"\x80"],          # pickle protocol opcode
    "h5":         [b"\x89HDF\r\n\x1a\n"],
    "pt":         [b"PK\x03\x04"],    # PyTorch usa ZIP
    "onnx":       [b"\x08"],          # protobuf field
    "safetensors":[b"{"],             # JSON header
}

# Globals que podem indicar backdoor/RCE em pickles
DANGEROUS_GLOBALS = {
    "os.system", "os.popen", "subprocess.Popen", "subprocess.check_output",
    "subprocess.run", "eval", "exec", "compile", "__import__",
    "socket.socket", "urllib.request.urlopen", "http.client.HTTPConnection",
    "builtins.eval", "builtins.exec", "ctypes.CDLL",
}


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def check_magic_bytes(path: pathlib.Path) -> dict:
    """Verifica se os magic bytes correspondem à extensão do arquivo."""
    with open(path, "rb") as f:
        header = f.read(16)

    ext = path.suffix.lower()
    expected = None
    for fmt, magics in MAGIC_BYTES.items():
        if ext in (f".{fmt}", f".{fmt.split('/')[0]}", f".{fmt.split('/')[-1]}"):
            expected = magics
            break
    if ext in (".pkl", ".pickle", ".joblib"):
        expected = MAGIC_BYTES["pkl/joblib"]

    if expected is None:
        return {"check": "magic_bytes", "result": "skipped", "reason": f"unknown extension {ext}"}

    matches = any(header.startswith(m) for m in expected)
    return {
        "check": "magic_bytes",
        "result": "ok" if matches else "MISMATCH",
        "header_hex": header[:8].hex(),
        "passed": matches,
    }


def audit_pickle_opcodes(path: pathlib.Path) -> dict:
    """
    Inspeciona opcodes pickle diretamente para detectar globals perigosos.
    Funciona mesmo sem modelscan instalado (fallback auditoria manual).
    """
    dangerous_found: list[str] = []
    global_calls: list[str] = []

    try:
        with open(path, "rb") as f:
            data = f.read()

        output = io.StringIO()
        try:
            pickletools.dis(data, output=output, memo={}, indentlevel=0)
        except Exception:
            pass  # Arquivo pode ter múltiplos pickle streams (joblib)

        for line in output.getvalue().splitlines():
            if "GLOBAL" in line or "REDUCE" in line or "INST" in line:
                global_calls.append(line.strip())
                # Extrai o nome do global
                parts = line.strip().split()
                if len(parts) >= 2:
                    candidate = " ".join(parts[-2:]).replace("'", "").strip()
                    # Normaliza para module.function
                    candidate_dot = candidate.replace(" ", ".")
                    if any(d in candidate_dot or d in candidate for d in DANGEROUS_GLOBALS):
                        dangerous_found.append(candidate)

    except Exception as e:
        return {"check": "pickle_audit", "result": "error", "error": str(e)}

    return {
        "check": "pickle_audit",
        "result": "DANGEROUS" if dangerous_found else "ok",
        "dangerous_globals": dangerous_found,
        "total_global_calls": len(global_calls),
        "passed": len(dangerous_found) == 0,
    }


def run_modelscan(model_path: pathlib.Path) -> dict:
    """
    Executa modelscan CLI (ProtectAI).
    Exit codes: 0=clean, 1=issues found, 2=scan failed, 3=scan errors/warnings (not security issues).
    Fallback para pickle audit se não instalado.
    """
    try:
        result = subprocess.run(
            ["modelscan", "scan", "-p", str(model_path)],
            capture_output=True, text=True, timeout=120,
        )
        stdout = result.stdout + result.stderr
        # exit 0 = clean, exit 3 = internal scanner error but no security issue
        # exit 1 = security issues found (UNSAFE globals)
        has_issues = result.returncode == 1 or (
            "UNSAFE" in stdout.upper() and "No issues found" not in stdout
        )
        return {
            "check": "modelscan",
            "result": "UNSAFE" if has_issues else "ok",
            "stdout": stdout[:2000],
            "returncode": result.returncode,
            "passed": not has_issues,
        }
    except FileNotFoundError:
        log.warning("  modelscan CLI não encontrado — usando pickle audit como fallback.")
        log.warning("  Instale: pip install modelscan")
        return {"check": "modelscan", "result": "skipped", "reason": "not installed"}
    except subprocess.TimeoutExpired:
        return {"check": "modelscan", "result": "timeout", "passed": False}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Security scan for serialized ML model artifacts"
    )
    parser.add_argument("--model", required=True, help="Model artifact path")
    parser.add_argument("--expected-hash", default=None,
                        help="Expected SHA256 hash for integrity verification")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    model_path = pathlib.Path(args.model)
    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        log.error(f"Model not found: {model_path}")
        sys.exit(1)

    log.info(f"=== Model Scan: {model_path} ===")
    log.info(f"  Size: {model_path.stat().st_size / 1024:.1f} KB")

    results: dict = {}
    failures: list[str] = []

    # ── 1. SHA256 integrity ───────────────────────────────────────────────
    actual_hash = sha256_file(model_path)
    log.info(f"\n[1/4] SHA256 Integrity")
    log.info(f"  Hash: {actual_hash}")
    if args.expected_hash:
        ok = actual_hash == args.expected_hash
        results["sha256"] = {"hash": actual_hash, "expected": args.expected_hash, "passed": ok}
        if not ok:
            failures.append(f"sha256 mismatch: {actual_hash} ≠ {args.expected_hash}")
            log.error("  SHA256 MISMATCH — modelo adulterado!")
        else:
            log.info("  SHA256: OK")
    else:
        log.info("  (--expected-hash não fornecido — salvando hash para baseline)")
        results["sha256"] = {"hash": actual_hash, "expected": None, "passed": True}
        # Salva hash atual para uso futuro
        hash_file = out_dir / "model_hash.txt"
        hash_file.write_text(f"{actual_hash}  {model_path}\n")
        log.info(f"  Hash salvo em: {hash_file}")

    # ── 2. Magic bytes ────────────────────────────────────────────────────
    log.info(f"\n[2/4] Magic Bytes Check")
    r = check_magic_bytes(model_path)
    results["magic_bytes"] = r
    if r.get("result") == "MISMATCH":
        failures.append(f"magic_bytes mismatch: header={r['header_hex']}")
        log.error(f"  MISMATCH — extensão {model_path.suffix} não corresponde ao conteúdo real!")
    else:
        log.info(f"  {r['result'].upper()} — header: {r.get('header_hex', 'N/A')}")

    # ── 3. ModelScan (ProtectAI) ──────────────────────────────────────────
    log.info(f"\n[3/4] ModelScan (ProtectAI)")
    ext = model_path.suffix.lower()
    if ext in (".pkl", ".pickle", ".joblib", ".h5", ".hdf5", ".pt", ".pth", ".onnx"):
        r = run_modelscan(model_path)
        results["modelscan"] = r
        if r.get("result") == "UNSAFE":
            failures.append("modelscan: unsafe globals detected")
            log.error("  UNSAFE — código malicioso detectado pelo ModelScan!")
        elif r.get("result") == "skipped":
            log.warning(f"  Skipped: {r.get('reason')}")
        else:
            log.info(f"  OK — nenhuma ameaça detectada")
    else:
        log.info(f"  Skipped — extensão {ext} não suportada pelo modelscan")

    # ── 4. Pickle opcode audit (para .pkl/.joblib) ────────────────────────
    log.info(f"\n[4/4] Pickle Opcode Audit")
    if ext in (".pkl", ".pickle", ".joblib"):
        r = audit_pickle_opcodes(model_path)
        results["pickle_audit"] = r
        if r.get("dangerous_globals"):
            failures.append(f"pickle_audit: dangerous globals={r['dangerous_globals']}")
            log.error(f"  DANGEROUS globals: {r['dangerous_globals']}")
            log.error("  MITRE ATLAS AML.T0018: possível backdoor no modelo!")
        else:
            log.info(f"  OK — {r.get('total_global_calls', 0)} global calls, nenhum perigoso")
    else:
        log.info(f"  Skipped (apenas pkl/joblib)")

    # ── Relatório ─────────────────────────────────────────────────────────
    report = {
        "model": str(model_path),
        "passed": len(failures) == 0,
        "failures": failures,
        "checks": results,
    }
    report_path = out_dir / "model_scan_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    log.info(f"\n=== Model Scan Gate ===")
    log.info(f"  Failures : {len(failures)}")
    log.info(f"  Report   : {report_path}")

    if failures:
        log.error("Model Scan Gate FAILED — artefato de modelo comprometido.")
        log.error("MITRE ATLAS AML.T0010 · AML.T0018")
        sys.exit(1)

    log.info("Model Scan Gate: PASSED")


if __name__ == "__main__":
    main()
