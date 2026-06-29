"""
scripts/compliance_check.py — Compliance Gate automatizado (InSpec para MLSecOps)
Lê policy.yml e verifica se todos os relatórios em results/ satisfazem os controles.

Equivalente ao Chef InSpec, mas para pipelines MLSecOps.
Nenhum modelo é promovido sem este gate passar.

OWASP MLSVS V10 — Compliance verification
Ref: DevSecOps Guides · AISP framework · Sotiropoulos ch18

Uso:
    python scripts/compliance_check.py --policy policy.yml --results results/
    python scripts/compliance_check.py --policy policy.yml --results results/ --strict
"""

import argparse
import json
import logging
import pathlib
import sys
from dataclasses import dataclass
from typing import Any

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ANSI_GREEN  = "\033[92m"
ANSI_YELLOW = "\033[93m"
ANSI_RED    = "\033[91m"
ANSI_RESET  = "\033[0m"
ANSI_BOLD   = "\033[1m"


@dataclass
class ControlResult:
    name: str
    stage: str
    blocking: bool
    passed: bool | None     # None = result file missing
    message: str
    atlas: str = ""
    mlsvs: str = ""


def color(text: str, code: str) -> str:
    return f"{code}{text}{ANSI_RESET}"


def load_result(result_file: pathlib.Path) -> dict | None:
    if not result_file.exists():
        return None
    try:
        with open(result_file) as f:
            if result_file.suffix == ".json":
                return json.load(f)
            return yaml.safe_load(f)
    except Exception as e:
        log.warning(f"  Could not parse {result_file}: {e}")
        return None


def evaluate_control(name: str, stage: str, spec: dict,
                     results_dir: pathlib.Path) -> ControlResult:
    """Avalia um controle individual contra seu arquivo de resultado."""
    result_path = results_dir / pathlib.Path(spec["result_file"]).name
    blocking = spec.get("blocking", True)
    # blocking_when_missing: permite que um controle bloqueante seja SKIP quando
    # o arquivo não existe — útil para controles de estágios upstream (data-validation,
    # secure-experiment) que não estão disponíveis no runner do adversarial-validation.
    blocking_when_missing = spec.get("blocking_when_missing", blocking)
    atlas = spec.get("atlas", "")
    mlsvs = spec.get("mlsvs", "")
    note = spec.get("note", "")

    data = load_result(result_path)

    if data is None:
        msg = f"Result file not found: {result_path}"
        if blocking_when_missing:
            return ControlResult(name, stage, blocking, None, msg, atlas, mlsvs)
        else:
            skip_note = note or "verified by upstream pipeline stage"
            # Override blocking=False so the existing SKIP path handles it gracefully
            return ControlResult(name, stage, False, None,
                                 f"{msg} — SKIP ({skip_note})", atlas, mlsvs)

    check = spec.get("check")

    # ── Verificação de campo booleano ──────────────────────────────────────
    if "field" in spec and "expected" in spec:
        field = spec["field"]
        expected = spec["expected"]
        actual = data.get(field)
        passed = actual == expected

        # Verificação adicional de threshold
        if passed and "threshold_field" in spec:
            tf = spec["threshold_field"]
            val = data.get(tf)
            if val is not None:
                if "max_threshold" in spec and val > spec["max_threshold"]:
                    passed = False
                    msg = f"{field}={actual} BUT {tf}={val:.4f} > max={spec['max_threshold']}"
                    if note:
                        msg += f" | {note}"
                    return ControlResult(name, stage, blocking, passed, msg, atlas, mlsvs)
                if "min_threshold" in spec and val < spec["min_threshold"]:
                    passed = False
                    msg = f"{field}={actual} BUT {tf}={val:.4f} < min={spec['min_threshold']}"
                    if note:
                        msg += f" | {note}"
                    return ControlResult(name, stage, blocking, passed, msg, atlas, mlsvs)

        msg = f"{field}={actual}" + (f" | {note}" if note and not passed else "")
        return ControlResult(name, stage, blocking, passed, msg, atlas, mlsvs)

    # ── Verificação de existência / not_empty ─────────────────────────────
    if check == "not_empty":
        field = spec.get("field", "")
        val = data.get(field)
        passed = bool(val)
        return ControlResult(name, stage, blocking, passed,
                             f"{field} is {'present' if passed else 'EMPTY'}", atlas, mlsvs)

    if check == "exists":
        return ControlResult(name, stage, blocking, True,
                             "file exists", atlas, mlsvs)

    return ControlResult(name, stage, blocking, None,
                         "Unknown check spec", atlas, mlsvs)


def run_compliance(policy: dict, results_dir: pathlib.Path,
                   strict: bool) -> tuple[list[ControlResult], bool]:
    results: list[ControlResult] = []
    overall_passed = True

    # Itera sobre os estágios da policy
    stages_order = ["data_quality", "supply_chain", "adversarial"]
    for stage in stages_order:
        if stage not in policy:
            continue
        stage_label = stage.replace("_", " ").title()
        log.info(f"\n{'─'*60}")
        log.info(f"  Stage: {stage_label}")
        log.info(f"{'─'*60}")

        for control_name, spec in policy[stage].items():
            if not isinstance(spec, dict):
                continue

            r = evaluate_control(control_name, stage_label, spec, results_dir)
            results.append(r)

            if r.passed is True:
                icon = color("PASS", ANSI_GREEN)
                log.info(f"  [{icon}] {control_name:<30} {r.message}")
            elif r.passed is False:
                icon = color("FAIL", ANSI_RED) if r.blocking else color("WARN", ANSI_YELLOW)
                log.warning(f"  [{icon}] {control_name:<30} {r.message}")
                if r.atlas:
                    log.warning(f"        ATLAS: {r.atlas} | MLSVS: {r.mlsvs}")
                if r.blocking:
                    overall_passed = False
                elif strict:
                    overall_passed = False
            else:  # None — missing
                if r.blocking:
                    icon = color("MISS", ANSI_RED)
                    log.error(f"  [{icon}] {control_name:<30} {r.message}")
                    overall_passed = False
                else:
                    icon = color("SKIP", ANSI_YELLOW)
                    log.warning(f"  [{icon}] {control_name:<30} {r.message}")

    return results, overall_passed


def print_summary(results: list[ControlResult], overall_passed: bool) -> None:
    blocking_passed  = [r for r in results if r.blocking     and r.passed is True]
    blocking_failed  = [r for r in results if r.blocking     and r.passed is not True]
    warning_controls = [r for r in results if not r.blocking and r.passed is False]
    missing          = [r for r in results if r.passed is None and r.blocking]

    log.info(f"\n{'═'*60}")
    log.info(color("  COMPLIANCE GATE SUMMARY", ANSI_BOLD))
    log.info(f"{'═'*60}")
    log.info(f"  Blocking passed : {color(str(len(blocking_passed)), ANSI_GREEN)}")
    log.info(f"  Blocking failed : {color(str(len(blocking_failed)), ANSI_RED)}")
    log.info(f"  Warnings        : {color(str(len(warning_controls)), ANSI_YELLOW)}")
    log.info(f"  Missing results : {color(str(len(missing)), ANSI_RED)}")

    if blocking_failed or missing:
        log.error(f"\n  Failed/Missing controls (BLOCKING):")
        for r in blocking_failed + missing:
            log.error(f"    • {r.stage} / {r.name} — {r.message}")

    if warning_controls:
        log.warning(f"\n  Warning controls (non-blocking — calibre os thresholds):")
        for r in warning_controls:
            log.warning(f"    • {r.stage} / {r.name} — {r.message}")

    status = color("PASSED", ANSI_GREEN) if overall_passed else color("FAILED", ANSI_RED)
    log.info(f"\n  Overall Compliance Gate: {status}")
    log.info(f"{'═'*60}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MLSecOps Compliance Gate — InSpec for ML Pipelines"
    )
    parser.add_argument("--policy",  default="policy.yml",
                        help="Path to policy.yml (default: policy.yml)")
    parser.add_argument("--results", default="results",
                        help="Path to results directory (default: results/)")
    parser.add_argument("--strict",  action="store_true",
                        help="Treat non-blocking warnings as failures")
    parser.add_argument("--output",  default=None,
                        help="Save compliance report to JSON file")
    args = parser.parse_args()

    policy_path  = pathlib.Path(args.policy)
    results_dir  = pathlib.Path(args.results)

    if not policy_path.exists():
        log.error(f"Policy file not found: {policy_path}")
        sys.exit(1)

    with open(policy_path) as f:
        policy = yaml.safe_load(f)

    log.info(f"Policy  : {policy_path}  (v{policy.get('version', '?')})")
    log.info(f"Results : {results_dir}")
    log.info(f"Mode    : {'STRICT' if args.strict else 'STANDARD'}")

    results, overall_passed = run_compliance(policy, results_dir, args.strict)
    print_summary(results, overall_passed)

    if args.output:
        report = {
            "passed": overall_passed,
            "controls": [
                {"name": r.name, "stage": r.stage, "blocking": r.blocking,
                 "passed": r.passed, "message": r.message}
                for r in results
            ],
        }
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        log.info(f"\n  Report saved: {args.output}")

    if not overall_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
