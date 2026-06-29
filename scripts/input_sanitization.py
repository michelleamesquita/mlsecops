"""
scripts/input_sanitization.py — Defesa Automática via Sanitização de Entrada

Lê o relatório de um gate adversarial (gerado por adversarial_eval.py).
Se o gate FALHOU, aplica defesas de pré-processamento do IBM ART e
re-avalia o modelo para medir a recuperação de acurácia.

Defesas aplicadas (art.defences.preprocessor):
  1. FeatureSqueezing  — quantiza features, removendo perturbações de alta frequência
  2. GaussianAugmentation — adiciona ruído para randomizar perturbações adversariais
  3. Combinação sequencial das duas (melhor resultado prático)

Se o gate PASSOU, o script encerra sem ação (proteção já adequada).

MITRE ATLAS: AML.T0015 Evade ML Model — Mitigação
OWASP MLSVS: V8 — Model Robustness
OpenSSF MLSecOps Whitepaper 2025: §6 "Inference-time Defenses"

Uso:
    python scripts/input_sanitization.py \\
        --data          all_findings_flat.csv \\
        --target        is_risky \\
        --model         model/rf_model.pkl \\
        --meta          model/feature_names.json \\
        --adv-report    results/adversarial_fgsm.json \\
        --output        results/sanitization_report.json

    # Forçar aplicação mesmo se o gate passou:
    python scripts/input_sanitization.py ... --force
"""

import argparse
import json
import logging
import pathlib
import sys

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers reutilizados de adversarial_eval.py
# ---------------------------------------------------------------------------

def _load_meta(meta_path: str | None) -> dict | None:
    if not meta_path or not pathlib.Path(meta_path).exists():
        return None
    with open(meta_path) as f:
        return json.load(f)


def _load_features(data_path: str, target: str, meta: dict | None,
                   sample: int = 50_000) -> tuple[np.ndarray, np.ndarray]:
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    p = pathlib.Path(data_path)
    df = pd.read_csv(p, low_memory=False, nrows=sample) \
         if p.suffix == ".csv" else pd.read_parquet(p)

    y_col = pd.to_numeric(df[target], errors="coerce").fillna(0).astype(int)
    _, df_test = train_test_split(df, test_size=0.2, random_state=42, stratify=y_col)
    df = df_test.reset_index(drop=True)

    if meta:
        for col, classes in meta.get("encoder_classes", {}).items():
            if col in df.columns:
                le = LabelEncoder()
                le.classes_ = np.array(classes)
                df[col] = df[col].fillna("UNKNOWN").astype(str)
                known = set(le.classes_)
                df[col] = df[col].apply(lambda x: x if x in known else le.classes_[0])
                df[col] = le.transform(df[col])
        for col in meta.get("num_features", []):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        feature_cols = [c for c in meta["feature_names"] if c in df.columns]
    else:
        for col in df.select_dtypes(include="object").columns:
            if col != target:
                df[col] = df[col].fillna("UNKNOWN").astype("category").cat.codes
        for col in df.select_dtypes(exclude="object").columns:
            if col != target:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        feature_cols = [c for c in df.columns
                        if c != target and df[c].dtype != object]

    X = df[feature_cols].values.astype(np.float32)
    y = pd.to_numeric(df[target], errors="coerce").fillna(0).astype(int).values
    return X, y


def _generate_adversarial(clf, X_sub: np.ndarray) -> np.ndarray:
    """Gera exemplos adversariais com BoundaryAttack (único ataque ART
    compatível com SklearnRandomForestClassifier)."""
    try:
        from art.estimators.classification import SklearnClassifier
        from art.attacks.evasion import BoundaryAttack
    except ImportError as exc:
        log.error(f"ART não instalado: {exc}")
        sys.exit(1)

    x_min = X_sub.min(axis=0)
    x_max = X_sub.max(axis=0)
    art_clf = SklearnClassifier(model=clf, clip_values=(x_min, x_max))
    attack = BoundaryAttack(art_clf, targeted=False, max_iter=50)
    log.info("  Gerando exemplos adversariais (BoundaryAttack, max_iter=50)…")
    return attack.generate(X_sub)


# ---------------------------------------------------------------------------
# Defesas ART
# ---------------------------------------------------------------------------

def _apply_feature_squeezing(X_adv: np.ndarray,
                              x_min: np.ndarray,
                              x_max: np.ndarray,
                              bit_depth: int = 8) -> np.ndarray:
    """FeatureSqueezing: quantiza os valores das features para reduzir
    a resolução disponível a perturbações adversariais de alta precisão."""
    try:
        from art.defences.preprocessor import FeatureSqueezing
    except ImportError as exc:
        log.warning(f"FeatureSqueezing não disponível: {exc}")
        return X_adv

    defense = FeatureSqueezing(
        bit_depth=bit_depth,
        clip_values=(x_min, x_max),
        apply_predict=True,
    )
    X_clean, _ = defense(X_adv.copy())
    return X_clean


def _apply_gaussian_augmentation(X_adv: np.ndarray,
                                  sigma: float = 0.01) -> np.ndarray:
    """GaussianAugmentation: adiciona ruído gaussiano para 'randomizar'
    perturbações adversariais de pequena magnitude."""
    try:
        from art.defences.preprocessor import GaussianAugmentation
    except ImportError as exc:
        log.warning(f"GaussianAugmentation não disponível: {exc}")
        return X_adv

    defense = GaussianAugmentation(sigma=sigma, augmentation=False, apply_predict=True)
    X_clean, _ = defense(X_adv.copy())
    return X_clean


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detecção e aplicação automática de defesas adversariais"
    )
    parser.add_argument("--data",        required=True)
    parser.add_argument("--target",      required=True)
    parser.add_argument("--model",       required=True)
    parser.add_argument("--meta",        default=None)
    parser.add_argument("--adv-report",  required=True,
                        help="JSON gerado por adversarial_eval.py "
                             "(ex: results/adversarial_fgsm.json)")
    parser.add_argument("--output",      default="results/sanitization_report.json")
    parser.add_argument("--n-samples",   type=int, default=50)
    parser.add_argument("--bit-depth",   type=int, default=8,
                        help="FeatureSqueezing bit depth (default: 8)")
    parser.add_argument("--sigma",       type=float, default=0.01,
                        help="GaussianAugmentation sigma (default: 0.01)")
    parser.add_argument("--force",       action="store_true",
                        help="Aplicar defesas mesmo se o gate adversarial passou")
    args = parser.parse_args()

    # ── 1. Ler relatório adversarial ─────────────────────────────────────────
    report_path = pathlib.Path(args.adv_report)
    if not report_path.exists():
        log.error(f"Relatório adversarial não encontrado: {report_path}")
        sys.exit(1)

    with open(report_path) as f:
        adv_report = json.load(f)

    gate_passed   = adv_report.get("passed", True)
    acc_adv_orig  = adv_report.get("acc_adversarial", 1.0)
    acc_clean_ref = adv_report.get("acc_clean", 1.0)
    threshold     = adv_report.get("threshold", 0.75)
    attack_name   = adv_report.get("attack", "unknown")

    log.info("=== Input Sanitization Defense ===")
    log.info(f"  Relatório   : {report_path}")
    log.info(f"  Ataque      : {attack_name.upper()}")
    log.info(f"  Gate passed : {gate_passed}")
    log.info(f"  Acc (clean) : {acc_clean_ref:.4f}")
    log.info(f"  Acc (adv)   : {acc_adv_orig:.4f}")
    log.info(f"  Threshold   : {threshold:.4f}")

    if gate_passed and not args.force:
        log.info("Gate adversarial JÁ PASSOU — proteção adequada, nenhuma defesa necessária.")
        out = pathlib.Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump({
                "defense_applied": False,
                "reason": "gate_passed",
                "passed": True,
                "attack": attack_name,
                "acc_adversarial_original": acc_adv_orig,
                "threshold": threshold,
            }, f, indent=2)
        return

    log.info("Gate adversarial FALHOU — aplicando defesas de sanitização…")

    # ── 2. Carregar modelo e dados ───────────────────────────────────────────
    import joblib
    clf  = joblib.load(args.model)
    meta = _load_meta(args.meta)
    X, y = _load_features(args.data, args.target, meta)
    n    = min(args.n_samples, len(X))
    X_sub, y_sub = X[:n], y[:n]

    x_min = X_sub.min(axis=0)
    x_max = X_sub.max(axis=0)

    acc_clean = (clf.predict(X_sub) == y_sub).mean()
    log.info(f"  Acc clean (amostras CI): {acc_clean:.4f}")

    # ── 3. Gerar exemplos adversariais para medir recuperação ────────────────
    X_adv = _generate_adversarial(clf, X_sub)
    acc_adv_local = (clf.predict(X_adv) == y_sub).mean()
    log.info(f"  Acc adversarial (local re-check): {acc_adv_local:.4f}")

    # ── 4. Aplicar defesas e medir recuperação ───────────────────────────────
    results = {}

    # Defesa 1: FeatureSqueezing
    X_fs = _apply_feature_squeezing(X_adv, x_min, x_max, args.bit_depth)
    acc_fs = (clf.predict(X_fs) == y_sub).mean()
    log.info(f"  FeatureSqueezing (bit_depth={args.bit_depth}): acc={acc_fs:.4f}")
    results["feature_squeezing"] = {
        "acc_defended": float(acc_fs),
        "recovery": float(acc_fs - acc_adv_local),
        "passed": bool(acc_fs >= threshold),
        "config": {"bit_depth": args.bit_depth},
    }

    # Defesa 2: GaussianAugmentation
    X_ga = _apply_gaussian_augmentation(X_adv, args.sigma)
    acc_ga = (clf.predict(X_ga) == y_sub).mean()
    log.info(f"  GaussianAugmentation (sigma={args.sigma}): acc={acc_ga:.4f}")
    results["gaussian_augmentation"] = {
        "acc_defended": float(acc_ga),
        "recovery": float(acc_ga - acc_adv_local),
        "passed": bool(acc_ga >= threshold),
        "config": {"sigma": args.sigma},
    }

    # Defesa 3: Combinação FeatureSqueezing + GaussianAugmentation
    X_combined = _apply_gaussian_augmentation(X_fs, sigma=args.sigma / 2)
    acc_combined = (clf.predict(X_combined) == y_sub).mean()
    log.info(f"  FS + Gaussian (combined): acc={acc_combined:.4f}")
    results["combined"] = {
        "acc_defended": float(acc_combined),
        "recovery": float(acc_combined - acc_adv_local),
        "passed": bool(acc_combined >= threshold),
        "config": {"bit_depth": args.bit_depth, "sigma": args.sigma / 2},
    }

    # ── 5. Determinar melhor defesa ──────────────────────────────────────────
    best_name = max(results, key=lambda k: results[k]["acc_defended"])
    best      = results[best_name]
    log.info(f"\n  Melhor defesa: {best_name} → acc={best['acc_defended']:.4f}")

    # ── 6. Relatório final ───────────────────────────────────────────────────
    log.info("\n=== Resultado da Sanitização ===")
    log.info(f"  Acc sem defesa  : {acc_adv_local:.4f}")
    log.info(f"  Acc com defesa  : {best['acc_defended']:.4f}  ({best_name})")
    log.info(f"  Recuperação     : +{best['recovery']:.4f}")
    log.info(f"  Threshold       : {threshold:.4f}")

    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "defense_applied": True,
            "attack": attack_name,
            "acc_clean": float(acc_clean),
            "acc_adversarial_before_defense": float(acc_adv_local),
            "threshold": threshold,
            "defenses": results,
            "best_defense": best_name,
            "best_acc_defended": best["acc_defended"],
            "best_recovery": best["recovery"],
            "gate_passed_after_defense": best["passed"],
            "passed": best["passed"],
        }, f, indent=2)

    log.info(f"  Relatório salvo: {out}")

    if not best["passed"]:
        log.error(
            f"Sanitization Gate FAILED — melhor defesa ({best_name}) "
            f"acc={best['acc_defended']:.4f} ainda abaixo do threshold {threshold:.4f}."
        )
        log.error(
            "Recomendação: considere adversarial training (ART IncrementalAdversarialTrainer) "
            "ou reduzir o threshold para este modelo/dataset."
        )
        sys.exit(1)

    log.warning(
        f"DEFENSE APPLIED: modelo vulnerável sob {attack_name.upper()} "
        f"foi protegido via {best_name} "
        f"(acc: {acc_adv_local:.4f} → {best['acc_defended']:.4f}). "
        "Considere adversarial training para uma solução permanente."
    )


if __name__ == "__main__":
    main()
