"""
scripts/binary_input_detector.py — Detector de Inputs Adversariais (ART)

Treina um classificador binário para distinguir inputs limpos de adversariais
ANTES que cheguem ao modelo principal. Funciona como um "porteiro" na inferência.

Fluxo:
  1. Carrega modelo e dados
  2. Gera exemplos adversariais com BoundaryAttack
  3. Constrói dataset de treino: X_clean → label 0, X_adv → label 1
  4. Treina um detector (Logistic Regression)
  5. Envolve com art.defences.detector.evasion.BinaryInputDetector
  6. Avalia: detection rate, false positive rate, accuracy do RF com filtro
  7. Salva detector em model/ e relatório em results/

MITRE ATLAS: AML.T0015 Evade ML Model — Detecção
OWASP MLSVS: V8 Model Robustness — Runtime Input Validation
OpenSSF MLSecOps Whitepaper 2025: §6 "Inference-time Defenses"

Uso:
    python scripts/binary_input_detector.py \\
        --data    all_findings_flat.csv \\
        --target  is_risky \\
        --model   model/rf_model.pkl \\
        --meta    model/feature_names.json \\
        --output-dir results/

    # Modo verificação (CI): carrega detector salvo e testa no conjunto de teste
    python scripts/binary_input_detector.py \\
        --data    all_findings_flat.csv \\
        --target  is_risky \\
        --model   model/rf_model.pkl \\
        --detector model/binary_detector.pkl \\
        --verify
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
# Helpers
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
# Treinar detector
# ---------------------------------------------------------------------------

def train_detector(X_clean: np.ndarray, X_adv: np.ndarray,
                   clf_main) -> object:
    """
    Treina um classificador binário (Logistic Regression) para detectar
    inputs adversariais. Retorna o detector treinado (sklearn estimator).

    Label 0 = limpo, label 1 = adversarial
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    X_det = np.vstack([X_clean, X_adv]).astype(np.float32)
    y_det = np.array([0] * len(X_clean) + [1] * len(X_adv))

    detector_clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    cv_scores = cross_val_score(detector_clf, X_det, y_det, cv=3, scoring="roc_auc")
    log.info(f"  Detector CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    detector_clf.fit(X_det, y_det)
    return detector_clf


# ---------------------------------------------------------------------------
# Wrap com ART BinaryInputDetector
# ---------------------------------------------------------------------------

def wrap_with_art_detector(detector_clf, x_min: np.ndarray,
                            x_max: np.ndarray):
    """
    Envolve o detector sklearn com art.defences.detector.evasion.BinaryInputDetector.
    Retorna o wrapper ART ou None se não disponível.
    """
    try:
        from art.estimators.classification import SklearnClassifier
        from art.defences.detector.evasion import BinaryInputDetector
    except ImportError as exc:
        log.warning(f"BinaryInputDetector ART não disponível: {exc}. "
                    "Usando detector sklearn direto.")
        return None

    art_detector_clf = SklearnClassifier(
        model=detector_clf,
        clip_values=(x_min, x_max),
    )
    return BinaryInputDetector(art_detector_clf)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Treina e avalia um BinaryInputDetector contra ataques adversariais"
    )
    parser.add_argument("--data",        required=True)
    parser.add_argument("--target",      required=True)
    parser.add_argument("--model",       required=True)
    parser.add_argument("--meta",        default=None)
    parser.add_argument("--detector",    default=None,
                        help="Caminho para detector salvo (.pkl). "
                             "Se ausente, treina um novo.")
    parser.add_argument("--n-samples",   type=int, default=50,
                        help="Amostras para gerar adversariais (default: 50)")
    parser.add_argument("--min-detection-rate", type=float, default=0.70,
                        help="Mínimo de adversariais que o detector deve capturar (default: 0.70)")
    parser.add_argument("--max-fpr",     type=float, default=0.10,
                        help="Máximo de falsos positivos aceito (default: 0.10)")
    parser.add_argument("--output-dir",  default="results")
    parser.add_argument("--verify",      action="store_true",
                        help="Modo verificação: carrega detector existente e avalia")
    args = parser.parse_args()

    import joblib

    # ── Carregar modelo principal ────────────────────────────────────────────
    if not pathlib.Path(args.model).exists():
        log.error(f"Modelo não encontrado: {args.model}")
        sys.exit(1)
    clf = joblib.load(args.model)
    meta = _load_meta(args.meta)

    # ── Carregar dados e gerar adversariais ──────────────────────────────────
    X, y = _load_features(args.data, args.target, meta)
    n = min(args.n_samples, len(X))
    X_sub, y_sub = X[:n], y[:n]
    x_min = X_sub.min(axis=0)
    x_max = X_sub.max(axis=0)

    log.info(f"  Amostras limpas : {n}")
    X_adv = _generate_adversarial(clf, X_sub)
    log.info(f"  Amostras adversariais geradas: {len(X_adv)}")

    # ── Carregar ou treinar detector ─────────────────────────────────────────
    detector_path = pathlib.Path(args.detector) if args.detector else \
                    pathlib.Path("model/binary_detector.pkl")

    if args.verify or detector_path.exists():
        if not detector_path.exists():
            log.error(f"Detector não encontrado para verificação: {detector_path}")
            sys.exit(1)
        log.info(f"  Carregando detector existente: {detector_path}")
        detector_clf = joblib.load(detector_path)
    else:
        log.info("  Treinando novo BinaryInputDetector…")
        detector_clf = train_detector(X_sub, X_adv, clf)
        detector_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(detector_clf, detector_path)
        log.info(f"  Detector salvo: {detector_path}")

    # ── Tentar usar wrapper ART ──────────────────────────────────────────────
    art_detector = wrap_with_art_detector(detector_clf, x_min, x_max)

    # ── Avaliar desempenho do detector ───────────────────────────────────────
    log.info("\n=== Avaliação do BinaryInputDetector ===")

    if art_detector is not None:
        # ART retorna (predictions, is_adversarial_bool_array)
        _, is_adv_on_adv   = art_detector.detect(X_adv)
        _, is_adv_on_clean = art_detector.detect(X_sub)
    else:
        # Fallback: sklearn direto
        is_adv_on_adv   = detector_clf.predict(X_adv).astype(bool)
        is_adv_on_clean = detector_clf.predict(X_sub).astype(bool)

    detection_rate = float(is_adv_on_adv.mean())    # True Positive Rate
    false_pos_rate = float(is_adv_on_clean.mean())  # False Positive Rate

    # Acurácia do RF com filtro: inputs marcados como adversariais são rejeitados
    X_filtered = X_adv[~is_adv_on_adv]
    y_filtered = y_sub[~is_adv_on_adv]
    acc_filtered = float((clf.predict(X_filtered) == y_filtered).mean()) \
                   if len(X_filtered) > 0 else 1.0

    acc_no_filter = float((clf.predict(X_adv) == y_sub).mean())

    log.info(f"  Detection rate (TPR) : {detection_rate:.4f}  "
             f"(gate: ≥ {args.min_detection_rate:.2f})")
    log.info(f"  False positive rate  : {false_pos_rate:.4f}  "
             f"(gate: ≤ {args.max_fpr:.2f})")
    log.info(f"  RF acc sem filtro    : {acc_no_filter:.4f}")
    log.info(f"  RF acc com filtro    : {acc_filtered:.4f}  "
             f"({len(X_filtered)}/{n} adversariais passaram)")

    # ── Relatório ────────────────────────────────────────────────────────────
    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "detector": str(detector_path),
        "n_samples": n,
        "detection_rate":  detection_rate,
        "false_positive_rate": false_pos_rate,
        "acc_rf_no_filter": acc_no_filter,
        "acc_rf_with_filter": acc_filtered,
        "adversarials_blocked": int(is_adv_on_adv.sum()),
        "adversarials_passed": int((~is_adv_on_adv).sum()),
        "gate_detection_rate": args.min_detection_rate,
        "gate_max_fpr": args.max_fpr,
        "passed": bool(detection_rate >= args.min_detection_rate
                       and false_pos_rate <= args.max_fpr),
    }
    out_path = out_dir / "binary_detector_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"  Relatório salvo: {out_path}")

    # ── Gate ─────────────────────────────────────────────────────────────────
    if detection_rate < args.min_detection_rate:
        log.error(
            f"Detector Gate FAILED — detection rate {detection_rate:.4f} "
            f"< {args.min_detection_rate:.4f}. "
            "O detector não captura adversariais suficientes."
        )
        sys.exit(1)

    if false_pos_rate > args.max_fpr:
        log.error(
            f"Detector Gate FAILED — false positive rate {false_pos_rate:.4f} "
            f"> {args.max_fpr:.4f}. "
            "O detector está bloqueando inputs legítimos em excesso."
        )
        sys.exit(1)

    log.info("BinaryInputDetector Gate: PASSED")


if __name__ == "__main__":
    main()
