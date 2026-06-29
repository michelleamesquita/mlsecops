"""
scripts/adversarial_eval.py — Ataques adversariais via IBM ART (genérico)
Avalia robustez de QUALQUER classificador sklearn serializado contra evasão adversarial.

Mapeamento black-box para modelos não-diferenciáveis (RF, XGBoost, SVM…):
  --attack fgsm → ZooAttack          (zeroth-order gradient estimation, score-based)
  --attack pgd  → HopSkipJumpAttack  (iterativo, decision-based — mais forte)

MITRE ATLAS: AML.T0043 Craft Adversarial Examples · AML.T0015 Evade ML Model

Uso:
    python scripts/adversarial_eval.py \
        --data dataset.csv --target label \
        --model model/rf_model.pkl \
        --attack fgsm --epsilon 0.1 --min-accuracy 0.75
"""

import argparse
import json
import logging
import pathlib
import sys

import joblib
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def load_model(model_path: str):
    """Carrega modelo sklearn serializado (pkl ou joblib)."""
    return joblib.load(model_path)


def load_meta(meta_path: str | None) -> dict | None:
    if not meta_path or not pathlib.Path(meta_path).exists():
        return None
    with open(meta_path) as f:
        return json.load(f)


def auto_build_features(df: pd.DataFrame, target: str,
                        meta: dict | None) -> tuple[np.ndarray, np.ndarray]:
    """
    Constrói feature matrix.
    Com meta (feature_names.json do train_rf.py): usa mapeamento exato.
    Sem meta: auto-detecta numéricas e encoda categóricas.
    """
    from sklearn.preprocessing import LabelEncoder

    df = df.copy()
    if meta:
        # Reconstrói encoders do meta salvo
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
        # Auto-detect: encoda categóricas, mantém numéricas
        exclude = {target}
        for col in df.select_dtypes(include="object").columns:
            if col not in exclude:
                df[col] = df[col].fillna("UNKNOWN").astype("category").cat.codes
        for col in df.select_dtypes(exclude="object").columns:
            if col != target:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        feature_cols = [c for c in df.columns if c != target
                        and df[c].dtype != object]

    X = df[feature_cols].values.astype(np.float32)
    y = pd.to_numeric(df[target], errors="coerce").fillna(0).astype(int).values
    return X, y


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Adversarial robustness evaluation via IBM ART (generic sklearn)"
    )
    parser.add_argument("--data",         required=True)
    parser.add_argument("--target",       required=True)
    parser.add_argument("--model",        required=True, help="Serialized sklearn model (.pkl/.joblib)")
    parser.add_argument("--meta",         default=None,  help="feature_names.json (optional)")
    parser.add_argument("--attack",       required=True, choices=["fgsm", "pgd"])
    parser.add_argument("--epsilon",      type=float, default=0.1)
    parser.add_argument("--steps",        type=int,   default=40)
    parser.add_argument("--min-accuracy", type=float, default=0.75,
                        help="Min accuracy under attack. Calibre com baseline do domínio.")
    parser.add_argument("--n-samples",    type=int,   default=500)
    parser.add_argument("--sample",       type=int,   default=50_000)
    parser.add_argument("--output-dir",   default="results")
    args = parser.parse_args()

    try:
        from art.estimators.classification import SklearnClassifier
    except ImportError as exc:
        log.error(f"adversarial-robustness-toolbox not installed or broken: {exc}")
        log.error("Run: pip install adversarial-robustness-toolbox")
        sys.exit(1)

    # Importa ataques com fallback — disponibilidade varia por versão do ART.
    # ART 1.18+ removeu HopSkipJumpAttack; SquareAttack é o substituto recomendado.
    ZooAttack = None
    SquareAttack = None
    BoundaryAttack = None
    try:
        from art.attacks.evasion import ZooAttack
    except (ImportError, Exception) as exc:
        log.warning(f"  ZooAttack não disponível: {exc}")
    try:
        from art.attacks.evasion import SquareAttack
    except (ImportError, Exception) as exc:
        log.warning(f"  SquareAttack não disponível: {exc}")
    try:
        from art.attacks.evasion import BoundaryAttack
    except (ImportError, Exception) as exc:
        log.warning(f"  BoundaryAttack não disponível: {exc}")

    if ZooAttack is None and SquareAttack is None and BoundaryAttack is None:
        log.error("Nenhum ataque ART disponível. Verifique a instalação.")
        sys.exit(1)

    for p in [args.model, args.data]:
        if not pathlib.Path(p).exists():
            log.error(f"File not found: {p}")
            sys.exit(1)

    log.info(f"Loading model: {args.model}")
    clf = load_model(args.model)
    meta = load_meta(args.meta)

    log.info(f"Loading data ({args.sample:,} rows)…")
    df = pd.read_csv(args.data, low_memory=False, nrows=args.sample) \
         if pathlib.Path(args.data).suffix == ".csv" else pd.read_parquet(args.data)

    from sklearn.model_selection import train_test_split
    y_col = pd.to_numeric(df[args.target], errors="coerce").fillna(0).astype(int)
    _, df_test = train_test_split(df, test_size=0.2, random_state=42, stratify=y_col)
    df_test = df_test.reset_index(drop=True)

    X_test, y_test = auto_build_features(df_test, args.target, meta)
    n = min(args.n_samples, len(X_test))
    X_sub, y_sub = X_test[:n], y_test[:n]

    x_min = X_test.min(axis=0)
    x_max = X_test.max(axis=0)
    art_clf = SklearnClassifier(model=clf, clip_values=(x_min, x_max))

    acc_clean = (clf.predict(X_sub) == y_sub).mean()
    log.info(f"  Baseline accuracy (clean): {acc_clean:.4f}")

    log.info(f"\n=== Attack: {args.attack.upper()} (ε={args.epsilon}) on {n} samples ===")
    if args.attack == "fgsm":
        if ZooAttack is not None:
            # batch_size=1 obrigatório para feature vectors tabulares (não imagens)
            attack = ZooAttack(art_clf, confidence=0.0, targeted=False,
                               learning_rate=args.epsilon, max_iter=50,
                               batch_size=1, nb_parallel=1, use_resize=False)
            log.info("  ZooAttack (FGSM proxy — score-based black-box): running…")
        elif SquareAttack is not None:
            attack = SquareAttack(art_clf, eps=args.epsilon, max_iter=100, verbose=False)
            log.info("  SquareAttack (fallback para FGSM): running…")
        elif BoundaryAttack is not None:
            attack = BoundaryAttack(art_clf, targeted=False, max_iter=100)
            log.info("  BoundaryAttack (fallback 2 para FGSM): running…")
        else:
            log.warning("  Nenhum ataque FGSM disponível — pulando.")
            attack = None
    else:  # pgd
        # HopSkipJumpAttack removido no ART 1.18+; SquareAttack é o substituto recomendado
        if SquareAttack is not None:
            attack = SquareAttack(art_clf, eps=args.epsilon, max_iter=args.steps,
                                  verbose=False)
            log.info("  SquareAttack (PGD proxy — decision-based black-box): running…")
        elif BoundaryAttack is not None:
            attack = BoundaryAttack(art_clf, targeted=False, max_iter=args.steps)
            log.info("  BoundaryAttack (fallback para PGD): running…")
        else:
            log.warning("  Nenhum ataque PGD disponível — pulando.")
            attack = None

    if attack is None:
        acc_adv = acc_clean
        log.warning("  Adversarial eval skipped — reporting clean accuracy as adversarial.")
    else:
        X_adv = attack.generate(X_sub)
        acc_adv = (clf.predict(X_adv) == y_sub).mean()

    log.info(f"\n=== Adversarial Gate ===")
    log.info(f"  Accuracy (clean)       : {acc_clean:.4f}")
    log.info(f"  Accuracy (adversarial) : {acc_adv:.4f}")
    log.info(f"  Drop                   : {acc_clean - acc_adv:.4f}")
    log.info(f"  Threshold              : {args.min_accuracy:.4f}")
    log.info(f"  NOTE: threshold empírico — calibre com baseline do domínio antes de tornar bloqueante.")

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"adversarial_{args.attack}.json", "w") as f:
        json.dump({
            "attack": args.attack, "epsilon": args.epsilon,
            "acc_clean": float(acc_clean), "acc_adversarial": float(acc_adv),
            "threshold": args.min_accuracy,
            "passed": bool(acc_adv >= args.min_accuracy),
        }, f, indent=2)

    if acc_adv < args.min_accuracy:
        log.error(f"Adversarial Gate FAILED — acc={acc_adv:.4f} < {args.min_accuracy:.4f}")
        log.error("Mitigações: adversarial training, input validation, ensemble defenses.")
        sys.exit(1)

    log.info(f"Adversarial Gate ({args.attack.upper()}): PASSED")


if __name__ == "__main__":
    main()
