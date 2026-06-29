"""
scripts/adversarial_eval.py — Ataques adversariais via IBM ART (genérico)
Suporta dois modos de input via --input-type:

  tabular (default) — sklearn models (RF, XGBoost, SVM…)
    --attack fgsm → SquareAttack     (score-based black-box)
    --attack pgd  → DecisionTreeAttack (tree-structure aware)

  images — PyTorch models (.pt / .pth)
    --attack fgsm → FastGradientMethod  (gradiente real)
    --attack pgd  → ProjectedGradientDescent (iterativo, mais forte)

MITRE ATLAS: AML.T0043 Craft Adversarial Examples · AML.T0015 Evade ML Model

Uso tabular:
    python scripts/adversarial_eval.py \\
        --data dataset.csv --target label \\
        --model model/rf_model.pkl \\
        --attack fgsm --epsilon 0.1 --min-accuracy 0.75

Uso imagens:
    python scripts/adversarial_eval.py \\
        --input-type images \\
        --data data/ci_images.pt \\
        --model model/cnn.pt \\
        --input-shape 3 224 224 --num-classes 2 \\
        --attack pgd --epsilon 0.03 --min-accuracy 0.70
"""

import argparse
import json
import logging
import pathlib
import sys

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_sklearn_model(model_path: str):
    import joblib
    return joblib.load(model_path)


def load_pytorch_model(model_path: str):
    try:
        import torch
    except ImportError:
        log.error("PyTorch não instalado. Run: pip install torch torchvision")
        sys.exit(1)
    model = torch.load(model_path, map_location="cpu", weights_only=False)
    model.eval()
    return model


def load_meta(meta_path: str | None) -> dict | None:
    if not meta_path or not pathlib.Path(meta_path).exists():
        return None
    with open(meta_path) as f:
        import json as _json
        return _json.load(f)


# ---------------------------------------------------------------------------
# Feature builders
# ---------------------------------------------------------------------------

def build_tabular_features(data_path: str, target: str, meta: dict | None,
                            sample: int) -> tuple[np.ndarray, np.ndarray]:
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder

    p = pathlib.Path(data_path)
    df = pd.read_csv(p, low_memory=False, nrows=sample) \
         if p.suffix == ".csv" else pd.read_parquet(p)

    from sklearn.model_selection import train_test_split
    y_col = pd.to_numeric(df[target], errors="coerce").fillna(0).astype(int)
    _, df_test = train_test_split(df, test_size=0.2, random_state=42, stratify=y_col)
    df_test = df_test.reset_index(drop=True)

    df = df_test.copy()
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
        exclude = {target}
        for col in df.select_dtypes(include="object").columns:
            if col not in exclude:
                df[col] = df[col].fillna("UNKNOWN").astype("category").cat.codes
        for col in df.select_dtypes(exclude="object").columns:
            if col != target:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        feature_cols = [c for c in df.columns if c != target and df[c].dtype != object]

    X = df[feature_cols].values.astype(np.float32)
    y = pd.to_numeric(df[target], errors="coerce").fillna(0).astype(int).values
    return X, y


def build_image_features(data_path: str, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Carrega imagens de um .pt file (dict com 'X' e 'y') ou de um diretório
    com subpastas por classe (torchvision ImageFolder format).
    """
    try:
        import torch
        import torchvision.transforms as T
    except ImportError:
        log.error("PyTorch não instalado. Run: pip install torch torchvision")
        sys.exit(1)

    p = pathlib.Path(data_path)
    if p.suffix in {".pt", ".pth"}:
        data = torch.load(p, map_location="cpu", weights_only=False)
        X = data["X"].numpy().astype(np.float32)
        y = data["y"].numpy().astype(int)
    elif p.is_dir():
        from torchvision.datasets import ImageFolder
        transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])
        dataset = ImageFolder(str(p), transform=transform)
        loader = torch.utils.data.DataLoader(dataset, batch_size=n_samples, shuffle=False)
        X_t, y_t = next(iter(loader))
        X = X_t.numpy().astype(np.float32)
        y = y_t.numpy().astype(int)
    else:
        log.error(f"Formato de imagem não suportado: {p.suffix}. Use .pt ou diretório ImageFolder.")
        sys.exit(1)

    X = X[:n_samples]
    y = y[:n_samples]
    # Normaliza para [0, 1] se necessário
    if X.max() > 1.0:
        X = X / 255.0
    log.info(f"  Images shape: {X.shape} | Labels: {np.unique(y)}")
    return X, y


# ---------------------------------------------------------------------------
# ART estimator builders
# ---------------------------------------------------------------------------

def build_tabular_estimator(clf, X: np.ndarray):
    try:
        from art.estimators.classification import SklearnClassifier
    except ImportError as exc:
        log.error(f"adversarial-robustness-toolbox não instalado: {exc}")
        sys.exit(1)
    x_min = X.min(axis=0)
    x_max = X.max(axis=0)
    return SklearnClassifier(model=clf, clip_values=(x_min, x_max))


def build_image_estimator(model, num_classes: int, input_shape: tuple[int, ...]):
    try:
        import torch
        import torch.nn as nn
        from art.estimators.classification import PyTorchClassifier
    except ImportError as exc:
        log.error(f"PyTorch ou ART não instalados: {exc}")
        sys.exit(1)
    return PyTorchClassifier(
        model=model,
        loss=nn.CrossEntropyLoss(),
        input_shape=input_shape,
        nb_classes=num_classes,
        clip_values=(0.0, 1.0),
    )


# ---------------------------------------------------------------------------
# Attack builders
# ---------------------------------------------------------------------------

def build_tabular_attack(attack_name: str, art_clf, epsilon: float, steps: int):
    """
    Ataques para dados tabulares / sklearn (sem gradientes disponíveis).
    Hierarquia:
      fgsm: SquareAttack → DecisionTreeAttack → BoundaryAttack
      pgd:  DecisionTreeAttack → SquareAttack → BoundaryAttack
    """
    DecisionTreeAttack = SquareAttack = BoundaryAttack = None
    try:
        from art.attacks.evasion import DecisionTreeAttack
    except Exception as exc:
        log.warning(f"  DecisionTreeAttack não disponível: {exc}")
    try:
        from art.attacks.evasion import SquareAttack
    except Exception as exc:
        log.warning(f"  SquareAttack não disponível: {exc}")
    try:
        from art.attacks.evasion import BoundaryAttack
    except Exception as exc:
        log.warning(f"  BoundaryAttack não disponível: {exc}")

    if all(a is None for a in [DecisionTreeAttack, SquareAttack, BoundaryAttack]):
        log.error("Nenhum ataque tabular disponível no ART.")
        sys.exit(1)

    if attack_name == "fgsm":
        if SquareAttack:
            log.info(f"  SquareAttack (FGSM proxy, ε={epsilon}): running…")
            return SquareAttack(art_clf, eps=epsilon, max_iter=100, nb_restarts=1, verbose=False)
        if DecisionTreeAttack:
            log.info("  DecisionTreeAttack (fallback FGSM): running…")
            return DecisionTreeAttack(art_clf, verbose=False)
        log.info("  BoundaryAttack (fallback 2 FGSM): running…")
        return BoundaryAttack(art_clf, targeted=False, max_iter=100)
    else:  # pgd
        if DecisionTreeAttack:
            log.info("  DecisionTreeAttack (PGD proxy — tree-structure aware): running…")
            return DecisionTreeAttack(art_clf, verbose=False)
        if SquareAttack:
            log.info(f"  SquareAttack (fallback PGD, ε={epsilon}): running…")
            return SquareAttack(art_clf, eps=epsilon, max_iter=steps, nb_restarts=3, verbose=False)
        log.info("  BoundaryAttack (fallback 2 PGD): running…")
        return BoundaryAttack(art_clf, targeted=False, max_iter=steps)


def build_image_attack(attack_name: str, art_clf, epsilon: float, steps: int):
    """
    Ataques para imagens / PyTorch — usa gradientes reais.
      fgsm: FastGradientMethod (perturbação única no gradiente da loss)
      pgd:  ProjectedGradientDescent (iterativo, mais forte)
    """
    try:
        from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent
    except ImportError as exc:
        log.error(f"ART evasion attacks não disponíveis: {exc}")
        sys.exit(1)

    if attack_name == "fgsm":
        log.info(f"  FastGradientMethod (FGSM real, ε={epsilon}): running…")
        return FastGradientMethod(art_clf, eps=epsilon, norm=np.inf)
    else:
        log.info(f"  ProjectedGradientDescent (PGD real, ε={epsilon}, steps={steps}): running…")
        return ProjectedGradientDescent(art_clf, eps=epsilon,
                                        eps_step=epsilon / 4,
                                        max_iter=steps, targeted=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Adversarial robustness evaluation via IBM ART"
    )
    # ── Input type ───────────────────────────────────────────────────────────
    parser.add_argument("--input-type", choices=["tabular", "images"], default="tabular",
                        help="Input type: 'tabular' (sklearn/CSV) or 'images' (PyTorch/.pt dir)")
    # ── Tabular args ─────────────────────────────────────────────────────────
    parser.add_argument("--data",         default=None,
                        help="CSV/Parquet file (tabular) or .pt/dir (images)")
    parser.add_argument("--target",       default=None,
                        help="Target column name (tabular only)")
    parser.add_argument("--meta",         default=None,
                        help="feature_names.json from train_rf.py (tabular only)")
    # ── Image args ───────────────────────────────────────────────────────────
    parser.add_argument("--input-shape",  nargs="+", type=int, default=[3, 224, 224],
                        help="Model input shape C H W (images only, default: 3 224 224)")
    parser.add_argument("--num-classes",  type=int,  default=2,
                        help="Number of output classes (images only, default: 2)")
    # ── Common args ──────────────────────────────────────────────────────────
    parser.add_argument("--model",        required=True,
                        help=".pkl/.joblib (tabular) or .pt/.pth (images)")
    parser.add_argument("--attack",       required=True, choices=["fgsm", "pgd"])
    parser.add_argument("--epsilon",      type=float, default=0.1)
    parser.add_argument("--steps",        type=int,   default=40)
    parser.add_argument("--min-accuracy", type=float, default=0.75)
    parser.add_argument("--n-samples",    type=int,   default=500)
    parser.add_argument("--sample",       type=int,   default=50_000,
                        help="Max rows for tabular loading")
    parser.add_argument("--output-dir",   default="results")
    args = parser.parse_args()

    if not pathlib.Path(args.model).exists():
        log.error(f"Model not found: {args.model}")
        sys.exit(1)
    if args.data and not pathlib.Path(args.data).exists():
        log.error(f"Data not found: {args.data}")
        sys.exit(1)

    # ── Mode: tabular ─────────────────────────────────────────────────────────
    if args.input_type == "tabular":
        if not args.data or not args.target:
            log.error("--data e --target são obrigatórios para --input-type tabular")
            sys.exit(1)
        log.info(f"Mode: TABULAR | model={args.model}")
        clf  = load_sklearn_model(args.model)
        meta = load_meta(args.meta)
        X, y = build_tabular_features(args.data, args.target, meta, args.sample)
        n    = min(args.n_samples, len(X))
        X_sub, y_sub = X[:n], y[:n]
        art_clf   = build_tabular_estimator(clf, X)
        acc_clean = (clf.predict(X_sub) == y_sub).mean()
        log.info(f"  Baseline accuracy (clean): {acc_clean:.4f}")
        attack    = build_tabular_attack(args.attack, art_clf, args.epsilon, args.steps)
        X_adv     = attack.generate(X_sub)
        acc_adv   = (clf.predict(X_adv) == y_sub).mean()

    # ── Mode: images ──────────────────────────────────────────────────────────
    else:
        if not args.data:
            log.error("--data é obrigatório para --input-type images")
            sys.exit(1)
        try:
            import torch
        except ImportError:
            log.error("PyTorch não instalado. Run: pip install torch torchvision")
            sys.exit(1)
        log.info(f"Mode: IMAGES | model={args.model} | shape={args.input_shape}")
        model    = load_pytorch_model(args.model)
        X, y     = build_image_features(args.data, args.n_samples)
        n        = len(X)
        X_sub, y_sub = X, y
        input_shape  = tuple(args.input_shape)
        art_clf      = build_image_estimator(model, args.num_classes, input_shape)
        preds        = np.argmax(art_clf.predict(X_sub), axis=1)
        acc_clean    = (preds == y_sub).mean()
        log.info(f"  Baseline accuracy (clean): {acc_clean:.4f}")
        attack       = build_image_attack(args.attack, art_clf, args.epsilon, args.steps)
        X_adv        = attack.generate(X_sub)
        acc_adv      = (np.argmax(art_clf.predict(X_adv), axis=1) == y_sub).mean()

    # ── Gate & report ─────────────────────────────────────────────────────────
    log.info(f"\n=== Adversarial Gate ({args.input_type.upper()}) ===")
    log.info(f"  Attack                 : {args.attack.upper()} (ε={args.epsilon})")
    log.info(f"  Accuracy (clean)       : {acc_clean:.4f}")
    log.info(f"  Accuracy (adversarial) : {acc_adv:.4f}")
    log.info(f"  Drop                   : {acc_clean - acc_adv:.4f}")
    log.info(f"  Threshold              : {args.min_accuracy:.4f}")
    log.info("  NOTE: threshold empírico — calibre com baseline antes de tornar bloqueante.")

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"adversarial_{args.attack}.json", "w") as f:
        json.dump({
            "input_type": args.input_type,
            "attack": args.attack,
            "epsilon": args.epsilon,
            "n_samples": n,
            "acc_clean": float(acc_clean),
            "acc_adversarial": float(acc_adv),
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
