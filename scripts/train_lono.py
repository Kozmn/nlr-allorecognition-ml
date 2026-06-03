"""
train_lono.py — XGBoost classifier with leave-one-NLR-out cross-validation.

Trains two binary classifiers (compatible=1 / incompatible=0):
  - mean-pool features  (data/models/features_mean.npz)
  - max-pool features   (data/models/features_max.npz)

Validation:
  Leave-One-NLR-Out (LONO): 25 folds. Each fold holds out one entire NLR
  sequence (with all 11 of its HET-C pairs) from training and predicts on it.
  After all 25 folds every one of the 275 pairs has been predicted exactly
  once out-of-fold.

Classifier: XGBoost with regularisation tuned for a small training set:
  max_depth=3           shallow trees (low complexity)
  min_child_weight=3    minimum sum of leaf weights (regularisation)
  subsample=0.8         row sub-sampling per tree (bagging)
  colsample_bytree=0.3  aggressive column sub-sampling — needed at d=4608
  reg_lambda=1.0        L2 regularisation
  reg_alpha=0.1         L1 regularisation (sparse)
  n_estimators=300
  learning_rate=0.05
  scale_pos_weight      set per fold (neg/pos) to handle class imbalance

Outputs:
  data/models/oof_predictions.csv     275 rows × columns:
        nlr_id, hetc_id, phenotype, confidence, label,
        pred_prob_mean, pred_prob_max, pred_label_mean, pred_label_max,
        fold_test_nlr
  data/models/fold_summary.csv        per fold: n_train, n_test, pos/neg,
                                       MCC and F1 on that fold
  data/models/xgb_feature_importance.csv  feature importances aggregated
                                          across the 25 folds

Usage:
  cd thesis/
  python scripts/train_lono.py
  python scripts/train_lono.py --threshold 0.4       # change decision threshold
  python scripts/train_lono.py --max-depth 4         # experiment with hyperparams

Author: Kacper Koźmin
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "data" / "models"


def display_path(p: Path) -> str:
    """Return the path relative to ROOT if possible, otherwise absolute.

    Works for relative CLI arguments (e.g. --out-dir data/models_hv) too.
    """
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(p.resolve())


def load_features(path: Path) -> dict:
    # allow_pickle=True is required because metadata columns (nlr_id, phenotype, ...)
    # are saved as object arrays by pandas .to_numpy(). These are project-internal
    # files, so loading with pickle is safe.
    npz = np.load(path, allow_pickle=True)
    pack = {k: npz[k] for k in npz.files}
    npz.close()
    return pack


def get_xgb_params(args, scale_pos_weight: float) -> dict:
    return dict(
        objective="binary:logistic",
        eval_metric="logloss",
        max_depth=args.max_depth,
        min_child_weight=args.min_child_weight,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        learning_rate=args.learning_rate,
        n_estimators=args.n_estimators,
        reg_lambda=args.reg_lambda,
        reg_alpha=args.reg_alpha,
        scale_pos_weight=scale_pos_weight,
        random_state=args.random_state,
        verbosity=0,
        n_jobs=-1,
    )


def per_fold_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """MCC and F1 for one fold, with a guard against degenerate cases."""
    from sklearn.metrics import matthews_corrcoef, f1_score
    # If the test fold contains a single class only, MCC and F1 are undefined.
    if len(np.unique(y_true)) < 2:
        return {"mcc": np.nan, "f1": np.nan,
                "n_pos": int(y_true.sum()), "n_neg": int((y_true == 0).sum())}
    return {
        "mcc": matthews_corrcoef(y_true, y_pred),
        "f1":  f1_score(y_true, y_pred, zero_division=0),
        "n_pos": int(y_true.sum()), "n_neg": int((y_true == 0).sum()),
    }


def run_lono(pack: dict, args, pool_name: str):
    """LONO loop — hold one NLR sequence out of training per fold.
    Returns out-of-fold predictions and per-fold statistics."""
    import xgboost as xgb

    X = pack["X"].astype(np.float32)
    y = pack["y"].astype(int)
    nlr_id = pack["nlr_id"].astype(str)
    hetc_id = pack["hetc_id"].astype(str)
    phenotype = pack["phenotype"].astype(str)
    confidence = pack["confidence"].astype(str)
    unique_nlrs = sorted(set(nlr_id))
    n_folds = len(unique_nlrs)
    print(f"\n=== LONO: {pool_name} pool, {n_folds} folds ===")

    # Storage for out-of-fold predictions
    proba_oof = np.full(len(y), np.nan, dtype=np.float32)
    pred_oof = np.full(len(y), -1, dtype=np.int8)
    fold_oof = np.full(len(y), "", dtype=object)
    importances_sum = np.zeros(X.shape[1], dtype=np.float64)
    fold_rows = []

    for fold_idx, held_out in enumerate(unique_nlrs, 1):
        test_mask = (nlr_id == held_out)
        train_mask = ~test_mask
        y_tr = y[train_mask]

        pos = int(y_tr.sum())
        neg = len(y_tr) - pos
        # scale_pos_weight = neg / pos; guard against pos == 0
        spw = neg / max(pos, 1)
        params = get_xgb_params(args, scale_pos_weight=spw)

        clf = xgb.XGBClassifier(**params)
        clf.fit(X[train_mask], y_tr)

        probas = clf.predict_proba(X[test_mask])[:, 1]
        preds = (probas >= args.threshold).astype(int)

        proba_oof[test_mask] = probas
        pred_oof[test_mask] = preds
        fold_oof[test_mask] = held_out

        try:
            imp = clf.feature_importances_
            importances_sum += imp
        except Exception:
            pass

        fm = per_fold_metrics(y[test_mask], preds)
        fm.update({
            "fold_test_nlr": held_out,
            "phenotype": phenotype[test_mask][0],
            "confidence": confidence[test_mask][0],
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
            "train_pos_frac": pos / len(y_tr) if len(y_tr) else np.nan,
            "test_pos_frac": y[test_mask].mean(),
        })
        fold_rows.append(fm)
        status = "MCC=n/a" if np.isnan(fm["mcc"]) else f"MCC={fm['mcc']:.2f}"
        print(f"  [{fold_idx:>2d}/{n_folds}] hold-out {held_out:<30s} "
              f"({phenotype[test_mask][0]})  pos_test={fm['n_pos']}/{fm['n_test']}  {status}")

    fold_df = pd.DataFrame(fold_rows)
    importances_mean = importances_sum / n_folds
    return proba_oof, pred_oof, fold_oof, fold_df, importances_mean


def main():
    ap = argparse.ArgumentParser()
    # Default features: [a, b] concat (2304-D). Ablation showed that explicit
    # interaction terms (|a-b|, a*b) do not improve XGBoost — the trees learn
    # interactions implicitly. Pass --features-mean features_mean.npz to use
    # the older 4608-D pack instead (full = [a, b, |a-b|, a*b]).
    ap.add_argument("--features-mean",
                    default=str(MODELS_DIR / "features_concat_mean.npz"))
    ap.add_argument("--features-max",
                    default=str(MODELS_DIR / "features_concat_max.npz"))
    ap.add_argument("--out-dir", default=str(MODELS_DIR))
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Decision threshold (default 0.5).")
    ap.add_argument("--n-estimators", type=int, default=300)
    ap.add_argument("--max-depth", type=int, default=3)
    ap.add_argument("--min-child-weight", type=int, default=3)
    ap.add_argument("--subsample", type=float, default=0.8)
    ap.add_argument("--colsample-bytree", type=float, default=0.3)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--reg-lambda", type=float, default=1.0)
    ap.add_argument("--reg-alpha", type=float, default=0.1)
    ap.add_argument("--random-state", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load both feature packs ──────────────────────────────────────────────
    pack_mean = load_features(Path(args.features_mean))
    pack_max  = load_features(Path(args.features_max))

    # ── Run LONO for both poolings ───────────────────────────────────────────
    proba_m, pred_m, fold_m, fold_df_m, imp_m = run_lono(pack_mean, args, "mean")
    proba_x, pred_x, fold_x, fold_df_x, imp_x = run_lono(pack_max,  args, "max")

    # ── Combine predictions into a single table ──────────────────────────────
    oof = pd.DataFrame({
        "nlr_id":     pack_mean["nlr_id"].astype(str),
        "hetc_id":    pack_mean["hetc_id"].astype(str),
        "phenotype":  pack_mean["phenotype"].astype(str),
        "confidence": pack_mean["confidence"].astype(str),
        "label":      pack_mean["y"].astype(int),
        "pred_prob_mean":  proba_m,
        "pred_prob_max":   proba_x,
        "pred_label_mean": pred_m,
        "pred_label_max":  pred_x,
        "fold_test_nlr":   fold_m,  # identical for both poolings
    })
    oof_path = out_dir / "oof_predictions.csv"
    oof.to_csv(oof_path, index=False)
    print(f"\nOOF predictions → {display_path(oof_path)}")

    # ── Per-fold summary ─────────────────────────────────────────────────────
    fold_df_m["pool"] = "mean"
    fold_df_x["pool"] = "max"
    fold_summary = pd.concat([fold_df_m, fold_df_x], ignore_index=True)
    fold_path = out_dir / "fold_summary.csv"
    fold_summary.to_csv(fold_path, index=False)
    print(f"Fold summary → {display_path(fold_path)}")

    # ── Feature importances ─────────────────────────────────────────────────
    imp_df = pd.DataFrame({
        "feature_idx":     np.arange(len(imp_m)),
        "importance_mean": imp_m,
        "importance_max":  imp_x,
    })
    imp_path = out_dir / "xgb_feature_importance.csv"
    imp_df.to_csv(imp_path, index=False)
    print(f"Feature importances → {display_path(imp_path)}")

    # ── Overall MCC / F1 (aggregated over 275 pairs) ────────────────────────
    from sklearn.metrics import matthews_corrcoef, f1_score
    for pool_name, pred_col in [("mean", "pred_label_mean"), ("max", "pred_label_max")]:
        mcc = matthews_corrcoef(oof["label"], oof[pred_col])
        f1w = f1_score(oof["label"], oof[pred_col], average="weighted")
        f1m = f1_score(oof["label"], oof[pred_col], average="macro")
        print(f"\n{pool_name:>4s} pool overall: MCC={mcc:.3f}  "
              f"F1(weighted)={f1w:.3f}  F1(macro)={f1m:.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
