"""
train_loo.py — XGBoost classifier with Leave-One-Pair-Out validation.

For each of the 275 NLR×HET-C pairs:
  - remove that single pair from training
  - fit a model on the remaining 274 pairs
  - predict the held-out pair

This answers a DIFFERENT question than LONO:
  LONO  → "How well does the model generalise to a completely new NLR?"
  LOO   → "How well does the model fill a missing cell of the interaction matrix?"

LOO is a weaker test because the model also sees other pairs of the held-out
NLR sequence during training — i.e. it has effectively learnt that NLR's
embedding from neighbouring pairs. However, for phenotypes with only one
NLR (D1, E2) LOO is the ONLY validation that produces a number, because
LONO removes the only example of that phenotype from training entirely.

In the thesis both are reported: LONO as the main rigorous test, LOO as
a diagnostic for matrix completion behaviour.

Inputs:  features_mean.npz, features_max.npz (built by build_pair_features.py)
Outputs:
  data/models/oof_predictions_loo.csv
  data/models/fold_summary_loo.csv
  data/models/xgb_feature_importance_loo.csv

Usage:
  cd thesis/
  python scripts/train_loo.py
  python scripts/train_loo.py --features-mean data/models/features_hv_mean.npz \
                               --features-max  data/models/features_hv_max.npz \
                               --out-dir       data/models_hv

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
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(p.resolve())


def load_features(path: Path) -> dict:
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


def run_loo(pack: dict, args, pool_name: str):
    """Leave-One-Pair-Out: 275 fits, each holding out exactly one pair."""
    import xgboost as xgb

    X = pack["X"].astype(np.float32)
    y = pack["y"].astype(int)
    nlr_id = pack["nlr_id"].astype(str)
    hetc_id = pack["hetc_id"].astype(str)
    phenotype = pack["phenotype"].astype(str)
    confidence = pack["confidence"].astype(str)
    n = len(y)
    print(f"\n=== LOO: {pool_name} pool, {n} pairs ===")

    proba_oof = np.full(n, np.nan, dtype=np.float32)
    pred_oof = np.full(n, -1, dtype=np.int8)
    importances_sum = np.zeros(X.shape[1], dtype=np.float64)
    fold_rows = []

    for i in range(n):
        test_idx = np.array([i])
        train_idx = np.delete(np.arange(n), i)
        y_tr = y[train_idx]

        pos = int(y_tr.sum())
        neg = len(y_tr) - pos
        spw = neg / max(pos, 1)
        params = get_xgb_params(args, scale_pos_weight=spw)

        clf = xgb.XGBClassifier(**params)
        clf.fit(X[train_idx], y_tr)

        proba = float(clf.predict_proba(X[test_idx])[0, 1])
        pred = int(proba >= args.threshold)

        proba_oof[i] = proba
        pred_oof[i] = pred

        try:
            importances_sum += clf.feature_importances_
        except Exception:
            pass

        fold_rows.append({
            "pair_idx": i,
            "nlr_id": nlr_id[i],
            "hetc_id": hetc_id[i],
            "phenotype": phenotype[i],
            "confidence": confidence[i],
            "label": int(y[i]),
            "proba": proba,
            "pred": pred,
            "correct": int(pred == y[i]),
            "n_train": len(train_idx),
            "train_pos_frac": pos / len(y_tr),
        })

        if (i + 1) % 25 == 0 or i == n - 1:
            correct_so_far = sum(r["correct"] for r in fold_rows)
            print(f"  [{i+1:>3d}/{n}]  acc_so_far={correct_so_far/(i+1):.3f}  "
                  f"last: {nlr_id[i]} × {hetc_id[i]} (label={y[i]}, "
                  f"proba={proba:.3f}, pred={pred})")

    fold_df = pd.DataFrame(fold_rows)
    importances_mean = importances_sum / n
    return proba_oof, pred_oof, fold_df, importances_mean


def overall_metrics(y_true, y_pred, y_proba):
    """MCC, F1, AUC. Falls back to a NumPy-only implementation if sklearn
    is unavailable."""
    try:
        from sklearn.metrics import matthews_corrcoef, f1_score, roc_auc_score
        return {
            "mcc": matthews_corrcoef(y_true, y_pred),
            "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
            "f1_macro": f1_score(y_true, y_pred, average="macro"),
            "auc": roc_auc_score(y_true, y_proba),
        }
    except ImportError:
        # Manual implementation
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        den = ((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)) ** 0.5
        mcc = (tp*tn - fp*fn) / den if den else 0.0
        prec = tp / (tp+fp) if tp+fp else 0.0
        rec  = tp / (tp+fn) if tp+fn else 0.0
        f1 = 2*prec*rec/(prec+rec) if prec+rec else 0.0
        return {"mcc": mcc, "f1_macro": f1, "f1_weighted": f1, "auc": np.nan}


def per_phenotype_metrics(df: pd.DataFrame, pred_col: str):
    rows = []
    for ph, g in df.groupby("phenotype"):
        m = overall_metrics(g["label"].to_numpy(), g[pred_col].to_numpy(),
                            g[pred_col.replace("pred_label", "pred_prob")].to_numpy())
        m.update({"phenotype": ph, "n": len(g),
                  "n_pos": int(g["label"].sum()),
                  "n_correct": int((g[pred_col] == g["label"]).sum())})
        rows.append(m)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-mean", default=str(MODELS_DIR / "features_mean.npz"))
    ap.add_argument("--features-max",  default=str(MODELS_DIR / "features_max.npz"))
    ap.add_argument("--out-dir", default=str(MODELS_DIR))
    ap.add_argument("--threshold", type=float, default=0.5)
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

    pack_mean = load_features(Path(args.features_mean))
    pack_max  = load_features(Path(args.features_max))

    proba_m, pred_m, fold_m, imp_m = run_loo(pack_mean, args, "mean")
    proba_x, pred_x, fold_x, imp_x = run_loo(pack_max,  args, "max")

    # OOF predictions
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
    })
    oof_path = out_dir / "oof_predictions_loo.csv"
    oof.to_csv(oof_path, index=False)
    print(f"\nOOF predictions → {display_path(oof_path)}")

    # Pair-by-pair summary
    fold_m["pool"] = "mean"
    fold_x["pool"] = "max"
    fold_summary = pd.concat([fold_m, fold_x], ignore_index=True)
    fold_path = out_dir / "fold_summary_loo.csv"
    fold_summary.to_csv(fold_path, index=False)
    print(f"Fold summary → {display_path(fold_path)}")

    # Feature importances
    imp_df = pd.DataFrame({
        "feature_idx":     np.arange(len(imp_m)),
        "importance_mean": imp_m,
        "importance_max":  imp_x,
    })
    imp_path = out_dir / "xgb_feature_importance_loo.csv"
    imp_df.to_csv(imp_path, index=False)
    print(f"Feature importances → {display_path(imp_path)}")

    # Overall metrics
    print("\n" + "="*60)
    print("OVERALL (275 pairs)")
    print("="*60)
    for pool_name, pred_col in [("mean", "pred_label_mean"), ("max", "pred_label_max")]:
        proba_col = pred_col.replace("pred_label", "pred_prob")
        m = overall_metrics(oof["label"].to_numpy(),
                            oof[pred_col].to_numpy(),
                            oof[proba_col].to_numpy())
        print(f"  {pool_name:>4s} pool:  MCC={m['mcc']:.3f}  "
              f"F1(weighted)={m['f1_weighted']:.3f}  "
              f"F1(macro)={m['f1_macro']:.3f}  AUC={m['auc']:.3f}")

    # Per-phenotype
    print("\n" + "="*60)
    print("PER PHENOTYPE")
    print("="*60)
    for pool_name, pred_col in [("mean", "pred_label_mean"), ("max", "pred_label_max")]:
        proba_col = pred_col.replace("pred_label", "pred_prob")
        print(f"\n  {pool_name} pool:")
        print(f"    {'phenotype':<10} {'n':>4} {'n_pos':>6} {'correct':>8} {'MCC':>8} {'AUC':>8}")
        for ph, g in oof.groupby("phenotype"):
            m = overall_metrics(g["label"].to_numpy(),
                                g[pred_col].to_numpy(),
                                g[proba_col].to_numpy())
            n_correct = int((g[pred_col] == g["label"]).sum())
            mcc_str = f"{m['mcc']:.3f}" if not np.isnan(m['mcc']) else "n/a"
            auc_str = f"{m['auc']:.3f}" if not np.isnan(m['auc']) else "n/a"
            print(f"    {ph:<10} {len(g):>4d} {int(g['label'].sum()):>6d} "
                  f"{n_correct:>8d} {mcc_str:>8s} {auc_str:>8s}")

    print("\nDone.")


if __name__ == "__main__":
    main()
