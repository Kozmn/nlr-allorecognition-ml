"""
train_lono_svm.py — Support Vector Machine (RBF kernel) baseline with LONO.

Requires scikit-learn. Output format is fully compatible with
ensemble_eval.py and the same as train_lono.py / train_lono_logistic.py.

Outputs:
  data/models/oof_predictions_svm.csv
  data/models/fold_summary_svm.csv
  data/models/eval/svm_per_phenotype.csv

Usage:
  cd thesis/
  python scripts/train_lono_svm.py
  python scripts/train_lono_svm.py --C 10 --gamma scale

Author: Kacper Koźmin
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


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


def standardize(X_train, X_test):
    mu = X_train.mean(axis=0, keepdims=True)
    sd = X_train.std(axis=0, keepdims=True) + 1e-8
    return (X_train - mu) / sd, (X_test - mu) / sd


def run_lono(pack: dict, args, pool_name: str):
    """LONO loop with RBF-kernel SVM, balanced class weights, Platt-calibrated probabilities."""
    from sklearn.svm import SVC

    X = pack["X"].astype(np.float32)
    y = pack["y"].astype(int)
    nlr_id = pack["nlr_id"].astype(str)
    phenotype = pack["phenotype"].astype(str)
    confidence = pack["confidence"].astype(str)
    unique_nlrs = sorted(set(nlr_id))
    n_folds = len(unique_nlrs)
    print(f"\n=== SVM-RBF LONO: {pool_name} pool, {n_folds} folds ===")

    proba_oof = np.full(len(y), np.nan, dtype=np.float32)
    pred_oof = np.full(len(y), -1, dtype=np.int8)
    fold_oof = np.full(len(y), "", dtype=object)
    fold_rows = []

    for fold_idx, held_out in enumerate(unique_nlrs, 1):
        test_mask = (nlr_id == held_out)
        train_mask = ~test_mask
        X_tr, X_te = X[train_mask], X[test_mask]
        y_tr = y[train_mask]
        # Standardise
        X_tr_s, X_te_s = standardize(X_tr, X_te)
        # RBF SVM with probability=True (Platt scaling) and class_weight='balanced'
        clf = SVC(C=args.C, gamma=args.gamma, kernel="rbf",
                  class_weight="balanced", probability=True,
                  random_state=args.seed, cache_size=500)
        clf.fit(X_tr_s, y_tr)
        probas = clf.predict_proba(X_te_s)[:, 1]
        preds = (probas >= args.threshold).astype(int)
        proba_oof[test_mask] = probas
        pred_oof[test_mask] = preds
        fold_oof[test_mask] = held_out
        # Per-fold metrics
        from sklearn.metrics import matthews_corrcoef, f1_score
        if len(np.unique(y[test_mask])) >= 2:
            m = matthews_corrcoef(y[test_mask], preds)
            f = f1_score(y[test_mask], preds, zero_division=0)
        else:
            m = float("nan")
            f = float("nan")
        fold_rows.append({
            "fold_test_nlr": held_out,
            "phenotype": phenotype[test_mask][0],
            "confidence": confidence[test_mask][0],
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
            "mcc": m, "f1": f,
            "n_pos": int(y[test_mask].sum()),
            "n_neg": int((y[test_mask] == 0).sum()),
        })
        status = "MCC=n/a" if np.isnan(m) else f"MCC={m:.2f}"
        print(f"  [{fold_idx:>2d}/{n_folds}] hold-out {held_out:<30s} "
              f"({phenotype[test_mask][0]}) {status}")
    return proba_oof, pred_oof, fold_oof, pd.DataFrame(fold_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-mean", default=str(ROOT / "data" / "models" / "features_mean.npz"))
    ap.add_argument("--features-max",  default=str(ROOT / "data" / "models" / "features_max.npz"))
    ap.add_argument("--out-dir", default=str(ROOT / "data" / "models"))
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--C", type=float, default=1.0,
                    help="SVM regularisation (larger C = weaker regularisation).")
    ap.add_argument("--gamma", default="scale",
                    help="RBF gamma. 'scale' = 1/(n_features * X.var()), 'auto' = 1/n_features, or a numeric value.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pack_mean = load_features(Path(args.features_mean))
    pack_max  = load_features(Path(args.features_max))

    proba_m, pred_m, fold_m, fold_df_m = run_lono(pack_mean, args, "mean")
    proba_x, pred_x, fold_x, fold_df_x = run_lono(pack_max,  args, "max")

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
        "fold_test_nlr":   fold_m,
    })
    oof_path = out_dir / "oof_predictions_svm.csv"
    oof.to_csv(oof_path, index=False)
    print(f"\nOOF predictions → {display_path(oof_path)}")

    fold_df_m["pool"] = "mean"; fold_df_x["pool"] = "max"
    fold_summary = pd.concat([fold_df_m, fold_df_x], ignore_index=True)
    fold_path = out_dir / "fold_summary_svm.csv"
    fold_summary.to_csv(fold_path, index=False)
    print(f"Fold summary → {display_path(fold_path)}")

    # Overall metrics + ensemble
    from sklearn.metrics import matthews_corrcoef, f1_score, roc_auc_score
    print("\n=== OVERALL ===")
    for pool_name, prob_col, pred_col in [
        ("mean", "pred_prob_mean", "pred_label_mean"),
        ("max",  "pred_prob_max",  "pred_label_max"),
    ]:
        m = matthews_corrcoef(oof["label"], oof[pred_col])
        a = roc_auc_score(oof["label"], oof[prob_col])
        f = f1_score(oof["label"], oof[pred_col], zero_division=0)
        print(f"  {pool_name:>4s} pool: MCC={m:.3f}  F1={f:.3f}  AUC={a:.3f}")
    ens_prob = (oof["pred_prob_mean"] + oof["pred_prob_max"]) / 2.0
    ens_pred = (ens_prob >= args.threshold).astype(int)
    m_ens = matthews_corrcoef(oof["label"], ens_pred)
    a_ens = roc_auc_score(oof["label"], ens_prob)
    f_ens = f1_score(oof["label"], ens_pred, zero_division=0)
    print(f"  ensemble: MCC={m_ens:.3f}  F1={f_ens:.3f}  AUC={a_ens:.3f}")

    # Per-phenotype
    print("\n=== PER PHENOTYPE (ensemble) ===")
    print(f"  {'phenotype':<10} {'n':>4} {'n_pos':>6} {'mcc':>8} {'f1':>8} {'auc':>8}")
    pheno_rows = []
    for ph, g in oof.groupby("phenotype"):
        gp = (g["pred_prob_mean"] + g["pred_prob_max"]) / 2.0
        gp_lbl = (gp >= args.threshold).astype(int)
        m_p = matthews_corrcoef(g["label"], gp_lbl) if len(np.unique(g["label"])) >= 2 else float("nan")
        a_p = roc_auc_score(g["label"], gp) if len(np.unique(g["label"])) >= 2 else float("nan")
        f_p = f1_score(g["label"], gp_lbl, zero_division=0)
        m_str = f"{m_p:.3f}" if not np.isnan(m_p) else "n/a"
        a_str = f"{a_p:.3f}" if not np.isnan(a_p) else "n/a"
        print(f"  {ph:<10} {len(g):>4d} {int(g['label'].sum()):>6d} {m_str:>8s} {f_p:.3f} {a_str:>8s}")
        pheno_rows.append({
            "phenotype": ph, "n": len(g), "n_pos": int(g["label"].sum()),
            "mcc": m_p, "f1": f_p, "auc": a_p,
        })

    eval_dir = out_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(pheno_rows).to_csv(eval_dir / "svm_per_phenotype.csv", index=False)
    print(f"\n→ {display_path(eval_dir / 'svm_per_phenotype.csv')}")
    print("Done.")


if __name__ == "__main__":
    main()
