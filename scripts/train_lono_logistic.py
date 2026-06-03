"""
train_lono_logistic.py — Logistic regression baseline with LONO, pure NumPy.

Implemented from scratch with no sklearn dependency:
  - L2 regularisation (ridge)
  - Class weights to handle the 1:3 class imbalance
  - Mini-batch Adam optimizer
  - Reads features_mean.npz / features_max.npz produced by build_pair_features.py
  - 25-fold leave-one-NLR-out cross-validation (same protocol as train_lono.py)

Outputs (compatible with ensemble_eval.py):
  data/models/oof_predictions_logistic.csv      275 rows with pred_prob_*, pred_label_*
  data/models/fold_summary_logistic.csv         per-fold MCC, F1, n_train, n_test
  data/models/eval/logistic_per_phenotype.csv   overall + per-phenotype metrics

Usage:
  cd thesis/
  python scripts/train_lono_logistic.py
  python scripts/train_lono_logistic.py --epochs 1000 --lr 0.001

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
    """Standardise using train statistics, then apply the same transform to test."""
    mu = X_train.mean(axis=0, keepdims=True)
    sd = X_train.std(axis=0, keepdims=True) + 1e-8
    return (X_train - mu) / sd, (X_test - mu) / sd


def sigmoid(z):
    # numerically stable sigmoid
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[~pos])
    out[~pos] = exp_z / (1.0 + exp_z)
    return out


def fit_logistic_l2(X, y, lr=0.01, epochs=500, lam=1.0, class_weight=True,
                   batch_size=64, seed=42):
    """Logistic regression with L2 regularisation, optimised by Adam.

    X: (N, d) feature matrix (should be standardised)
    y: (N,) labels {0, 1}
    lr: learning rate
    epochs: number of epochs
    lam: L2 strength (larger = stronger regularisation)
    class_weight: when True, sample weights are set inversely proportional
        to class frequency
    batch_size: mini-batch size
    Returns: (w, b) — weight vector and bias scalar
    """
    rng = np.random.RandomState(seed)
    n, d = X.shape
    w = np.zeros(d, dtype=np.float64)
    b = 0.0

    # Adam state
    m_w = np.zeros(d); v_w = np.zeros(d)
    m_b = 0.0; v_b = 0.0
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    # Class weights
    if class_weight:
        n_pos = y.sum()
        n_neg = n - n_pos
        w_pos = n / (2.0 * max(n_pos, 1))
        w_neg = n / (2.0 * max(n_neg, 1))
        sample_w = np.where(y == 1, w_pos, w_neg)
    else:
        sample_w = np.ones(n)

    for epoch in range(epochs):
        # Shuffle
        idx = rng.permutation(n)
        for start in range(0, n, batch_size):
            ii = idx[start:start + batch_size]
            xb, yb, sb = X[ii], y[ii], sample_w[ii]
            # Forward
            z = xb @ w + b
            p = sigmoid(z)
            # Gradient (weighted BCE + L2)
            err = (p - yb) * sb
            grad_w = (xb.T @ err) / len(ii) + lam * w / n
            grad_b = err.mean()
            # Adam update
            t += 1
            m_w = beta1 * m_w + (1 - beta1) * grad_w
            v_w = beta2 * v_w + (1 - beta2) * (grad_w ** 2)
            m_b = beta1 * m_b + (1 - beta1) * grad_b
            v_b = beta2 * v_b + (1 - beta2) * (grad_b ** 2)
            m_hat_w = m_w / (1 - beta1 ** t)
            v_hat_w = v_w / (1 - beta2 ** t)
            m_hat_b = m_b / (1 - beta1 ** t)
            v_hat_b = v_b / (1 - beta2 ** t)
            w -= lr * m_hat_w / (np.sqrt(v_hat_w) + eps)
            b -= lr * m_hat_b / (np.sqrt(v_hat_b) + eps)
    return w, b


def predict_proba(X, w, b):
    return sigmoid(X @ w + b)


def mcc(y, p):
    y = np.asarray(y, dtype=int); p = np.asarray(p, dtype=int)
    tp = int(((p == 1) & (y == 1)).sum())
    tn = int(((p == 0) & (y == 0)).sum())
    fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    den = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    if den == 0:
        return float("nan")
    return (tp * tn - fp * fn) / den


def f1(y, p):
    y = np.asarray(y, dtype=int); p = np.asarray(p, dtype=int)
    tp = int(((p == 1) & (y == 1)).sum())
    fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    if tp == 0:
        return 0.0
    prec = tp / (tp + fp); rec = tp / (tp + fn)
    return 2 * prec * rec / (prec + rec)


def auc(y, prob):
    y = np.asarray(y, dtype=int); prob = np.asarray(prob, dtype=float)
    if len(np.unique(y)) < 2:
        return float("nan")
    order = np.argsort(-prob)
    sorted_probs = prob[order]
    n = len(prob)
    ranks_sorted = np.arange(1, n + 1, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_probs[j + 1] == sorted_probs[i]:
            j += 1
        if j > i:
            avg = ranks_sorted[i:j + 1].mean()
            ranks_sorted[i:j + 1] = avg
        i = j + 1
    ranks = np.empty(n, dtype=float)
    ranks[order] = ranks_sorted
    n_pos = int(y.sum()); n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    sum_ranks_pos = ranks[y == 1].sum()
    U = n_pos * n_neg - (sum_ranks_pos - n_pos * (n_pos + 1) / 2)
    return U / (n_pos * n_neg)


def run_lono(pack: dict, args, pool_name: str):
    X = pack["X"].astype(np.float32)
    y = pack["y"].astype(int)
    nlr_id = pack["nlr_id"].astype(str)
    phenotype = pack["phenotype"].astype(str)
    confidence = pack["confidence"].astype(str)
    unique_nlrs = sorted(set(nlr_id))
    n_folds = len(unique_nlrs)
    print(f"\n=== Logistic LONO: {pool_name} pool, {n_folds} folds ===")

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
        # Training
        w, b = fit_logistic_l2(X_tr_s, y_tr, lr=args.lr, epochs=args.epochs,
                               lam=args.l2, class_weight=True,
                               batch_size=args.batch_size, seed=args.seed)
        # Prediction
        probas = predict_proba(X_te_s, w, b)
        preds = (probas >= args.threshold).astype(int)
        proba_oof[test_mask] = probas
        pred_oof[test_mask] = preds
        fold_oof[test_mask] = held_out
        # Per-fold metrics
        m = mcc(y[test_mask], preds)
        fold_rows.append({
            "fold_test_nlr": held_out,
            "phenotype": phenotype[test_mask][0],
            "confidence": confidence[test_mask][0],
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
            "mcc": m,
            "f1": f1(y[test_mask], preds),
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
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pack_mean = load_features(Path(args.features_mean))
    pack_max  = load_features(Path(args.features_max))

    proba_m, pred_m, fold_m, fold_df_m = run_lono(pack_mean, args, "mean")
    proba_x, pred_x, fold_x, fold_df_x = run_lono(pack_max,  args, "max")

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
        "fold_test_nlr":   fold_m,
    })
    oof_path = out_dir / "oof_predictions_logistic.csv"
    oof.to_csv(oof_path, index=False)
    print(f"\nOOF predictions → {display_path(oof_path)}")

    # Fold summary
    fold_df_m["pool"] = "mean"
    fold_df_x["pool"] = "max"
    fold_summary = pd.concat([fold_df_m, fold_df_x], ignore_index=True)
    fold_path = out_dir / "fold_summary_logistic.csv"
    fold_summary.to_csv(fold_path, index=False)
    print(f"Fold summary → {display_path(fold_path)}")

    # Overall metrics
    print("\n=== OVERALL ===")
    for pool_name, prob_col, pred_col in [
        ("mean", "pred_prob_mean", "pred_label_mean"),
        ("max",  "pred_prob_max",  "pred_label_max"),
    ]:
        m = mcc(oof["label"], oof[pred_col])
        a = auc(oof["label"], oof[prob_col])
        f = f1(oof["label"], oof[pred_col])
        print(f"  {pool_name:>4s} pool: MCC={m:.3f}  F1={f:.3f}  AUC={a:.3f}")

    # Ensemble
    ens_prob = (oof["pred_prob_mean"] + oof["pred_prob_max"]) / 2.0
    ens_pred = (ens_prob >= args.threshold).astype(int)
    m_ens = mcc(oof["label"], ens_pred)
    a_ens = auc(oof["label"], ens_prob)
    f_ens = f1(oof["label"], ens_pred)
    print(f"  ensemble: MCC={m_ens:.3f}  F1={f_ens:.3f}  AUC={a_ens:.3f}")

    # Per-phenotype ensemble
    print("\n=== PER PHENOTYPE (ensemble) ===")
    print(f"  {'phenotype':<10} {'n':>4} {'n_pos':>6} {'mcc':>8} {'f1':>8} {'auc':>8}")
    pheno_rows = []
    for ph, g in oof.groupby("phenotype"):
        gp = (g["pred_prob_mean"] + g["pred_prob_max"]) / 2.0
        gp_lbl = (gp >= args.threshold).astype(int)
        m_p = mcc(g["label"], gp_lbl)
        a_p = auc(g["label"], gp)
        f_p = f1(g["label"], gp_lbl)
        m_str = f"{m_p:.3f}" if not np.isnan(m_p) else "n/a"
        a_str = f"{a_p:.3f}" if not np.isnan(a_p) else "n/a"
        print(f"  {ph:<10} {len(g):>4d} {int(g['label'].sum()):>6d} {m_str:>8s} {f_p:.3f} {a_str:>8s}")
        pheno_rows.append({
            "phenotype": ph, "n": len(g), "n_pos": int(g["label"].sum()),
            "mcc": m_p, "f1": f_p, "auc": a_p,
        })

    eval_dir = out_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(pheno_rows).to_csv(eval_dir / "logistic_per_phenotype.csv", index=False)
    print(f"\n→ {display_path(eval_dir / 'logistic_per_phenotype.csv')}")
    print("Done.")


if __name__ == "__main__":
    main()
