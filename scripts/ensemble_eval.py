"""
ensemble_eval.py — Ensemble (mean+max), threshold tuning, and whole-seq vs HV comparison.

Inputs:
  data/models/oof_predictions.csv      (whole-seq LONO)
  data/models_hv/oof_predictions.csv   (HV LONO)
  data/models/oof_predictions_loo.csv      (whole-seq LOO, optional)
  data/models_hv/oof_predictions_loo.csv   (HV LOO, optional)

Performs:
  1. Ensemble: prob_ens = (prob_mean + prob_max) / 2 → MCC, F1, AUC.
  2. Global threshold tuning: scan 0.05..0.95 in steps of 0.01, pick the value that maximises MCC.
  3. Whole-seq vs HV comparison in one table (overall + per phenotype).
  4. (If LOO results are present) LONO vs LOO comparison in a second table.

Outputs (CSV files in data/models/eval/):
  ensemble_summary.csv         — overall MCC/F1/AUC for each pool/ensemble × validation × dataset
  threshold_scan.csv           — MCC vs threshold curve per variant
  per_phenotype.csv            — per-phenotype MCC/AUC for all variants
  lono_vs_loo.csv              — (if LOO results are available) comparison

Usage:
  cd thesis/
  python scripts/ensemble_eval.py
  python scripts/ensemble_eval.py --threshold 0.4   # change base threshold
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def mcc(y, p):
    """Matthews correlation coefficient — manual implementation, no sklearn."""
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=int)
    tp = int(((p == 1) & (y == 1)).sum())
    tn = int(((p == 0) & (y == 0)).sum())
    fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    den = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    if den == 0:
        return float("nan")
    return (tp * tn - fp * fn) / den


def f1(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=int)
    tp = int(((p == 1) & (y == 1)).sum())
    fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    if tp == 0:
        return 0.0 if fp + fn else float("nan")
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    return 2 * prec * rec / (prec + rec)


def auc(y, prob):
    """ROC-AUC — manual implementation. Returns NaN when only one class is present."""
    y = np.asarray(y, dtype=int)
    prob = np.asarray(prob, dtype=float)
    if len(np.unique(y)) < 2:
        return float("nan")
    # Sort by prob descending
    order = np.argsort(-prob)
    y_sorted = y[order]
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    # AUC = sum_{i:pos} rank_i / (n_pos*n_neg) - (n_pos+1)/(2*n_neg)
    # Easier: rank-based Mann-Whitney U formula
    ranks = np.empty(len(y), dtype=float)
    # Ranks 1..N (highest probability gets rank 1)
    # Tie handling: use average rank for runs of equal probabilities
    # Keep it simple: assign mean rank to tied values
    sorted_probs = prob[order]
    # average ranks
    n = len(prob)
    raw_ranks = np.arange(1, n + 1, dtype=float)
    # ties handling
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_probs[j + 1] == sorted_probs[i]:
            j += 1
        if j > i:
            avg = raw_ranks[i:j + 1].mean()
            raw_ranks[i:j + 1] = avg
        i = j + 1
    ranks[order] = raw_ranks
    sum_ranks_pos = ranks[y == 1].sum()
    # auc = (sum_ranks_pos - n_pos*(n_pos+1)/2) / (n_pos*n_neg)
    # but we invert because rank 1 = highest prob = positives
    # if positives end up with lower ranks (better sorted), AUC > 0.5
    # Mann-Whitney: U = sum_ranks_pos - n_pos*(n_pos+1)/2  (when rank 1 = lowest)
    # Here rank 1 = highest, so we invert: U = n_pos*n_neg - (sum - n_pos*(n_pos+1)/2)
    # Therefore AUC = (n_pos*n_neg - (sum_ranks_pos - n_pos*(n_pos+1)/2)) / (n_pos*n_neg)
    U = n_pos * n_neg - (sum_ranks_pos - n_pos * (n_pos + 1) / 2)
    return U / (n_pos * n_neg)


def load_oof(path: Path):
    if not path.exists():
        return None
    return pd.read_csv(path)


def evaluate_block(df: pd.DataFrame, prob_col: str, threshold: float, label: str):
    """Returns a dictionary of overall metrics."""
    y = df["label"].to_numpy()
    prob = df[prob_col].to_numpy()
    pred = (prob >= threshold).astype(int)
    return {
        "variant": label,
        "n": len(df),
        "n_pos": int(y.sum()),
        "threshold": threshold,
        "mcc": mcc(y, pred),
        "f1": f1(y, pred),
        "auc": auc(y, prob),
        "accuracy": float((pred == y).mean()),
    }


def per_pheno_block(df: pd.DataFrame, prob_col: str, threshold: float, label: str):
    rows = []
    for ph, g in df.groupby("phenotype"):
        y = g["label"].to_numpy()
        prob = g[prob_col].to_numpy()
        pred = (prob >= threshold).astype(int)
        rows.append({
            "variant": label,
            "phenotype": ph,
            "n": len(g),
            "n_pos": int(y.sum()),
            "mcc": mcc(y, pred),
            "f1": f1(y, pred),
            "auc": auc(y, prob),
            "n_correct": int((pred == y).sum()),
        })
    return rows


def threshold_scan(df: pd.DataFrame, prob_col: str, label: str):
    rows = []
    y = df["label"].to_numpy()
    prob = df[prob_col].to_numpy()
    for t in np.arange(0.05, 0.96, 0.01):
        pred = (prob >= t).astype(int)
        rows.append({
            "variant": label,
            "threshold": round(float(t), 2),
            "mcc": mcc(y, pred),
            "f1": f1(y, pred),
            "n_pred_pos": int(pred.sum()),
        })
    return rows


def add_ensemble(df: pd.DataFrame):
    """Adds a pred_prob_ensemble = (mean + max) / 2 column."""
    df = df.copy()
    df["pred_prob_ensemble"] = (df["pred_prob_mean"] + df["pred_prob_max"]) / 2.0
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out-dir", default=str(ROOT / "data" / "models" / "eval"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load all 4 possible prediction files
    files = {
        "whole-seq LONO": ROOT / "data" / "models" / "oof_predictions.csv",
        "HV LONO":        ROOT / "data" / "models_hv" / "oof_predictions.csv",
        "whole-seq LOO":  ROOT / "data" / "models" / "oof_predictions_loo.csv",
        "HV LOO":         ROOT / "data" / "models_hv" / "oof_predictions_loo.csv",
    }
    loaded = {}
    for k, p in files.items():
        df = load_oof(p)
        if df is not None:
            loaded[k] = add_ensemble(df)
            print(f"[OK]  {k}: {len(df)} pairs   ({p.relative_to(ROOT)})")
        else:
            print(f"[--]  {k}: missing ({p.relative_to(ROOT)}) — skipped")

    if not loaded:
        print("No oof_predictions found. Run train_lono.py / train_loo.py first.")
        return

    # ── (1) Overall summary ───────────────────────────────────────────────
    overall_rows = []
    for ds, df in loaded.items():
        for col, lbl in [("pred_prob_mean", "mean"),
                         ("pred_prob_max", "max"),
                         ("pred_prob_ensemble", "ensemble")]:
            r = evaluate_block(df, col, args.threshold, f"{ds} | {lbl}")
            overall_rows.append(r)
    overall = pd.DataFrame(overall_rows)
    overall_path = out_dir / "ensemble_summary.csv"
    overall.to_csv(overall_path, index=False)
    print(f"\n=== OVERALL (threshold={args.threshold}) ===")
    print(overall.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\n→ {overall_path.relative_to(ROOT)}")

    # ── (2) Threshold scan ────────────────────────────────────────────────
    scan_rows = []
    for ds, df in loaded.items():
        for col, lbl in [("pred_prob_mean", "mean"),
                         ("pred_prob_max", "max"),
                         ("pred_prob_ensemble", "ensemble")]:
            scan_rows.extend(threshold_scan(df, col, f"{ds} | {lbl}"))
    scan = pd.DataFrame(scan_rows)
    scan_path = out_dir / "threshold_scan.csv"
    scan.to_csv(scan_path, index=False)
    print(f"\n=== BEST THRESHOLD per variant (max MCC) ===")
    best = scan.loc[scan.groupby("variant")["mcc"].idxmax()].reset_index(drop=True)
    print(best[["variant", "threshold", "mcc", "f1", "n_pred_pos"]].to_string(index=False,
        float_format=lambda x: f"{x:.3f}"))
    print(f"\n→ {scan_path.relative_to(ROOT)}")

    # ── (3) Per-phenotype ─────────────────────────────────────────────────
    pheno_rows = []
    for ds, df in loaded.items():
        for col, lbl in [("pred_prob_mean", "mean"),
                         ("pred_prob_max", "max"),
                         ("pred_prob_ensemble", "ensemble")]:
            pheno_rows.extend(per_pheno_block(df, col, args.threshold, f"{ds} | {lbl}"))
    pheno = pd.DataFrame(pheno_rows)
    pheno_path = out_dir / "per_phenotype.csv"
    pheno.to_csv(pheno_path, index=False)
    print(f"\n=== PER FENOTYP (threshold={args.threshold}) — only ensemble ===")
    pheno_ens = pheno[pheno["variant"].str.contains("ensemble")].copy()
    print(pheno_ens.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\n→ {pheno_path.relative_to(ROOT)}")

    # ── (4) LONO vs LOO comparison (if LOO are) ─────────────────────────
    lono_keys = [k for k in loaded if "LONO" in k]
    loo_keys = [k for k in loaded if "LOO" in k]
    if loo_keys:
        cmp_rows = []
        for ds in loo_keys:
            counterpart = ds.replace("LOO", "LONO")
            if counterpart not in loaded:
                continue
            for col in ["pred_prob_mean", "pred_prob_max", "pred_prob_ensemble"]:
                lbl = col.replace("pred_prob_", "")
                # Per phenotype
                df_lono = loaded[counterpart]
                df_loo = loaded[ds]
                for ph in sorted(set(df_lono["phenotype"]) | set(df_loo["phenotype"])):
                    g_lono = df_lono[df_lono["phenotype"] == ph]
                    g_loo = df_loo[df_loo["phenotype"] == ph]
                    if len(g_lono) == 0 or len(g_loo) == 0:
                        continue
                    p_lono = (g_lono[col] >= args.threshold).astype(int)
                    p_loo = (g_loo[col] >= args.threshold).astype(int)
                    mcc_lono = mcc(g_lono["label"], p_lono)
                    mcc_loo = mcc(g_loo["label"], p_loo)
                    cmp_rows.append({
                        "dataset": ds.replace(" LOO", ""),
                        "pool": lbl,
                        "phenotype": ph,
                        "n": len(g_lono),
                        "n_pos": int(g_lono["label"].sum()),
                        "mcc_lono": mcc_lono,
                        "mcc_loo": mcc_loo,
                        "delta": (mcc_loo - mcc_lono) if not (np.isnan(mcc_lono) or np.isnan(mcc_loo)) else float("nan"),
                    })
        if cmp_rows:
            cmp_df = pd.DataFrame(cmp_rows)
            cmp_path = out_dir / "lono_vs_loo.csv"
            cmp_df.to_csv(cmp_path, index=False)
            print(f"\n=== LONO vs LOO ===")
            print(cmp_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
            print(f"\n→ {cmp_path.relative_to(ROOT)}")
    else:
        print("\n(LOO results still not run — comparison LONO/LOO skipped.)")

    print("\nDone.")


if __name__ == "__main__":
    main()
