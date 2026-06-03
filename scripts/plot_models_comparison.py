"""
plot_models_comparison.py — aggregate comparison of all classifiers with LONO.

Reads every available oof_predictions_*.csv file under data/models/ and
data/models_hv/, and produces:
  - an overall metrics table (MCC, F1, AUC, accuracy) for each model,
    pool (mean/max), and ensemble (mean+max)/2
  - a per-phenotype metrics table
  - a horizontal bar chart of overall MCC per model
  - a grouped bar chart of MCC per phenotype across models

Models recognised:
  XGBoost                  (oof_predictions.csv)
  XGBoost LOO              (oof_predictions_loo.csv)
  XGBoost tuned            (oof_predictions_xgb_tuned.csv)
  XGBoost HV               (data/models_hv/oof_predictions.csv)
  XGBoost HV LOO           (data/models_hv/oof_predictions_loo.csv)
  Logistic Regression      (oof_predictions_logistic.csv)
  SVM RBF                  (oof_predictions_svm.csv)
  SupCon                   (oof_predictions_supcon.csv)

Missing files are skipped silently. The script works incrementally: as
results from the cluster (XGBoost grid, SupCon) get synced, just rerun it.

Outputs (CSV + PNG):
  data/models/eval/all_models_overall.csv
  data/models/eval/all_models_per_phenotype.csv
  data/models/eval/all_models_overall.png
  data/models/eval/all_models_per_phenotype.png

Usage:
  cd thesis/
  python scripts/plot_models_comparison.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def display_path(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(p.resolve())


# ─── Metrics implemented manually so the script has no sklearn dependency ───

def mcc(y, p):
    """Matthews correlation coefficient. Returns NaN when undefined."""
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
    """ROC-AUC computed via the Mann-Whitney U statistic with tie handling."""
    y = np.asarray(y, dtype=int); prob = np.asarray(prob, dtype=float)
    if len(np.unique(y)) < 2:
        return float("nan")
    order = np.argsort(-prob)
    sorted_probs = prob[order]
    n = len(prob)
    raw_ranks = np.arange(1, n + 1, dtype=float)
    # Tie handling: assign average rank to runs of equal probabilities
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_probs[j + 1] == sorted_probs[i]:
            j += 1
        if j > i:
            avg = raw_ranks[i:j + 1].mean()
            raw_ranks[i:j + 1] = avg
        i = j + 1
    ranks = np.empty(n, dtype=float)
    ranks[order] = raw_ranks
    n_pos = int(y.sum()); n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    sum_ranks_pos = ranks[y == 1].sum()
    U = n_pos * n_neg - (sum_ranks_pos - n_pos * (n_pos + 1) / 2)
    return U / (n_pos * n_neg)


# Registry of (display_name, relative_path, validation_label, short_id)
MODEL_REGISTRY = [
    ("XGBoost",            "data/models/oof_predictions.csv",         "LONO", "xgb"),
    ("XGBoost (LOO)",      "data/models/oof_predictions_loo.csv",     "LOO",  "xgb_loo"),
    ("XGBoost tuned",      "data/models/oof_predictions_xgb_tuned.csv", "LONO", "xgb_tuned"),
    ("XGBoost HV",         "data/models_hv/oof_predictions.csv",      "LONO", "xgb_hv"),
    ("XGBoost HV (LOO)",   "data/models_hv/oof_predictions_loo.csv",  "LOO",  "xgb_hv_loo"),
    ("Logistic Regression","data/models/oof_predictions_logistic.csv","LONO", "logistic"),
    ("SVM RBF",            "data/models/oof_predictions_svm.csv",     "LONO", "svm"),
    ("SupCon",             "data/models/oof_predictions_supcon.csv",  "LONO", "supcon"),
]


def evaluate_one(df, prob_col, threshold=0.5):
    """Return overall metrics for one (df, probability column) combination."""
    y = df["label"].to_numpy()
    if prob_col not in df.columns:
        return None
    prob = df[prob_col].to_numpy()
    pred = (prob >= threshold).astype(int)
    return {
        "mcc": mcc(y, pred),
        "f1":  f1(y, pred),
        "auc": auc(y, prob),
        "accuracy": float((pred == y).mean()),
        "n": len(df), "n_pos": int(y.sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out-dir", default=str(ROOT / "data" / "models" / "eval"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    pheno_rows = []
    found_models = []

    print(f"\nScanning {ROOT}/data/models[_hv]/ for predictions...\n")

    for name, rel_path, valid, short in MODEL_REGISTRY:
        path = ROOT / rel_path
        if not path.exists():
            print(f"  [--]  {name}: missing ({rel_path}) — skipped")
            continue
        df = pd.read_csv(path)
        print(f"  [OK]  {name}: {len(df)} pairs")
        found_models.append((name, df, valid, short))

        # Overall metrics for each pool present, plus ensemble (mean+max)/2
        for col_prob, col_pred, pool_lbl in [
            ("pred_prob_mean", "pred_label_mean", "mean"),
            ("pred_prob_max",  "pred_label_max",  "max"),
        ]:
            if col_prob not in df.columns:
                continue
            r = evaluate_one(df, col_prob, args.threshold)
            if r is None:
                continue
            rows.append({"model": name, "valid": valid, "pool": pool_lbl, **r})
        if "pred_prob_mean" in df.columns and "pred_prob_max" in df.columns:
            df = df.copy()
            df["pred_prob_ensemble"] = (df["pred_prob_mean"] + df["pred_prob_max"]) / 2.0
            r = evaluate_one(df, "pred_prob_ensemble", args.threshold)
            if r is not None:
                rows.append({"model": name, "valid": valid, "pool": "ensemble", **r})

        # Per-phenotype metrics
        for ph, g in df.groupby("phenotype"):
            for col_prob, col_pred, pool_lbl in [
                ("pred_prob_mean", "pred_label_mean", "mean"),
                ("pred_prob_max",  "pred_label_max",  "max"),
            ]:
                if col_prob not in g.columns:
                    continue
                r = evaluate_one(g, col_prob, args.threshold)
                if r is None:
                    continue
                pheno_rows.append({"model": name, "valid": valid, "pool": pool_lbl,
                                  "phenotype": ph, **r})
            if "pred_prob_mean" in g.columns and "pred_prob_max" in g.columns:
                gg = g.copy()
                gg["pred_prob_ensemble"] = (gg["pred_prob_mean"] + gg["pred_prob_max"]) / 2.0
                r = evaluate_one(gg, "pred_prob_ensemble", args.threshold)
                if r is not None:
                    pheno_rows.append({"model": name, "valid": valid, "pool": "ensemble",
                                      "phenotype": ph, **r})

    overall = pd.DataFrame(rows)
    pheno_df = pd.DataFrame(pheno_rows)

    cols = ["model", "valid", "pool", "n", "n_pos", "mcc", "f1", "auc", "accuracy"]
    overall = overall[[c for c in cols if c in overall.columns]]
    overall_path = out_dir / "all_models_overall.csv"
    overall.to_csv(overall_path, index=False)
    print(f"\n→ {display_path(overall_path)}")

    pheno_cols = ["model", "valid", "pool", "phenotype", "n", "n_pos", "mcc", "f1", "auc"]
    pheno_df = pheno_df[[c for c in pheno_cols if c in pheno_df.columns]]
    pheno_path = out_dir / "all_models_per_phenotype.csv"
    pheno_df.to_csv(pheno_path, index=False)
    print(f"→ {display_path(pheno_path)}")

    print(f"\n=== OVERALL (threshold={args.threshold}) ===")
    print(overall.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # Plot 1: best ensemble MCC per model
    overall_ens = overall[overall["pool"] == "ensemble"].copy()
    if len(overall_ens) == 0:
        # Some models (e.g. SupCon) have only one pool — keep the highest MCC row
        overall_ens = overall.loc[overall.groupby("model")["mcc"].idxmax()].copy()
    overall_ens = overall_ens.sort_values("mcc", ascending=True)
    if len(overall_ens) > 0:
        fig, ax = plt.subplots(figsize=(10, max(4, len(overall_ens) * 0.6)))
        labels = [f"{m} ({v})" for m, v in zip(overall_ens["model"], overall_ens["valid"])]
        bars = ax.barh(labels, overall_ens["mcc"],
                      color=["#1f77b4" if v == "LONO" else "#aec7e8"
                             for v in overall_ens["valid"]])
        for bar, mc in zip(bars, overall_ens["mcc"]):
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                   f"{mc:.3f}", va="center", fontsize=10)
        ax.set_xlabel("MCC", fontsize=11)
        ax.set_xlim(0, 1.0)
        ax.set_title(f"Model comparison (best pool / ensemble, threshold={args.threshold})",
                    fontsize=12, pad=12)
        ax.grid(True, alpha=0.25, axis="x", linestyle="--")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        from matplotlib.patches import Patch
        legend_elems = [Patch(facecolor="#1f77b4", label="LONO (generalises to a new NLR)"),
                       Patch(facecolor="#aec7e8", label="LOO (matrix completion)")]
        ax.legend(handles=legend_elems, loc="lower right", fontsize=9)
        fig.tight_layout()
        fig.savefig(out_dir / "all_models_overall.png", dpi=150)
        plt.close(fig)
        print(f"\n→ {display_path(out_dir / 'all_models_overall.png')}")

    # Plot 2: per-phenotype MCC, LONO ensemble where available
    pheno_lono_ens = pheno_df[(pheno_df["valid"] == "LONO") & (pheno_df["pool"] == "ensemble")].copy()
    if len(pheno_lono_ens) == 0:
        pheno_lono_ens = pheno_df[pheno_df["valid"] == "LONO"].copy()
    if len(pheno_lono_ens) > 0:
        pivot = pheno_lono_ens.pivot_table(index="phenotype", columns="model", values="mcc")
        pheno_order = ["E1", "E3", "E2", "D1", "D2", "d3", "e4"]
        pivot = pivot.reindex([p for p in pheno_order if p in pivot.index])
        fig, ax = plt.subplots(figsize=(11, 6))
        pivot.plot(kind="bar", ax=ax, width=0.85, edgecolor="white")
        ax.set_ylabel("MCC", fontsize=11)
        ax.set_xlabel("Phenotype", fontsize=11)
        ax.set_title("MCC per phenotype (LONO, ensemble where available)", fontsize=12, pad=12)
        ax.set_ylim(-0.1, 1.05)
        ax.axhline(0, color="gray", linewidth=0.6)
        ax.grid(True, alpha=0.25, axis="y", linestyle="--")
        ax.legend(loc="upper right", fontsize=9, ncol=2)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        plt.xticks(rotation=0)
        fig.tight_layout()
        fig.savefig(out_dir / "all_models_per_phenotype.png", dpi=150)
        plt.close(fig)
        print(f"→ {display_path(out_dir / 'all_models_per_phenotype.png')}")

    print("\nDone.")


if __name__ == "__main__":
    main()
