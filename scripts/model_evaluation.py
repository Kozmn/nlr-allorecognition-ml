"""
model_evaluation.py — evaluate Model 1 (XGBoost with leave-one-NLR-out CV).

Aggregates the out-of-fold predictions written by train_lono.py and produces:

METRICS (text report):
  - Overall: MCC, F1 (weighted + macro), precision, recall, AUC
  - Per phenotype: MCC, F1
  - Per fold (one held-out NLR): MCC, F1
  - mean-pool vs max-pool comparison

FIGURES (written to data/validation/reports/model1/):
  01_confusion_matrix.png       two confusion matrices (mean / max pool)
  02_roc_curves.png             two ROC curves on one plot
  03_per_fold_mcc.png           bar chart: MCC per NLR (25 bars x 2 poolings)
  04_per_phenotype_metrics.png  MCC + F1 for each phenotype (7 groups)
  05_calibration.png            probability calibration
  06_feature_importance.png     top 40 features (mean vs max comparison)
  07_pool_comparison.png        scatter: pred_prob_mean vs pred_prob_max

REPORT:
  data/validation/reports/model1/REPORT.md

Inputs:
  data/models/oof_predictions.csv
  data/models/fold_summary.csv
  data/models/xgb_feature_importance.csv

Usage:
  cd thesis/
  python scripts/model_evaluation.py

Author: Kacper Koźmin
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "data" / "models"
REPORT_DIR = ROOT / "data" / "validation" / "reports" / "model1"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Colours consistent with the rest of the pipeline (see aggregate_heatmaps.py)
COLOR_MEAN = "#2257b8"          # blue
COLOR_MAX  = "#c0182d"          # red
COLOR_POS  = "#1f5fcc"
COLOR_NEG  = "#bdc3c7"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


# ─────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────

def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                        y_proba: np.ndarray) -> dict:
    from sklearn.metrics import (matthews_corrcoef, f1_score, precision_score,
                                 recall_score, roc_auc_score, accuracy_score)
    return {
        "n": len(y_true),
        "n_pos": int(y_true.sum()),
        "n_neg": int((y_true == 0).sum()),
        "accuracy":     accuracy_score(y_true, y_pred),
        "mcc":          matthews_corrcoef(y_true, y_pred),
        "f1_weighted":  f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_macro":     f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_pos":       f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "precision":    precision_score(y_true, y_pred, zero_division=0),
        "recall":       recall_score(y_true, y_pred, zero_division=0),
        "auc": (roc_auc_score(y_true, y_proba)
                if len(np.unique(y_true)) > 1 else np.nan),
    }


# ─────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────

def plot_confusion_matrices(oof: pd.DataFrame, savepath: Path):
    from sklearn.metrics import confusion_matrix
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, pool in zip(axes, ["mean", "max"]):
        cm = confusion_matrix(oof["label"], oof[f"pred_label_{pool}"])
        im = ax.imshow(cm, cmap="Blues", aspect="equal")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, int(cm[i, j]),
                        ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black",
                        fontsize=16, fontweight="bold")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["0 (missing)", "1 (reakcja)"])
        ax.set_yticklabels(["0 (missing)", "1 (reakcja)"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Prawda")
        ax.set_title(f"{pool}-pool", fontweight="bold")
    fig.suptitle("Matrix konfuzji (out-of-fold, 275 pairs)",
                 fontsize=13, fontweight="bold", y=0.99)
    fig.tight_layout()
    fig.savefig(savepath, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → {savepath.relative_to(ROOT)}")


def plot_roc_curves(oof: pd.DataFrame, savepath: Path):
    from sklearn.metrics import roc_curve, roc_auc_score
    fig, ax = plt.subplots(figsize=(6, 5.5))
    for pool, color in [("mean", COLOR_MEAN), ("max", COLOR_MAX)]:
        fpr, tpr, _ = roc_curve(oof["label"], oof[f"pred_prob_{pool}"])
        auc = roc_auc_score(oof["label"], oof[f"pred_prob_{pool}"])
        ax.plot(fpr, tpr, color=color, linewidth=2.3,
                label=f"{pool}-pool (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#888888",
            linewidth=1, label="Losowy (AUC = 0.5)")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("Krzywe ROC — Model 1 (LONO)", fontsize=13,
                 fontweight="bold", pad=10)
    ax.legend(loc="lower right", frameon=False)
    ax.grid(linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(savepath, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → {savepath.relative_to(ROOT)}")


def plot_per_fold_mcc(fold_summary: pd.DataFrame, savepath: Path):
    # Pivot: rows = nlr, columns = pool, values = mcc
    pivot = fold_summary.pivot(index="fold_test_nlr", columns="pool",
                               values="mcc")
    # Sort by MCC(mean) ascending
    pivot = pivot.sort_values("mean")
    y = np.arange(len(pivot))
    fig, ax = plt.subplots(figsize=(9, max(5, 0.35 * len(pivot))))
    ax.barh(y - 0.2, pivot["mean"].fillna(0), 0.38,
            color=COLOR_MEAN, label="mean-pool", edgecolor="white", linewidth=0.6)
    ax.barh(y + 0.2, pivot["max"].fillna(0), 0.38,
            color=COLOR_MAX, label="max-pool", edgecolor="white", linewidth=0.6)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_xlabel("MCC na danym foldzie")
    ax.set_xlim(-1.05, 1.05)
    ax.set_title("MCC per fold (Leave-One-NLR-Out)",
                 fontsize=13, fontweight="bold", pad=10, loc="left")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(savepath, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → {savepath.relative_to(ROOT)}")


def plot_per_phenotype_metrics(oof: pd.DataFrame, savepath: Path):
    from sklearn.metrics import matthews_corrcoef, f1_score
    rows = []
    for pheno, grp in oof.groupby("phenotype"):
        for pool in ("mean", "max"):
            y_true = grp["label"].to_numpy()
            y_pred = grp[f"pred_label_{pool}"].to_numpy()
            if len(np.unique(y_true)) < 2:
                mcc = np.nan
                f1 = f1_score(y_true, y_pred, zero_division=0) \
                     if y_true.sum() > 0 or y_pred.sum() > 0 else np.nan
            else:
                mcc = matthews_corrcoef(y_true, y_pred)
                f1 = f1_score(y_true, y_pred, zero_division=0)
            rows.append({"phenotype": pheno, "pool": pool, "mcc": mcc, "f1": f1,
                         "n": len(grp), "n_pos": int(y_true.sum())})
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    phenos = sorted(df["phenotype"].unique())
    x = np.arange(len(phenos))
    for ax, metric, title in [(axes[0], "mcc", "MCC per phenotype"),
                              (axes[1], "f1",  "F1 per phenotype")]:
        for i, pool in enumerate(("mean", "max")):
            vals = [df[(df["phenotype"] == p) & (df["pool"] == pool)][metric].iloc[0]
                    for p in phenos]
            offset = -0.2 if pool == "mean" else 0.2
            color = COLOR_MEAN if pool == "mean" else COLOR_MAX
            bars = ax.bar(x + offset, [0 if np.isnan(v) else v for v in vals],
                          width=0.38, color=color, label=f"{pool}-pool",
                          edgecolor="white", linewidth=0.8)
            # Oznacz NaN slupki (jedna klasa → metryka niezdefiniowana)
            for bar, v in zip(bars, vals):
                if np.isnan(v):
                    ax.text(bar.get_x() + bar.get_width() / 2, 0.02,
                            "n/d", ha="center", va="bottom",
                            fontsize=8, color="#888888", style="italic")
        ax.set_xticks(x)
        # Add n_pos/n info below the label
        xtick_labels = []
        for p in phenos:
            n = df[df["phenotype"] == p]["n"].iloc[0]
            npos = df[df["phenotype"] == p]["n_pos"].iloc[0]
            xtick_labels.append(f"{p}\n{npos}/{n}")
        ax.set_xticklabels(xtick_labels, fontsize=9)
        ax.set_ylim(-0.05 if metric == "mcc" else 0, 1.05)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        if ax is axes[0]:
            ax.legend(loc="upper right", frameon=False, fontsize=9)
    fig.suptitle("Metryki per phenotype (pozytywow/ogolem pod labelem)",
                 fontsize=13, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(savepath, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → {savepath.relative_to(ROOT)}")


def plot_calibration(oof: pd.DataFrame, savepath: Path, n_bins: int = 10):
    fig, ax = plt.subplots(figsize=(6, 5.5))
    for pool, color in [("mean", COLOR_MEAN), ("max", COLOR_MAX)]:
        probas = oof[f"pred_prob_{pool}"].to_numpy()
        labels = oof["label"].to_numpy()
        bins = np.linspace(0, 1, n_bins + 1)
        idx = np.digitize(probas, bins) - 1
        idx = np.clip(idx, 0, n_bins - 1)
        bin_true, bin_pred, bin_count = [], [], []
        for b in range(n_bins):
            m = (idx == b)
            if m.any():
                bin_true.append(labels[m].mean())
                bin_pred.append(probas[m].mean())
                bin_count.append(int(m.sum()))
        ax.plot(bin_pred, bin_true, "o-", color=color, linewidth=2,
                markersize=8, label=f"{pool}-pool")
    ax.plot([0, 1], [0, 1], "--", color="#888888", linewidth=1,
            label="Idealna kalibracja")
    ax.set_xlabel("Srednie przewidywane prawdopodobienstwo")
    ax.set_ylabel("Rzeczywisty odsetek pozytywow")
    ax.set_title("Krzywa kalibracji", fontsize=13, fontweight="bold", pad=10)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(loc="upper left", frameon=False)
    ax.grid(linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(savepath, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → {savepath.relative_to(ROOT)}")


def plot_feature_importance(imp_df: pd.DataFrame, savepath: Path, top_k: int = 40):
    """Top K najwazniejszych cech (aggregation po foldach).
    Features 0..d-1: e_NLR
    Features d..2d-1: e_HC
    Features 2d..3d-1: |e_NLR - e_HC|
    Features 3d..4d-1: e_NLR * e_HC
    """
    d = len(imp_df) // 4

    def feature_label(idx: int) -> str:
        block = idx // d
        within = idx % d
        names = ["e_NLR", "e_HC", "|ΔNLR,HC|", "NLR⊙HC"]
        return f"{names[block]}[{within}]"

    # Top for mean
    top_mean = imp_df.nlargest(top_k, "importance_mean").copy()
    top_mean["label"] = top_mean["feature_idx"].apply(feature_label)
    top_max = imp_df.nlargest(top_k, "importance_max").copy()
    top_max["label"] = top_max["feature_idx"].apply(feature_label)

    fig, axes = plt.subplots(1, 2, figsize=(12, max(6, top_k * 0.22)))
    for ax, df, pool, color in [
        (axes[0], top_mean, "mean", COLOR_MEAN),
        (axes[1], top_max,  "max",  COLOR_MAX),
    ]:
        df_sorted = df.sort_values(f"importance_{pool}")
        ax.barh(df_sorted["label"], df_sorted[f"importance_{pool}"],
                color=color, edgecolor="white", linewidth=0.5)
        ax.set_title(f"{pool}-pool: top {top_k} cech", fontweight="bold", pad=6)
        ax.set_xlabel("Mean waznosc (XGBoost)")
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(axis="x", linestyle=":", alpha=0.4)
    fig.suptitle("Feature importance (usredniona po 25 foldach)",
                 fontsize=13, fontweight="bold", y=0.99)
    fig.tight_layout()
    fig.savefig(savepath, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → {savepath.relative_to(ROOT)}")


def plot_pool_comparison(oof: pd.DataFrame, savepath: Path):
    """Scatter pred_prob_mean vs pred_prob_max, kolorowany po prawdziwej etykiecie."""
    fig, ax = plt.subplots(figsize=(6, 6))
    for lbl, color, name in [(0, COLOR_NEG, "missing reakcji"),
                             (1, COLOR_POS, "reakcja")]:
        sub = oof[oof["label"] == lbl]
        ax.scatter(sub["pred_prob_mean"], sub["pred_prob_max"],
                   c=color, s=30, alpha=0.6, edgecolor="white", linewidth=0.5,
                   label=f"{name} (n={len(sub)})")
    ax.plot([0, 1], [0, 1], "--", color="#888888", linewidth=1)
    ax.axvline(0.5, color="#aaaaaa", linewidth=0.5, linestyle=":")
    ax.axhline(0.5, color="#aaaaaa", linewidth=0.5, linestyle=":")
    ax.set_xlabel("pred_prob (mean-pool)")
    ax.set_ylabel("pred_prob (max-pool)")
    ax.set_title("Comparison predykcji: mean vs max pool",
                 fontsize=13, fontweight="bold", pad=10)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(loc="lower right", frameon=False)
    ax.grid(linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(savepath, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → {savepath.relative_to(ROOT)}")


# ─────────────────────────────────────────────────────────────────────
# REPORT
# ─────────────────────────────────────────────────────────────────────

def write_report(oof: pd.DataFrame, fold_summary: pd.DataFrame,
                 overall_metrics: dict, savepath: Path):
    with open(savepath, "w") as f:
        f.write("# Model 1 — evaluation (ESM-C 600M + XGBoost + LONO)\n\n")
        f.write("**Task:** binary classification — does the pair (NLR WD40, HET-C) "
                "trigger an incompatibility reaction (apoptosis)?\n\n")
        f.write("**Validation:** Leave-One-NLR-Out (25 folds). Each of the 275 pairs "
                "is predicted exactly once (out-of-fold).\n\n")
        f.write("---\n\n## Overall results (275 pairs)\n\n")
        f.write("| Metryka | mean-pool | max-pool |\n|---|---:|---:|\n")
        for name, key in [
            ("N", "n"), ("Positives", "n_pos"), ("Negatives", "n_neg"),
            ("Accuracy", "accuracy"),
            ("MCC", "mcc"),
            ("F1 (weighted)", "f1_weighted"),
            ("F1 (macro)", "f1_macro"),
            ("F1 (positive)", "f1_pos"),
            ("Precision", "precision"),
            ("Recall", "recall"),
            ("AUC", "auc"),
        ]:
            v_m = overall_metrics["mean"][key]
            v_x = overall_metrics["max"][key]
            if isinstance(v_m, float) and not np.isnan(v_m):
                f.write(f"| {name} | {v_m:.3f} | {v_x:.3f} |\n")
            else:
                f.write(f"| {name} | {v_m} | {v_x} |\n")

        # Per phenotype
        f.write("\n---\n\n## Per-phenotype metrics\n\n")
        f.write("| Phenotype | n | pos | MCC (mean) | MCC (max) | F1 (mean) | F1 (max) |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        from sklearn.metrics import matthews_corrcoef, f1_score
        for pheno, grp in oof.groupby("phenotype"):
            y_true = grp["label"].to_numpy()
            npos = int(y_true.sum())
            n = len(grp)
            row = [f"{pheno}", f"{n}", f"{npos}"]
            for pool in ("mean", "max"):
                y_pred = grp[f"pred_label_{pool}"].to_numpy()
                if len(np.unique(y_true)) < 2:
                    row.append("n/a")
                else:
                    row.append(f"{matthews_corrcoef(y_true, y_pred):.3f}")
            for pool in ("mean", "max"):
                y_pred = grp[f"pred_label_{pool}"].to_numpy()
                row.append(f"{f1_score(y_true, y_pred, zero_division=0):.3f}")
            f.write("| " + " | ".join(row) + " |\n")

        # Distribution of per-fold MCC
        f.write("\n---\n\n## Distribution of MCC across folds\n\n")
        for pool in ("mean", "max"):
            fs = fold_summary[fold_summary["pool"] == pool]["mcc"].dropna()
            if len(fs) == 0:
                continue
            f.write(f"- **{pool}-pool** (n={len(fs)} folds with MCC):\n")
            f.write(f"  - Median: {fs.median():.3f}\n")
            f.write(f"  - Mean: {fs.mean():.3f} ± {fs.std():.3f}\n")
            f.write(f"  - Min:     {fs.min():.3f}\n")
            f.write(f"  - Max:     {fs.max():.3f}\n")

        # Main take-away
        f.write("\n---\n\n## Main take-away\n\n")
        best = max(overall_metrics.items(), key=lambda kv: kv[1]["mcc"])
        other = "max" if best[0] == "mean" else "mean"
        f.write(f"- Better pooling: **{best[0]}** (MCC = {best[1]['mcc']:.3f} vs "
                f"{overall_metrics[other]['mcc']:.3f} for {other}).\n")
        mcc = best[1]["mcc"]
        if mcc > 0.5:
            f.write(f"- MCC = {mcc:.2f} — **strong classification signal**. "
                    "ESM-C correctly encodes features relevant to compatibility.\n")
        elif mcc > 0.3:
            f.write(f"- MCC = {mcc:.2f} — **moderate signal**. The model beats "
                    "random classification, but is not ready for biological "
                    "applications without further validation.\n")
        elif mcc > 0.1:
            f.write(f"- MCC = {mcc:.2f} — **weak signal**. ESM-C captures some "
                    "sequence differences, but not enough to predict "
                    "compatibility. Consider V2 (different features or more data).\n")
        else:
            f.write(f"- MCC = {mcc:.2f} — **no signal**. Raw ESM-C embeddings "
                    "do not encode the information needed to distinguish "
                    "compatible from incompatible pairs.\n")
    print(f"  → {savepath.relative_to(ROOT)}")


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", default=str(MODELS_DIR / "oof_predictions.csv"))
    ap.add_argument("--fold-summary", default=str(MODELS_DIR / "fold_summary.csv"))
    ap.add_argument("--feature-importance",
                    default=str(MODELS_DIR / "xgb_feature_importance.csv"))
    ap.add_argument("--out-dir", default=str(REPORT_DIR))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    oof = pd.read_csv(args.oof)
    fold_summary = pd.read_csv(args.fold_summary)
    imp_df = pd.read_csv(args.feature_importance)

    # Overall metrics
    overall = {}
    for pool in ("mean", "max"):
        overall[pool] = compute_all_metrics(
            oof["label"].to_numpy(),
            oof[f"pred_label_{pool}"].to_numpy(),
            oof[f"pred_prob_{pool}"].to_numpy())
    print("\n=== Overall metryki ===")
    for pool, m in overall.items():
        print(f"  {pool:>4s}: MCC={m['mcc']:.3f}  F1w={m['f1_weighted']:.3f}  "
              f"F1m={m['f1_macro']:.3f}  AUC={m['auc']:.3f}")

    print("\n=== Generating plots ===")
    plot_confusion_matrices(oof, out_dir / "01_confusion_matrix.png")
    plot_roc_curves(oof,         out_dir / "02_roc_curves.png")
    plot_per_fold_mcc(fold_summary, out_dir / "03_per_fold_mcc.png")
    plot_per_phenotype_metrics(oof, out_dir / "04_per_phenotype_metrics.png")
    plot_calibration(oof,        out_dir / "05_calibration.png")
    plot_feature_importance(imp_df, out_dir / "06_feature_importance.png")
    plot_pool_comparison(oof,    out_dir / "07_pool_comparison.png")

    print("\n=== Writing report ===")
    write_report(oof, fold_summary, overall, out_dir / "REPORT.md")
    print("\nDone.")


if __name__ == "__main__":
    main()
