"""
visualize_embeddings_pca.py — PCA on pair features (4608-D) → 2-D visualisation.

Goal: check whether RAW embeddings already separate classes (compatible vs not)
without any supervised transformation. This is the baseline for
contrastive learning (covered later).

Produces two plots from the same projection:
  Plot A: color = interaction label
  Plot B: color = NLR phenotype

PCA is implemented manually with NumPy SVD (no sklearn).

Outputs:
  data/models/eval/pca_pair_features_label.png
  data/models/eval/pca_pair_features_phenotype.png
  data/models/eval/pca_explained_variance.csv

Usage:
  cd thesis/
  python scripts/visualize_embeddings_pca.py
  python scripts/visualize_embeddings_pca.py --features data/models/features_max.npz
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


def load_features(path: Path) -> dict:
    npz = np.load(path, allow_pickle=True)
    pack = {k: npz[k] for k in npz.files}
    npz.close()
    return pack


def pca_2d(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """PCA via SVD. Returns (X_proj 2-D, explained_variance_ratio[2])."""
    # Centering
    X = X - X.mean(axis=0, keepdims=True)
    # SVD (we only need the first two components; full SVD is OK for n=275)
    
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    # Principal components: first two right singular vectors
    components = Vt[:2]  # (2, d)
    X_proj = X @ components.T  # (N, 2)
    var_total = (S ** 2).sum()
    var_explained = (S[:2] ** 2) / var_total
    # All variance ratios (saved to CSV)
    all_ratios = (S ** 2) / var_total
    return X_proj, var_explained, all_ratios


def make_plot_label(X_proj: np.ndarray, labels: np.ndarray, var_exp: np.ndarray,
                    out_path: Path, pool: str):
    """Plot A: color = interaction label (compatible / incompatible)."""
    fig, ax = plt.subplots(figsize=(8, 6.5))
    # Negatives first (in the back), positives on top
    neg_mask = labels == 0
    pos_mask = labels == 1
    ax.scatter(X_proj[neg_mask, 0], X_proj[neg_mask, 1],
               c="#9aa0a6", s=28, alpha=0.55, label=f"incompatible (n={neg_mask.sum()})",
               edgecolor="white", linewidth=0.4)
    ax.scatter(X_proj[pos_mask, 0], X_proj[pos_mask, 1],
               c="#d62728", s=44, alpha=0.85, label=f"compatible (n={pos_mask.sum()})",
               edgecolor="black", linewidth=0.5)
    ax.set_xlabel(f"PC1 ({var_exp[0]*100:.1f}% variance)", fontsize=11)
    ax.set_ylabel(f"PC2 ({var_exp[1]*100:.1f}% variance)", fontsize=11)
    ax.set_title(f"PCA pair features ({pool}-pool, 4608-D → 2-D)\n"
                 f"color: interaction label",
                 fontsize=12, pad=12)
    ax.legend(loc="best", framealpha=0.9, fontsize=10)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def make_plot_phenotype(X_proj: np.ndarray, phenotypes: np.ndarray,
                        var_exp: np.ndarray, out_path: Path, pool: str):
    """Plot B: color = NLR phenotype."""
    # Fixed color map for phenotypes
    pheno_colors = {
        "D1": "#1f77b4",  # blue
        "D2": "#ff7f0e",  # orange
        "d3": "#aec7e8",  # light blue
        "E1": "#d62728",  # red
        "E2": "#9467bd",  # purple
        "E3": "#2ca02c",  # green
        "e4": "#c5b0d5",  # light purple
    }
    fig, ax = plt.subplots(figsize=(8, 6.5))
    for ph in ["d3", "e4", "D2", "D1", "E2", "E1", "E3"]:  # order: largest groups drawn first (so they end up underneath)
        mask = phenotypes == ph
        if not mask.any():
            continue
        ax.scatter(X_proj[mask, 0], X_proj[mask, 1],
                   c=pheno_colors.get(ph, "gray"), s=34, alpha=0.78,
                   label=f"{ph} (n={mask.sum()})",
                   edgecolor="white", linewidth=0.4)
    ax.set_xlabel(f"PC1 ({var_exp[0]*100:.1f}% variance)", fontsize=11)
    ax.set_ylabel(f"PC2 ({var_exp[1]*100:.1f}% variance)", fontsize=11)
    ax.set_title(f"PCA pair features ({pool}-pool, 4608-D → 2-D)\n"
                 f"color: NLR phenotype",
                 fontsize=12, pad=12)
    ax.legend(loc="best", framealpha=0.9, fontsize=9, ncol=2)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-mean", default=str(ROOT / "data" / "models" / "features_mean.npz"))
    ap.add_argument("--features-max",  default=str(ROOT / "data" / "models" / "features_max.npz"))
    ap.add_argument("--out-dir", default=str(ROOT / "data" / "models" / "eval"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Iterate over both pools (mean and max)
    for pool, feat_path in [("mean", args.features_mean), ("max", args.features_max)]:
        path = Path(feat_path)
        if not path.exists():
            print(f"[--]  {pool}: missing {path}, skipping")
            continue
        pack = load_features(path)
        X = pack["X"].astype(np.float32)
        y = pack["y"].astype(int)
        phenos = pack["phenotype"].astype(str)

        print(f"\n=== {pool}-pool: PCA on (n={len(X)}, d={X.shape[1]}) ===")
        X_proj, var_exp, all_ratios = pca_2d(X)
        print(f"  PC1 variance: {var_exp[0]*100:.1f}%")
        print(f"  PC2 variance: {var_exp[1]*100:.1f}%")
        print(f"  Cumulative PC1+PC2: {(var_exp[0]+var_exp[1])*100:.1f}%")

        # Save explained variance for top-10 PCs
        var_df = pd.DataFrame({
            "PC": np.arange(1, 11),
            "explained_variance_ratio": all_ratios[:10],
            "cumulative": np.cumsum(all_ratios[:10]),
        })
        var_path = out_dir / f"pca_explained_variance_{pool}.csv"
        var_df.to_csv(var_path, index=False)

        # Plot A: label
        plot_path = out_dir / f"pca_pair_features_label_{pool}.png"
        make_plot_label(X_proj, y, var_exp, plot_path, pool)
        print(f"  → {plot_path.relative_to(ROOT)}")

        # Plot B: phenotype
        plot_path_p = out_dir / f"pca_pair_features_phenotype_{pool}.png"
        make_plot_phenotype(X_proj, phenos, var_exp, plot_path_p, pool)
        print(f"  → {plot_path_p.relative_to(ROOT)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
