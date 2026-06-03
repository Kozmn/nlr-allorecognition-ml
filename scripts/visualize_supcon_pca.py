"""
visualize_supcon_pca.py — PCA on SupCon projections (256-D → 2-D).

Analogous to visualize_embeddings_pca.py but on the projected
embeddings from contrastive_supcon.py (saved in supcon_projections_*.npz).

Outputs:
  data/models/eval/supcon_pca_label_{pool}.png      ← Plot A: color: interaction label
  data/models/eval/supcon_pca_phenotype_{pool}.png  ← Plot B: color: NLR phenotype

Compare these plots against `pca_pair_features_label_*.png` to check
whether SupCon separates classes better than raw embeddings.

Usage:
  cd thesis/
  python scripts/visualize_supcon_pca.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def pca_2d(X: np.ndarray):
    X = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    components = Vt[:2]
    X_proj = X @ components.T
    var_total = (S ** 2).sum()
    var_explained = (S[:2] ** 2) / var_total
    return X_proj, var_explained


def make_plot_label(X_proj, labels, var_exp, out_path, pool):
    fig, ax = plt.subplots(figsize=(8, 6.5))
    neg_mask = labels == 0; pos_mask = labels == 1
    ax.scatter(X_proj[neg_mask, 0], X_proj[neg_mask, 1],
               c="#9aa0a6", s=28, alpha=0.55,
               label=f"incompatible (n={neg_mask.sum()})",
               edgecolor="white", linewidth=0.4)
    ax.scatter(X_proj[pos_mask, 0], X_proj[pos_mask, 1],
               c="#d62728", s=44, alpha=0.85,
               label=f"compatible (n={pos_mask.sum()})",
               edgecolor="black", linewidth=0.5)
    ax.set_xlabel(f"PC1 ({var_exp[0]*100:.1f}% variance)", fontsize=11)
    ax.set_ylabel(f"PC2 ({var_exp[1]*100:.1f}% variance)", fontsize=11)
    ax.set_title(f"SupCon projection 256-D → PCA 2-D ({pool}-pool)\n"
                 f"color: interaction label",
                 fontsize=12, pad=12)
    ax.legend(loc="best", framealpha=0.9, fontsize=10)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def make_plot_phenotype(X_proj, phenotypes, var_exp, out_path, pool):
    pheno_colors = {
        "D1": "#1f77b4", "D2": "#ff7f0e", "d3": "#aec7e8",
        "E1": "#d62728", "E2": "#9467bd", "E3": "#2ca02c", "e4": "#c5b0d5",
    }
    fig, ax = plt.subplots(figsize=(8, 6.5))
    for ph in ["d3", "e4", "D2", "D1", "E2", "E1", "E3"]:
        mask = phenotypes == ph
        if not mask.any():
            continue
        ax.scatter(X_proj[mask, 0], X_proj[mask, 1],
                   c=pheno_colors.get(ph, "gray"), s=34, alpha=0.78,
                   label=f"{ph} (n={mask.sum()})",
                   edgecolor="white", linewidth=0.4)
    ax.set_xlabel(f"PC1 ({var_exp[0]*100:.1f}% variance)", fontsize=11)
    ax.set_ylabel(f"PC2 ({var_exp[1]*100:.1f}% variance)", fontsize=11)
    ax.set_title(f"SupCon projection 256-D → PCA 2-D ({pool}-pool)\n"
                 f"color: NLR phenotype",
                 fontsize=12, pad=12)
    ax.legend(loc="best", framealpha=0.9, fontsize=9, ncol=2)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(ROOT / "data" / "models" / "eval"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)

    for pool in ("mean", "max"):
        path = out_dir / f"supcon_projections_{pool}.npz"
        if not path.exists():
            print(f"[--]  {pool}: missing {path}, skipping")
            continue
        print(f"\n=== SupCon PCA: {pool}-pool ===")
        npz = np.load(path)
        Z = npz["Z"]
        labels = npz["labels"]
        phenos = npz["phenotypes"]
        npz.close()

        X_proj, var_exp = pca_2d(Z)
        print(f"  PC1 variance: {var_exp[0]*100:.1f}%")
        print(f"  PC2 variance: {var_exp[1]*100:.1f}%")
        print(f"  Cumulative PC1+PC2: {(var_exp[0]+var_exp[1])*100:.1f}%")

        plot_label = out_dir / f"supcon_pca_label_{pool}.png"
        make_plot_label(X_proj, labels, var_exp, plot_label, pool)
        print(f"  → {plot_label.relative_to(ROOT)}")

        plot_pheno = out_dir / f"supcon_pca_phenotype_{pool}.png"
        make_plot_phenotype(X_proj, phenos, var_exp, plot_pheno, pool)
        print(f"  → {plot_pheno.relative_to(ROOT)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
