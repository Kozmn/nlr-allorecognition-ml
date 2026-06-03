"""
make_contact_heatmaps.py — clean, publication-quality heatmaps built from
the aggregated AF3 contact cache (aggregated_data.npz).

This is a plotting-only step: it reads the pre-aggregated arrays and renders
three focused figures (instead of the five produced by aggregate_heatmaps.py),
each making a single point:

  01_global.png          global contact map (n = 211 bundles)
  02_l0_vs_l1_split.png  compatible (L0) vs incompatible (L1) side by side,
                         on a shared colour scale
  03_hetc_density.png    1-D contact density along HET-C, L0 vs L1

Design choices:
  - continuous colormap (Reds), no discrete bins
  - minimum text, maximum data
  - hypervariable positions marked subtly (thin guides + label above the axis)
  - shared colorbar for the split panel
  - 300 dpi, sans-serif, white background

Requires only numpy + matplotlib. Reads aggregated_data.npz, which is produced
by aggregate_heatmaps.py (211 bundles, L0 = 160, L1 = 51).

Usage:
  cd thesis/
  python scripts/make_contact_heatmaps.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
NPZ = ROOT / "data" / "validation" / "reports" / "heatmaps" / "aggregated_data.npz"
OUT = ROOT / "data" / "validation" / "reports" / "heatmaps"

# Hipervariable positions
NLR_HV = {10, 11, 12, 14, 30, 32, 39}
HETC_HV = {118, 133, 153}

# ── Style ────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 9.5,
    "axes.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 300,
})

# Czysty sekwencyjny colormap white→deep red, ale z subtelnym przejsciem
CMAP = LinearSegmentedColormap.from_list(
    "contact_seq",
    [
        (0.00, "#ffffff"),
        (0.08, "#fff5e6"),
        (0.25, "#fed8a1"),
        (0.50, "#f4995a"),
        (0.75, "#cc4a2a"),
        (1.00, "#67000d"),
    ],
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _annotate_hetc(ax, ymin: float, ymax: float, label_y: float):
    """Cienkie pionowe linie na 118/133/153 z label na gorze."""
    style = dict(color="#1f4d8c", linestyle=(0, (3, 3)),
                 linewidth=0.9, alpha=0.7, zorder=3)
    for pos in (118, 133, 153):
        ax.axvline(pos, **style)
        ax.text(pos, label_y, str(pos),
                ha="center", va="bottom",
                fontsize=8, fontweight="bold", color="#1f4d8c",
                clip_on=False, zorder=10)


def _annotate_nlr(ax, n_rows: int = 42):
    """Subtelne grey pasy na hypervariable rzedach NLR."""
    for pos in sorted(NLR_HV):
        ax.axhspan(pos - 0.5, pos + 0.5,
                   facecolor="#cccccc", edgecolor="none",
                   alpha=0.18, zorder=1)


def _setup_axes(ax, title: str, xlabel: bool = True, ylabel: bool = True):
    ax.set_xticks([1, 50, 100, 150, 200])
    ax.set_yticks([1, 7, 14, 21, 28, 35, 42])
    ax.tick_params(direction="out", length=2.5, pad=2)
    if xlabel:
        ax.set_xlabel("HET-C residue (1–208)")
    if ylabel:
        ax.set_ylabel("NLR position in WD40 repeat (1–42)")
    if title:
        ax.set_title(title, pad=8, loc="left", fontweight="bold")


# ── Plot 1: global heatmapa ────────────────────────────────────────────────

def plot_global(global_mat: np.ndarray, n: int, out_path: Path):
    fig, ax = plt.subplots(figsize=(11, 4.6),
                           gridspec_kw={"left": 0.07, "right": 0.92,
                                        "top": 0.85, "bottom": 0.16})

    vmax = float(np.percentile(global_mat, 99.5))
    vmax = max(vmax, 0.05)

    im = ax.imshow(global_mat,
                   aspect="auto", cmap=CMAP,
                   vmin=0.0, vmax=vmax,
                   extent=[0.5, global_mat.shape[1] + 0.5,
                           global_mat.shape[0] + 0.5, 0.5],
                   interpolation="nearest")

    _annotate_hetc(ax, ymin=0.5, ymax=42.5, label_y=-1.5)
    _annotate_nlr(ax)

    _setup_axes(ax, title=f"Mean AF3 contact probability across {n} pairs")

    # colorbar
    cax = fig.add_axes([0.93, 0.16, 0.014, 0.69])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("contact_prob", fontsize=8.5)
    cbar.ax.tick_params(labelsize=8, length=2)
    cbar.outline.set_linewidth(0.5)

    fig.savefig(out_path)
    plt.close(fig)
    print(f"  → {out_path.relative_to(ROOT)}")


# ── Plot 2: L0 vs L1 split ───────────────────────────────────────────────────

def plot_l0_vs_l1_split(l0: np.ndarray, l1: np.ndarray,
                         n_l0: int, n_l1: int, out_path: Path):
    fig = plt.figure(figsize=(13, 5.0))
    gs = fig.add_gridspec(1, 2, left=0.05, right=0.94,
                          top=0.86, bottom=0.27, wspace=0.10)

    # wspolna scale
    vmax = float(np.percentile(np.concatenate([l0.ravel(), l1.ravel()]), 99.5))
    vmax = max(vmax, 0.05)

    panels = [
        (l0, "L0 — compatible", n_l0),
        (l1, "L1 — incompatible", n_l1),
    ]
    last_im = None
    for i, (mat, title, n) in enumerate(panels):
        ax = fig.add_subplot(gs[0, i])
        im = ax.imshow(mat, aspect="auto", cmap=CMAP,
                       vmin=0.0, vmax=vmax,
                       extent=[0.5, mat.shape[1] + 0.5,
                               mat.shape[0] + 0.5, 0.5],
                       interpolation="nearest")
        last_im = im
        _annotate_hetc(ax, ymin=0.5, ymax=42.5, label_y=-1.5)
        _annotate_nlr(ax)
        _setup_axes(ax,
                    title=f"{title}   (n={n})",
                    xlabel=True,
                    ylabel=(i == 0))
        if i == 1:
            ax.set_yticklabels([])

    # shared colorbar below both panels
    cax = fig.add_axes([0.30, 0.10, 0.40, 0.022])
    cbar = fig.colorbar(last_im, cax=cax, orientation="horizontal")
    cbar.set_label("contact_prob (mean)", fontsize=8.5)
    cbar.ax.tick_params(labelsize=8, length=2)
    cbar.outline.set_linewidth(0.5)

    fig.savefig(out_path)
    plt.close(fig)
    print(f"  → {out_path.relative_to(ROOT)}")


# ── Plot 3: HET-C 1D density ─────────────────────────────────────────────────

def plot_hetc_density(d_l0: np.ndarray, d_l1: np.ndarray,
                       n_l0: int, n_l1: int, out_path: Path):
    fig, ax = plt.subplots(figsize=(13, 3.6),
                           gridspec_kw={"left": 0.06, "right": 0.97,
                                        "top": 0.86, "bottom": 0.18})

    x = np.arange(1, len(d_l0) + 1)
    ax.bar(x, d_l0, color="#3a6cb8", alpha=0.62, width=1.0,
           label=f"L0 — compatible (n={n_l0})", linewidth=0)
    ax.bar(x, d_l1, color="#cc4a2a", alpha=0.62, width=1.0,
           label=f"L1 — incompatible (n={n_l1})", linewidth=0)

    ymax = max(d_l0.max(), d_l1.max())

    # opisz top-3 piki + all hypervariable positions obok
    top3 = set(np.argsort(d_l0 + d_l1)[::-1][:3].tolist())
    label_set = top3 | {p - 1 for p in HETC_HV}  # 0-indexed
    for i in sorted(label_set):
        pos = int(i) + 1
        if max(d_l0[i], d_l1[i]) < ymax * 0.05 and pos not in HETC_HV:
            continue
        is_hv = pos in HETC_HV
        ax.text(pos, max(d_l0[i], d_l1[i]) + ymax * 0.04,
                f"{pos}",
                ha="center", va="bottom",
                fontsize=8.5,
                fontweight="bold" if pos in top3 else "normal",
                color="#1f4d8c" if is_hv else "#222222")

    # cienkie linie na 118/133/153 (oprocz tych already wyroznionych w top3)
    for pos in (118, 133, 153):
        ax.axvline(pos, color="#1f4d8c",
                   linestyle=(0, (2, 3)), linewidth=0.7,
                   alpha=0.55, zorder=1)

    ax.set_xlim(0.5, len(d_l0) + 0.5)
    ax.set_ylim(0, ymax * 1.18)
    ax.set_xlabel("HET-C residue (1–208)")
    ax.set_ylabel("Σ contact_prob / pair")
    ax.set_title("Aggregate contact density along HET-C",
                 pad=8, loc="left", fontweight="bold")
    ax.legend(loc="upper left", frameon=False, fontsize=8.5)
    ax.grid(axis="y", linestyle=":", linewidth=0.4, alpha=0.45)

    fig.savefig(out_path)
    plt.close(fig)
    print(f"  → {out_path.relative_to(ROOT)}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    if not NPZ.exists():
        raise SystemExit(f"Missing aggregate cache: {NPZ}\n"
                         f"Run aggregate_heatmaps.py first.")

    d = np.load(NPZ, allow_pickle=True)
    global_mat = d["global_"]
    l0 = d["l0"]
    l1 = d["l1"]
    den_l0 = d["hetc_density_l0"]
    den_l1 = d["hetc_density_l1"]
    n_total = int(d["n_total"])
    n_l0 = int(d["n_l0"])
    n_l1 = int(d["n_l1"])

    print(f"Loaded aggregate: total={n_total}, L0={n_l0}, L1={n_l1}")
    OUT.mkdir(parents=True, exist_ok=True)

    plot_global(global_mat, n_total, OUT / "01_global.png")
    plot_l0_vs_l1_split(l0, l1, n_l0, n_l1, OUT / "02_l0_vs_l1_split.png")
    plot_hetc_density(den_l0, den_l1, n_l0, n_l1, OUT / "03_hetc_density.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
