"""
aggregate_heatmaps.py — aggregate AF3 inter-chain contact probabilities
across all bundles into heatmaps and 1-D density plots.

This is the upstream aggregation step for the AF3 contact-validation track.
It reads the per-bundle contact_probs from every accepted AF3 prediction and
collapses them into a few summary arrays that show *where* the inter-chain
contacts (NLR <-> HET-C) accumulate over the full set of ~233 bundles. The
aggregated arrays are cached to disk so the figures can be re-plotted without
re-reading every bundle. Several downstream plotting scripts (make_contact_
heatmaps.py, replot_*.py, density_per_phenotype.py, etc.) consume this cache.

Outputs (written to data/validation/reports/heatmaps/):
  01_global_heatmap.png   mean contact_prob over ALL bundles.
                          Y axis: position within the WD40 repeat (1-42);
                          X axis: HET-C position (1-208).
  02_l0_vs_l1_diff.png    difference of means: incompatible (L1) - compatible (L0).
  03_by_hetc_allele.png   grid of heatmaps, one per HET-C allele C1..C11.
  04_by_nlr_phenotype.png grid of heatmaps, one per NLR phenotype E1..d3.
  05_hetc_density_1d.png  bar plot per HET-C position, stratified by L0/L1.
  aggregated_data.npz     the aggregated arrays, for downstream analysis and
                          for re-plotting without re-running the aggregation.

Usage:
  python scripts/aggregate_heatmaps.py                # full aggregation + plots
  python scripts/aggregate_heatmaps.py --replot-only  # re-plot from the .npz only
  python scripts/aggregate_heatmaps.py --threshold 0.2

Plot styling:
  - Discrete colour scale, 6 bins: <0.05 / 0.05-0.2 / 0.2-0.3 / 0.3-0.5 /
    0.5-0.7 / >0.7
  - Dashed vertical guides at HET-C positions 118 (blue) and 153 (red)
  - NLR hypervariable positions {10,11,12,14,30,32,39} marked as grey bands
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle, Patch

from constants import (
    HETC_HYPERVARIABLE,
    NLR_REPEAT_HYPERVARIABLE,
    REPEAT_LENGTH,
    TLEGH_RE,
)

# ─────────────────────────────────────────────────────────────────────
# AESTHETIC — discrete contact scale (matches the reference plots)
# ─────────────────────────────────────────────────────────────────────

CONTACT_BOUNDS = [0.0, 0.05, 0.20, 0.30, 0.50, 0.70, 1.001]
CONTACT_COLORS = [
    "#f7f7f7",   # <0.05 — near-white (no contact)
    "#cfe0f3",   # 0.05-0.20 — very light blue (noise)
    "#3a6cb8",   # 0.20-0.30 — blue (detection threshold)
    "#f1a93a",   # 0.30-0.50 — orange (strong contact)
    "#d63838",   # 0.50-0.70 — red (very strong)
    "#7a0e1a",   # >0.70 — maroon (confident contact)
]
CONTACT_LABELS = [
    "< 0.05",
    "0.05–0.20",
    "0.20–0.30",
    "0.30–0.50",
    "0.50–0.70",
    "> 0.70",
]
CONTACT_CMAP = ListedColormap(CONTACT_COLORS, name="contact_disc")
CONTACT_NORM = BoundaryNorm(CONTACT_BOUNDS, CONTACT_CMAP.N)

# matplotlib defaults — czysta typografia
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

# ─────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
BUNDLES_DIR = ROOT / "data" / "af3_outputs"
INPUTS_DIR = ROOT / "data" / "af3_inputs" / "labeled"
OUT_DIR = ROOT / "data" / "validation" / "reports" / "heatmaps"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────

JOB_RE = re.compile(r"^([a-z0-9]+)_(\w+?)_vs_(c\d+)_(l[01])$", re.IGNORECASE)


def find_repeat_boundaries(seq: str) -> list[tuple[int, int]]:
    """1-indexed (start, end) for each WD40 repeat detected via TLEGH."""
    starts = [max(0, m.start() - 4) for m in TLEGH_RE.finditer(seq)]
    out = []
    for i, s in enumerate(starts):
        end = (starts[i + 1] - 1 if i + 1 < len(starts)
               else min(s + REPEAT_LENGTH - 1, len(seq) - 1))
        out.append((s + 1, end + 1))
    return out


def load_sequences_for_job(job_name: str) -> tuple[str, str] | None:
    """Load NLR and HET-C sequences from data/af3_inputs/labeled/<Job>.json (case-insensitive)."""
    target = job_name.lower()
    for p in INPUTS_DIR.glob("*.json"):
        if p.stem.lower() == target:
            with open(p) as f:
                d = json.load(f)
            seqs = []
            for s in d["sequences"]:
                if "proteinChain" in s:
                    seqs.append(s["proteinChain"]["sequence"])
                elif "protein" in s:
                    seqs.append(s["protein"]["sequence"])
            if len(seqs) == 2:
                return seqs[0], seqs[1]
    return None


def parse_job_meta(name: str) -> dict | None:
    """Parse 'd1_chehdap_conf_vs_c8_l1' → phenotype/strain/allele/label."""
    m = JOB_RE.match(name)
    if not m:
        return None
    nlr_id, suffix, allele, label = m.groups()
    return {
        "phenotype": nlr_id.upper(),       # 'D1', 'E2', etc.
        "strain_suffix": suffix,            # 'chehdap_conf', etc.
        "nlr_id": f"{nlr_id}_{suffix}",
        "allele": allele.upper(),           # 'C8'
        "label": label.upper(),             # 'L0' / 'L1'
    }


# ─────────────────────────────────────────────────────────────────────
# AGGREGATION
# ─────────────────────────────────────────────────────────────────────

def collect_all_contacts(threshold: float = 0.2):
    """Iterate bundles, return per-bundle records.

    Each record:
      meta:           dict (phenotype, allele, label, nlr_id)
      n_a, n_b:       int (chain lengths)
      cp_ab:          np.array (n_a × n_b) of contact probs
      repeat_bounds:  list[(start,end)] in NLR sequence
    """
    records = []
    bundles = sorted([b for b in BUNDLES_DIR.iterdir() if b.is_dir()])
    print(f"Scanning {len(bundles)} bundles...")
    for b in bundles:
        meta = parse_job_meta(b.name)
        if meta is None:
            print(f"  WARN unparseable: {b.name}")
            continue
        cf_files = sorted([f for f in b.glob("*confidences.json")
                           if "summary" not in f.name])
        if not cf_files:
            continue
        cf = cf_files[0]
        with open(cf) as f:
            d = json.load(f)
        tc = np.array(d["token_chain_ids"])
        cp = np.array(d["contact_probs"])
        n_a = (tc == "A").sum()
        n_b = (tc == "B").sum()
        cp_ab = cp[:n_a, n_a:n_a + n_b]
        # filter out contacts < threshold (set to 0)
        cp_ab_filt = np.where(cp_ab >= threshold, cp_ab, 0.0)

        # NLR sequence for detection of boundaries repeatow
        seqs = load_sequences_for_job(b.name)
        if seqs is None:
            print(f"  WARN no seq for {b.name}")
            continue
        nlr_seq, hetc_seq = seqs
        repeat_bounds = find_repeat_boundaries(nlr_seq)
        records.append({
            "meta": meta,
            "name": b.name,
            "n_a": int(n_a),
            "n_b": int(n_b),
            "cp_ab": cp_ab_filt,
            "cp_ab_raw": cp_ab,
            "repeat_bounds": repeat_bounds,
            "nlr_seq_len": len(nlr_seq),
            "hetc_seq_len": len(hetc_seq),
        })
    print(f"Loaded {len(records)} records.")
    return records


def collapse_to_repeat_relative(records: list[dict],
                                use_filtered: bool = True
                                ) -> dict[str, np.ndarray]:
    """For each bundle, project NLR-axis contacts into repeat-relative
    (1..42) coordinate. HET-C axis stays at 1..208. Aggregate as MEAN.

    Returns dict of:
      'global'           — (42, 208) mean across all bundles
      'l0', 'l1'         — (42, 208) means by label
      'by_allele'        — dict[allele → (42,208)]
      'by_phenotype'     — dict[phenotype → (42,208)]
      'hetc_density_l0'  — (208,) sum of contacts per HET-C residue (L0)
      'hetc_density_l1'  — (208,) sum of contacts per HET-C residue (L1)
      'n_total', 'n_l0', 'n_l1'
    """
    HETC_LEN = 208
    REPEAT_RELATIVE_LEN = 42

    # accumulators
    sum_total = np.zeros((REPEAT_RELATIVE_LEN, HETC_LEN), dtype=np.float64)
    cnt_total = 0
    sum_l0 = np.zeros_like(sum_total); cnt_l0 = 0
    sum_l1 = np.zeros_like(sum_total); cnt_l1 = 0
    sum_by_allele: dict[str, np.ndarray] = {}
    cnt_by_allele: dict[str, int] = {}
    sum_by_pheno: dict[str, np.ndarray] = {}
    cnt_by_pheno: dict[str, int] = {}
    hetc_density_l0 = np.zeros(HETC_LEN, dtype=np.float64)
    hetc_density_l1 = np.zeros(HETC_LEN, dtype=np.float64)

    for rec in records:
        cp = rec["cp_ab"] if use_filtered else rec["cp_ab_raw"]
        n_a, n_b = rec["n_a"], rec["n_b"]
        bounds = rec["repeat_bounds"]
        meta = rec["meta"]

        # HET-C may be slightly different length per allele; align by truncation/pad
        if n_b != HETC_LEN:
            # pad or trim
            if n_b < HETC_LEN:
                cp_pad = np.zeros((n_a, HETC_LEN), dtype=cp.dtype)
                cp_pad[:, :n_b] = cp
                cp = cp_pad
            else:
                cp = cp[:, :HETC_LEN]

        # for each NLR position with at least one contact, project to repeat-relative
        local = np.zeros((REPEAT_RELATIVE_LEN, HETC_LEN), dtype=np.float64)
        # check whether each NLR residue is inside a repeat?
        for nlr_pos in range(n_a):
            res_1idx = nlr_pos + 1
            for start, end in bounds:
                if start <= res_1idx <= end:
                    intra = res_1idx - start  # 0..41
                    if 0 <= intra < REPEAT_RELATIVE_LEN:
                        local[intra, :] += cp[nlr_pos, :]
                    break

        # average PER REPEAT (so that long NLRs do not dominate short ones)
        # number of repeats = len(bounds)
        n_repeats = max(1, len(bounds))
        local /= n_repeats

        sum_total += local
        cnt_total += 1
        if meta["label"] == "L0":
            sum_l0 += local; cnt_l0 += 1
            hetc_density_l0[:] += cp.sum(axis=0)[:HETC_LEN]
        else:
            sum_l1 += local; cnt_l1 += 1
            hetc_density_l1[:] += cp.sum(axis=0)[:HETC_LEN]

        a = meta["allele"]
        sum_by_allele.setdefault(a, np.zeros_like(sum_total))
        sum_by_allele[a] += local
        cnt_by_allele[a] = cnt_by_allele.get(a, 0) + 1

        p = meta["phenotype"]
        sum_by_pheno.setdefault(p, np.zeros_like(sum_total))
        sum_by_pheno[p] += local
        cnt_by_pheno[p] = cnt_by_pheno.get(p, 0) + 1

    out = {
        "global": sum_total / max(1, cnt_total),
        "l0": sum_l0 / max(1, cnt_l0),
        "l1": sum_l1 / max(1, cnt_l1),
        "by_allele": {a: s / cnt_by_allele[a] for a, s in sum_by_allele.items()},
        "by_phenotype": {p: s / cnt_by_pheno[p] for p, s in sum_by_pheno.items()},
        "hetc_density_l0": hetc_density_l0 / max(1, cnt_l0),
        "hetc_density_l1": hetc_density_l1 / max(1, cnt_l1),
        "n_total": cnt_total, "n_l0": cnt_l0, "n_l1": cnt_l1,
        "cnt_by_allele": cnt_by_allele,
        "cnt_by_phenotype": cnt_by_pheno,
    }
    return out


# ─────────────────────────────────────────────────────────────────────
# PLOTS — pretty, discrete-bin aesthetic matching the reference image
# ─────────────────────────────────────────────────────────────────────

def _draw_scale_legend(fig, y: float = 0.965):
    """Draws the legend 'Skala:' u gory figury — 6 kolorowych prostokatow."""
    n = len(CONTACT_COLORS)
    box_w = 0.055
    box_h = 0.018
    gap = 0.005
    label_w = 0.045
    total_w = n * (box_w + label_w + gap) - gap
    x0 = 0.5 - total_w / 2 + 0.02

    # naglowek "Skala:"
    fig.text(x0 - 0.06, y + box_h / 2 - 0.005, "Skala:",
             ha="left", va="center", fontsize=10, fontweight="bold")

    for i, (color, label) in enumerate(zip(CONTACT_COLORS, CONTACT_LABELS)):
        x = x0 + i * (box_w + label_w + gap)
        rect = Rectangle((x, y - box_h / 2), box_w, box_h,
                         transform=fig.transFigure,
                         facecolor=color, edgecolor="#333333", linewidth=0.5)
        fig.patches.append(rect)
        fig.text(x + box_w + 0.003, y, label,
                 ha="left", va="center", fontsize=8.5, color="#222222")


def _annotate_hetc_hypervariable(ax, ymax_text: float | None = None):
    """Pionowe linie przerywane na pozycjach HET-C 118 (blue) i 153 (red)."""
    style = {"linestyle": (0, (5, 4)), "linewidth": 1.0, "alpha": 0.75, "zorder": 3}
    annotations = {118: ("#2257b8", "118"), 133: ("#7a3eb8", "133"), 153: ("#c0182d", "153")}
    for pos, (color, label) in annotations.items():
        ax.axvline(pos, color=color, **style)
        if ymax_text is not None:
            ax.text(pos, ymax_text, label, color=color,
                    ha="center", va="bottom", fontsize=8.5, fontweight="bold")


def _annotate_nlr_hypervariable_rows(ax, n_rows: int = 42, color: str = "#888888"):
    """Light grey bars at hypervariable positions NLR (Y axis: intra-repeat positions)."""
    for pos in sorted(NLR_REPEAT_HYPERVARIABLE):
        ax.axhspan(pos - 0.5, pos + 0.5, color=color, alpha=0.08, zorder=1)
        # small label on the right
        ax.text(ax.get_xlim()[1] + 1, pos, f"{pos}",
                ha="left", va="center", fontsize=7, color="#555555")


def _footer(fig, text: str, y: float = 0.012):
    fig.text(0.5, y, text, ha="center", va="bottom",
             fontsize=8.8, color="#444444", style="italic")


def plot_heatmap_disc(matrix: np.ndarray, title: str, subtitle: str,
                      savepath: Path, footer: str = ""):
    """Aggregate heatmap (42 × 208) z discrete scale i annotations."""
    fig = plt.figure(figsize=(12, 5.6))
    fig.subplots_adjust(left=0.07, right=0.93, top=0.86, bottom=0.13)
    ax = fig.add_subplot(1, 1, 1)

    im = ax.imshow(matrix, aspect="auto",
                   cmap=CONTACT_CMAP, norm=CONTACT_NORM,
                   extent=[0.5, matrix.shape[1] + 0.5,
                           matrix.shape[0] + 0.5, 0.5],
                   interpolation="nearest")

    # adnotacje
    _annotate_hetc_hypervariable(ax, ymax_text=-0.5)
    _annotate_nlr_hypervariable_rows(ax)

    # osie i ticki
    ax.set_xlim(0.5, matrix.shape[1] + 0.5)
    ax.set_ylim(matrix.shape[0] + 0.5, 0.5)
    ax.set_xticks([1, 25, 50, 75, 100, 118, 133, 153, 175, 200])
    ax.set_yticks([1, 7, 14, 21, 28, 35, 42])
    ax.set_xlabel("Pozycja HET-C (1–208)")
    ax.set_ylabel("Position within the WD40 repeat (1–42)")

    # titley
    fig.suptitle(title, fontsize=13, fontweight="bold", x=0.5, y=0.94, ha="center")
    if subtitle:
        fig.text(0.5, 0.905, subtitle, ha="center", va="top",
                 fontsize=10, color="#444444")

    _draw_scale_legend(fig, y=0.975)
    if footer:
        _footer(fig, footer)

    fig.savefig(savepath, dpi=170, facecolor="white")
    plt.close(fig)
    print(f"  → {savepath.relative_to(ROOT)}")


def plot_diff_heatmap(diff: np.ndarray, title: str, subtitle: str,
                      n_l0: int, n_l1: int,
                      savepath: Path):
    """Difference plot L1 − L0, diverging colormap (red = L1 wyzszy)."""
    fig = plt.figure(figsize=(12, 5.6))
    fig.subplots_adjust(left=0.07, right=0.93, top=0.86, bottom=0.13)
    ax = fig.add_subplot(1, 1, 1)

    absmax = max(0.005, np.max(np.abs(diff)))
    norm = TwoSlopeNorm(vmin=-absmax, vcenter=0, vmax=absmax)
    im = ax.imshow(diff, aspect="auto", cmap="RdBu_r", norm=norm,
                   extent=[0.5, diff.shape[1] + 0.5,
                           diff.shape[0] + 0.5, 0.5],
                   interpolation="nearest")

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.015)
    cbar.set_label("Δ contact_prob  (L1 − L0)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    _annotate_hetc_hypervariable(ax, ymax_text=-0.5)

    ax.set_xticks([1, 25, 50, 75, 100, 118, 133, 153, 175, 200])
    ax.set_yticks([1, 7, 14, 21, 28, 35, 42])
    ax.set_xlabel("Pozycja HET-C (1–208)")
    ax.set_ylabel("Position within the WD40 repeat (1–42)")

    fig.suptitle(title, fontsize=13, fontweight="bold", x=0.5, y=0.94, ha="center")
    if subtitle:
        fig.text(0.5, 0.905, subtitle, ha="center", va="top",
                 fontsize=10, color="#444444")

    # legend (small inset below the title)
    legend_handles = [
        Patch(facecolor="#b2182b", edgecolor="#333", label=f"L1 higher (incompatible, n={n_l1})"),
        Patch(facecolor="#f7f7f7", edgecolor="#333", label="no difference"),
        Patch(facecolor="#2166ac", edgecolor="#333", label=f"L0 higher (compatible, n={n_l0})"),
    ]
    ax.legend(handles=legend_handles, loc="upper right",
              frameon=True, framealpha=0.95, fontsize=8.5)

    _footer(fig, "Value = mean(L1) − mean(L0). "
                 "Czerwone obszary: AF3 silniej kontaktuje na parach incompatible.")
    fig.savefig(savepath, dpi=170, facecolor="white")
    plt.close(fig)
    print(f"  → {savepath.relative_to(ROOT)}")


def plot_grid_disc(matrices: dict[str, np.ndarray],
                   counts: dict[str, int],
                   title: str, subtitle: str,
                   savepath: Path,
                   ncols: int = 4):
    """Siatka heatmap z discrete scale wspolna for all panels."""
    keys = sorted(matrices.keys(), key=lambda x: (len(x), x))
    n = len(keys)
    nrows = (n + ncols - 1) // ncols
    fig = plt.figure(figsize=(4.2 * ncols, 2.6 * nrows + 1.4))
    fig.subplots_adjust(left=0.05, right=0.97,
                        top=0.88, bottom=0.07,
                        hspace=0.55, wspace=0.18)

    for idx, key in enumerate(keys):
        ax = fig.add_subplot(nrows, ncols, idx + 1)
        m = matrices[key]
        ax.imshow(m, aspect="auto",
                  cmap=CONTACT_CMAP, norm=CONTACT_NORM,
                  extent=[0.5, m.shape[1] + 0.5,
                          m.shape[0] + 0.5, 0.5],
                  interpolation="nearest")

        # adnotacje positions 118/153
        for pos, color in [(118, "#2257b8"), (133, "#7a3eb8"), (153, "#c0182d")]:
            ax.axvline(pos, color=color, linestyle=(0, (4, 3)),
                       linewidth=0.8, alpha=0.7)

        ax.set_title(f"{key}  (n={counts.get(key, 0)})",
                     fontsize=10, fontweight="bold")
        ax.set_xticks([1, 50, 100, 153, 200])
        ax.set_yticks([1, 14, 28, 42])
        ax.tick_params(labelsize=8)
        if idx % ncols == 0:
            ax.set_ylabel("position w pow.", fontsize=8)
        if idx // ncols == nrows - 1:
            ax.set_xlabel("HET-C", fontsize=8)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.965)
    if subtitle:
        fig.text(0.5, 0.925, subtitle, ha="center", va="top",
                 fontsize=10, color="#444444")
    _draw_scale_legend(fig, y=0.97)
    _footer(fig, "Kazdy panel: mean contact map w within subgroups "
                 "(NLR position within the repeat × HET-C).")
    fig.savefig(savepath, dpi=160, facecolor="white")
    plt.close(fig)
    print(f"  → {savepath.relative_to(ROOT)}")


def plot_hetc_density_1d(density_l0: np.ndarray,
                          density_l1: np.ndarray,
                          n_l0: int, n_l1: int,
                          savepath: Path):
    """1D bar plot: which HET-C position accumulates the most contacts."""
    fig, ax = plt.subplots(figsize=(13, 4.4))
    fig.subplots_adjust(left=0.07, right=0.97, top=0.86, bottom=0.16)

    x = np.arange(1, len(density_l0) + 1)
    ax.bar(x, density_l0, color="#3a6cb8", alpha=0.75, width=1.0,
           label=f"L0 — compatible (n={n_l0})")
    ax.bar(x, density_l1, color="#d63838", alpha=0.55, width=1.0,
           label=f"L1 — incompatible (n={n_l1})")

    ymax = max(density_l0.max(), density_l1.max())
    for pos, color, label in [(118, "#2257b8", "118"),
                               (133, "#7a3eb8", "133"),
                               (153, "#c0182d", "153")]:
        ax.axvline(pos, color=color, linestyle="--", linewidth=1.0, alpha=0.85)
        ax.text(pos, ymax * 1.02, label, ha="center", va="bottom",
                fontsize=9, color=color, fontweight="bold",
                bbox=dict(facecolor="white", edgecolor=color,
                          linewidth=0.6, pad=1.5, boxstyle="round,pad=0.25"))

    ax.set_xlabel("Pozycja HET-C (1–208)")
    ax.set_ylabel("Mean summed contact_prob per bundle")
    ax.set_xlim(0.5, len(density_l0) + 0.5)
    ax.set_ylim(0, ymax * 1.18)
    ax.set_title("HET-C density — where AF3 sees the most kontaktow per position",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", frameon=True, framealpha=0.95, fontsize=9)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.5)
    _footer(fig, "Pozycje 118 (blue) i 153 (red) — Bastiaans 2014: "
                 "key for specificity alleles HET-C.")
    fig.savefig(savepath, dpi=170, facecolor="white")
    plt.close(fig)
    print(f"  → {savepath.relative_to(ROOT)}")


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────

def _load_agg_from_npz(path: Path) -> dict:
    """Load cached data z .npz (if aggregation has already been computed)."""
    d = np.load(path, allow_pickle=True)
    agg = {
        "global": d["global_"],
        "l0": d["l0"],
        "l1": d["l1"],
        "hetc_density_l0": d["hetc_density_l0"],
        "hetc_density_l1": d["hetc_density_l1"],
        "n_total": int(d["n_total"]),
        "n_l0": int(d["n_l0"]),
        "n_l1": int(d["n_l1"]),
    }
    # opcjonalne rozszerzenia
    if "by_allele_keys" in d.files:
        keys = [str(k) for k in d["by_allele_keys"]]
        mats = d["by_allele_mats"]
        cnts = d["by_allele_cnts"]
        agg["by_allele"] = {k: mats[i] for i, k in enumerate(keys)}
        agg["cnt_by_allele"] = {k: int(cnts[i]) for i, k in enumerate(keys)}
    if "by_pheno_keys" in d.files:
        keys = [str(k) for k in d["by_pheno_keys"]]
        mats = d["by_pheno_mats"]
        cnts = d["by_pheno_cnts"]
        agg["by_phenotype"] = {k: mats[i] for i, k in enumerate(keys)}
        agg["cnt_by_phenotype"] = {k: int(cnts[i]) for i, k in enumerate(keys)}
    return agg


def _save_agg_to_npz(agg: dict, diff: np.ndarray, path: Path):
    payload = {
        "global_": agg["global"],
        "l0": agg["l0"], "l1": agg["l1"],
        "diff_l1_minus_l0": diff,
        "hetc_density_l0": agg["hetc_density_l0"],
        "hetc_density_l1": agg["hetc_density_l1"],
        "n_total": agg["n_total"],
        "n_l0": agg["n_l0"],
        "n_l1": agg["n_l1"],
    }
    if "by_allele" in agg:
        keys = sorted(agg["by_allele"].keys())
        payload["by_allele_keys"] = np.array(keys)
        payload["by_allele_mats"] = np.stack([agg["by_allele"][k] for k in keys])
        payload["by_allele_cnts"] = np.array([agg["cnt_by_allele"][k] for k in keys])
    if "by_phenotype" in agg:
        keys = sorted(agg["by_phenotype"].keys())
        payload["by_pheno_keys"] = np.array(keys)
        payload["by_pheno_mats"] = np.stack([agg["by_phenotype"][k] for k in keys])
        payload["by_pheno_cnts"] = np.array([agg["cnt_by_phenotype"][k] for k in keys])
    np.savez(path, **payload)


def render_all_plots(agg: dict, threshold: float):
    n_total = agg["n_total"]; n_l0 = agg["n_l0"]; n_l1 = agg["n_l1"]
    subtitle_global = (f"Mean contact_prob value w {n_total} bundles AF3  ·  "
                       f"Y axis: position w within single WD40 repeats  ·  "
                       f"filtr: contact_prob ≥ {threshold}")
    footer_global = ("Kazdy bundle contributes srednia along swoich 12–14 repeats. "
                     "Jasne kolory = no contact, ciemne = pewny kontakt. "
                     "ALL contacts are shown (after thresholding), not a sample.")

    plot_heatmap_disc(agg["global"],
                      title="Global contact map NLR ↔ HET-C",
                      subtitle=subtitle_global,
                      footer=footer_global,
                      savepath=OUT_DIR / "01_global_heatmap.png")

    diff = agg["l1"] - agg["l0"]
    plot_diff_heatmap(diff,
                      title="Differential contact map:  L1 (incompatible) − L0 (compatible)",
                      subtitle="Ile on average kontaktow more AF3 shows on the pairs incompatible",
                      n_l0=n_l0, n_l1=n_l1,
                      savepath=OUT_DIR / "02_l0_vs_l1_diff.png")

    if "by_allele" in agg:
        plot_grid_disc(agg["by_allele"], agg["cnt_by_allele"],
                       title="Contact maps per allele HET-C (C1–C11)",
                       subtitle="Mean contact_prob value for pairs with a given allelem",
                       savepath=OUT_DIR / "03_by_hetc_allele.png",
                       ncols=4)
    if "by_phenotype" in agg:
        plot_grid_disc(agg["by_phenotype"], agg["cnt_by_phenotype"],
                       title="Contact maps per NLR phenotype (E1–D3)",
                       subtitle="Mean contact_prob value for pairs with a given wariantem NLR",
                       savepath=OUT_DIR / "04_by_nlr_phenotype.png",
                       ncols=4)

    plot_hetc_density_1d(agg["hetc_density_l0"], agg["hetc_density_l1"],
                         n_l0, n_l1,
                         OUT_DIR / "05_hetc_density_1d.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.2,
                    help="contact_prob threshold (default: 0.2)")
    ap.add_argument("--replot-only", action="store_true",
                    help="Skip aggregation — use exists aggregated_data.npz.")
    args = ap.parse_args()

    npz_path = OUT_DIR / "aggregated_data.npz"

    if args.replot_only and npz_path.exists():
        print(f"[replot-only] loading {npz_path.relative_to(ROOT)}")
        agg = _load_agg_from_npz(npz_path)
        print(f"  n_total={agg['n_total']}, L0={agg['n_l0']}, L1={agg['n_l1']}")
        render_all_plots(agg, threshold=args.threshold)
        print("\nDone (replot-only).")
        return

    print(f"Threshold: contact_prob >= {args.threshold}")
    print(f"Bundles dir: {BUNDLES_DIR}")
    print(f"Output dir:  {OUT_DIR}")

    records = collect_all_contacts(threshold=args.threshold)

    # only bundle z any kontaktem >= threshold
    records_with_contacts = [r for r in records if (r["cp_ab"] > 0).any()]
    print(f"Bundles with contacts ≥ {args.threshold}: "
          f"{len(records_with_contacts)}/{len(records)}")

    agg = collapse_to_repeat_relative(records_with_contacts)

    print(f"\nLabels: total={agg['n_total']}, L0={agg['n_l0']}, L1={agg['n_l1']}")
    print(f"Alleles: {sorted(agg['cnt_by_allele'].items())}")
    print(f"Phenotypes: {sorted(agg['cnt_by_phenotype'].items())}")

    print("\nGenerating pretty heatmaps:")
    render_all_plots(agg, threshold=args.threshold)

    diff = agg["l1"] - agg["l0"]
    _save_agg_to_npz(agg, diff, npz_path)
    print(f"  → {npz_path.relative_to(ROOT)}")

    # quick stats
    print("\n=== Top 10 hot positions HET-C (ogolnie) ===")
    overall = agg["hetc_density_l0"] * agg["n_l0"] + agg["hetc_density_l1"] * agg["n_l1"]
    top = np.argsort(overall)[::-1][:10]
    for i in top:
        marker = " ← HYPERVAR" if (i + 1) in HETC_HYPERVARIABLE else ""
        print(f"  pos {i+1:3d}: total density {overall[i]:.2f}{marker}")

    eps = 0.01
    enrichment = (agg["hetc_density_l1"] + eps) / (agg["hetc_density_l0"] + eps)
    print("\n=== Top 10 positions HET-C z najwiekszym wzbogaceniem L1/L0 ===")
    top_e = np.argsort(enrichment)[::-1][:10]
    for i in top_e:
        marker = " ← HYPERVAR" if (i + 1) in HETC_HYPERVARIABLE else ""
        print(f"  pos {i+1:3d}: L1/L0 = {enrichment[i]:.2f}, "
              f"L1={agg['hetc_density_l1'][i]:.2f}, "
              f"L0={agg['hetc_density_l0'][i]:.2f}{marker}")

    print("\nDone.")


if __name__ == "__main__":
    main()
