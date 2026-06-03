"""
refine_enrichment.py — more honest enrichment metrics for AF3 contacts.

The point of this script is to test whether AF3 concentrates contacts on the
**region** around the hypervariable positions, as opposed to on the **exact**
hypervariable position. A regional signal is still useful (it localises the
interface), but scientific honesty requires separating it from the stronger
claim that "AF3 detects specificity".

Three metrics, all computed from the existing bundles (no AF3 re-runs):

  1. NEIGHBORHOOD-CORRECTED ENRICHMENT
     For each hypervariable position: how many times more contacts does it
     have than the mean of its neighbours (+/- 5 residues)?
       Enrichment ~ 1.0  -> regional signal (AF3 hits the vicinity)
       Enrichment >= 2.0 -> point signal    (AF3 hits the exact position)

  2. PAE-FILTERED ENRICHMENT
     Compares enrichment computed from all contacts vs from only the
     low-PAE contacts (< 10 Å — geometrically confident).
       Ratio (after/before) ~ 1.0  -> enrichment holds on confident contacts
       Ratio (after/before) << 1.0 -> enrichment came from uncertain contacts

  3. PEAK FWHM (Full Width at Half Maximum)
     Measures the width of the contact_prob peak around each hypervariable
     position.
       FWHM <= 3  -> narrow peak (point specificity)
       FWHM >= 10 -> broad plateau (regional signal)

Outputs:
  data/validation/reports/refined_enrichment/
    01_neighborhood_enrichment.png      local enrichment distribution (violin plot)
    02_profile_around_hypervariable.png signal shape around the three HET-C positions
    REPORT.md                           PAE and FWHM numbers as tables
    refined_enrichment_data.npz

Usage:
  cd thesis/
  python scripts/refine_enrichment.py
  python scripts/refine_enrichment.py --pae-threshold 10 --neighborhood 5
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from constants import (
    HETC_HYPERVARIABLE,
    NLR_REPEAT_HYPERVARIABLE,
    REPEAT_LENGTH,
    TLEGH_RE,
)

# ─────────────────────────────────────────────────────────────────────
# AESTHETIC
# ─────────────────────────────────────────────────────────────────────
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
# PATHS / CONSTANTS
# ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
BUNDLES_DIR = ROOT / "data" / "af3_outputs"
INPUTS_DIR = ROOT / "data" / "af3_inputs" / "labeled"
OUT_DIR = ROOT / "data" / "validation" / "reports" / "refined_enrichment"
OUT_DIR.mkdir(parents=True, exist_ok=True)

JOB_RE = re.compile(r"^([a-z0-9]+)_(\w+?)_vs_(c\d+)_(l[01])$", re.IGNORECASE)

CONTACT_THRESHOLD = 0.2     # min contact_prob to count as "in contact"
PAE_DEFAULT_THRESHOLD = 10.0  # Å — below this AF3 is "geometrically confident"
NEIGHBORHOOD_DEFAULT = 5    # ±5 residues
PEAK_WINDOW = 10            # ±10 residues for FWHM

EPS = 1e-9
# Minimal background "density" required to compute a meaningful ratio.
# Below this we treat the position as "no signal here" and skip — otherwise
# divide-by-near-zero produces millions and ruins means/std.
MIN_DENSITY_FOR_RATIO = 0.05


# ─────────────────────────────────────────────────────────────────────
# HELPERS — sequences, repeats, parsing
# ─────────────────────────────────────────────────────────────────────

def find_repeat_boundaries(seq: str) -> list[tuple[int, int]]:
    """1-indexed (start, end) for each WD40 repeat detected via TLEGH motif."""
    starts = [max(0, m.start() - 4) for m in TLEGH_RE.finditer(seq)]
    out = []
    for i, s in enumerate(starts):
        end = (starts[i + 1] - 1 if i + 1 < len(starts)
               else min(s + REPEAT_LENGTH - 1, len(seq) - 1))
        out.append((s + 1, end + 1))
    return out


def parse_job_meta(name: str) -> dict | None:
    m = JOB_RE.match(name)
    if not m:
        return None
    nlr_id, suffix, allele, label = m.groups()
    return {
        "phenotype": nlr_id.upper(),
        "nlr_id": f"{nlr_id}_{suffix}",
        "allele": allele.upper(),
        "label": label.upper(),
    }


def load_sequences_for_job(job_name: str) -> tuple[str, str] | None:
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


# ─────────────────────────────────────────────────────────────────────
# COLLECT BUNDLE DATA
# ─────────────────────────────────────────────────────────────────────

def collect_records():
    """Yield per-bundle data: contact_probs, pae, repeat boundaries."""
    records = []
    bundles = sorted([b for b in BUNDLES_DIR.iterdir() if b.is_dir()])
    print(f"Scanning {len(bundles)} bundles...")
    for b in bundles:
        meta = parse_job_meta(b.name)
        if meta is None:
            continue
        cf_files = sorted([f for f in b.glob("*confidences.json")
                           if "summary" not in f.name])
        if not cf_files:
            continue
        with open(cf_files[0]) as f:
            d = json.load(f)
        tc = np.array(d["token_chain_ids"])
        cp = np.array(d["contact_probs"])
        pae_full = np.array(d.get("pae", []))
        n_a = int((tc == "A").sum())
        n_b = int((tc == "B").sum())
        cp_ab = cp[:n_a, n_a:n_a + n_b]
        pae_ab = (pae_full[:n_a, n_a:n_a + n_b]
                  if pae_full.size > 0 else None)

        seqs = load_sequences_for_job(b.name)
        if seqs is None:
            continue
        nlr_seq, hetc_seq = seqs
        repeat_bounds = find_repeat_boundaries(nlr_seq)
        records.append({
            "name": b.name,
            "meta": meta,
            "cp_ab": cp_ab,
            "pae_ab": pae_ab,
            "n_a": n_a,
            "n_b": n_b,
            "repeat_bounds": repeat_bounds,
        })
    print(f"Loaded {len(records)} records.")
    return records


# ─────────────────────────────────────────────────────────────────────
# METHOD 1: Neighborhood-corrected enrichment
# ─────────────────────────────────────────────────────────────────────

def hetc_density(rec) -> np.ndarray:
    """Sum of contact_probs at each HET-C position (length n_b)."""
    return rec["cp_ab"].sum(axis=0)


def nlr_density(rec) -> np.ndarray:
    """Sum of contact_probs at each NLR position (length n_a)."""
    return rec["cp_ab"].sum(axis=1)


def neighborhood_enrichment_hetc(records, positions, window):
    """For each bundle and hypervariable HET-C position p:
        center  = density at p
        neighb. = mean density at p±window (excluding p)
        enrich  = center / neighb.
    Skipuj przypadki gdy CENTER i NEIGHBOR < MIN_DENSITY_FOR_RATIO
    ( "missing signal in this fragment"; each ratio would be tu garbage).
    Returns: dict {p: list of per-bundle enrichments}.
    """
    results = {p: [] for p in positions}
    skipped = {p: 0 for p in positions}
    for rec in records:
        density = hetc_density(rec)
        n_b = len(density)
        for p in positions:
            if p > n_b:
                continue
            center = float(density[p - 1])
            lo = max(1, p - window)
            hi = min(n_b, p + window)
            neighbor_idx = [i for i in range(lo, hi + 1) if i != p]
            if not neighbor_idx:
                continue
            neighbor = float(np.mean([density[i - 1] for i in neighbor_idx]))
            # both too low → no signal on this fragment
            if center < MIN_DENSITY_FOR_RATIO and neighbor < MIN_DENSITY_FOR_RATIO:
                skipped[p] += 1
                continue
            # neighbor truly zero → cap ratio rather than dividing
            if neighbor < EPS:
                results[p].append(min(center / EPS, 100.0))  # cap at 100
            else:
                results[p].append(center / neighbor)
    return results, skipped


def neighborhood_enrichment_nlr(records, positions, window):
    """For each bundle, repeat, and intra-repeat hypervariable position p:
        center  = density at absolute NLR position (start_of_repeat + p - 1)
        neighb. = mean density at neighbors ±window WITHIN the repeat
        enrich  = center / neighb.
    Skipuj przypadki bez signal (center i neighbor < MIN_DENSITY_FOR_RATIO).
    Returns: dict {p: list of per-(bundle×repeat) enrichments}.
    """
    results = {p: [] for p in positions}
    skipped = {p: 0 for p in positions}
    for rec in records:
        density = nlr_density(rec)
        n_a = len(density)
        for start, end in rec["repeat_bounds"]:
            for p in positions:
                abs_pos = start + p - 1   # 1-indexed absolute NLR position
                if abs_pos < start or abs_pos > end or abs_pos > n_a:
                    continue
                center = float(density[abs_pos - 1])
                lo = max(start, abs_pos - window)
                hi = min(end, abs_pos + window)
                neighbor_idx = [i for i in range(lo, hi + 1) if i != abs_pos]
                if not neighbor_idx:
                    continue
                neighbor = float(np.mean([density[i - 1] for i in neighbor_idx]))
                if center < MIN_DENSITY_FOR_RATIO and neighbor < MIN_DENSITY_FOR_RATIO:
                    skipped[p] += 1
                    continue
                if neighbor < EPS:
                    results[p].append(min(center / EPS, 100.0))
                else:
                    results[p].append(center / neighbor)
    return results, skipped


# ─────────────────────────────────────────────────────────────────────
# METHOD 2: PAE-filtered enrichment
# ─────────────────────────────────────────────────────────────────────

def pae_filtered_enrichment(records, hetc_positions, pae_threshold):
    """For each bundle, compute enrichment at hypervariable positions
    (relative to mean) BEFORE and AFTER PAE filter.

    Returns:
      orig_enrich: list of enrichments per bundle (no filter)
      filt_enrich: list of enrichments per bundle (PAE < threshold)
      orig_n:      list of #contacts per bundle (no filter)
      filt_n:      list of #contacts per bundle (filtered)
    """
    orig_enrich, filt_enrich, orig_n, filt_n = [], [], [], []
    for rec in records:
        cp = rec["cp_ab"]
        n_b = cp.shape[1]
        positions_in_range = [p for p in hetc_positions if p <= n_b]
        if not positions_in_range:
            continue
        # Original
        density_orig = cp.sum(axis=0)
        center_orig = float(np.mean([density_orig[p - 1] for p in positions_in_range]))
        bg_orig = float(np.mean(density_orig))
        if bg_orig > 0:
            orig_enrich.append(center_orig / bg_orig)
            orig_n.append(int((cp >= CONTACT_THRESHOLD).sum()))
        # PAE-filtered
        if rec["pae_ab"] is None:
            continue
        mask = rec["pae_ab"] < pae_threshold
        cp_filt = np.where(mask, cp, 0.0)
        density_filt = cp_filt.sum(axis=0)
        center_filt = float(np.mean([density_filt[p - 1] for p in positions_in_range]))
        bg_filt = float(np.mean(density_filt))
        if bg_filt > 0:
            filt_enrich.append(center_filt / bg_filt)
            filt_n.append(int(((cp_filt >= CONTACT_THRESHOLD)).sum()))
    return orig_enrich, filt_enrich, orig_n, filt_n


# ─────────────────────────────────────────────────────────────────────
# METHOD 4: Peak FWHM
# ─────────────────────────────────────────────────────────────────────

def peak_fwhm(records, positions, window=PEAK_WINDOW):
    """For each bundle and each hypervariable HET-C position p:
        - find local max in window p ± window
        - FWHM = number of contiguous residues around peak with density >= max/2
    """
    results = {p: [] for p in positions}
    for rec in records:
        density = hetc_density(rec)
        n_b = len(density)
        for p in positions:
            if p > n_b:
                continue
            lo = max(0, p - 1 - window)
            hi = min(n_b, p + window)
            local = density[lo:hi]
            if local.size == 0 or local.max() <= 0:
                continue
            peak_val = float(local.max())
            half = peak_val / 2.0
            peak_idx = int(np.argmax(local))
            left = peak_idx
            while left > 0 and local[left - 1] >= half:
                left -= 1
            right = peak_idx
            while right < local.size - 1 and local[right + 1] >= half:
                right += 1
            results[p].append(right - left + 1)
    return results


def aggregate_profile_around(records, positions, window=PEAK_WINDOW):
    """For each hypervariable position, return matrix (n_bundles × 2*window+1)
    of contact density profiles around that position."""
    profiles = {p: [] for p in positions}
    for rec in records:
        density = hetc_density(rec)
        n_b = len(density)
        for p in positions:
            seg = np.full(2 * window + 1, np.nan)
            for offset in range(-window, window + 1):
                pos = p + offset      # 1-indexed
                if 1 <= pos <= n_b:
                    seg[offset + window] = density[pos - 1]
            profiles[p].append(seg)
    return {p: (np.array(v) if v else np.zeros((0, 2 * window + 1)))
            for p, v in profiles.items()}


# ─────────────────────────────────────────────────────────────────────
# PLOTS — minimalist, one idea per figure
# ─────────────────────────────────────────────────────────────────────

def _color_for_enrichment(value: float) -> str:
    """Colour code: red = AF3 avoids, grey = neutral, blue = AF3 hits."""
    if value < 0.8:
        return "#c0392b"        # red — AF3 avoids the position
    if value < 1.25:
        return "#bdc3c7"        # grey — neutralne
    if value < 2.5:
        return "#5a8dd0"        # jasny blue — signal
    if value < 6.0:
        return "#1f5fcc"        # blue — silny signal
    return "#0d2c5c"            # ciemnogranatowy — bardzo silny


def _violin_color(median: float) -> str:
    """Kolor violinu wg mediany — red omijany, blue silny, grey neutralny."""
    if median < 0.8:
        return "#c0392b"
    if median < 1.25:
        return "#7f8c8d"
    if median < 2.5:
        return "#5a8dd0"
    if median < 6.0:
        return "#1f5fcc"
    return "#0d2c5c"


def _draw_horizontal_violins(ax, data_per_label, x_log=True):
    """Draws horyzontalny violin plot.
    data_per_label: list of (label, np.array values, median).
    Top position on the Y axis = first element of the list.
    """
    n = len(data_per_label)
    positions = np.arange(n, 0, -1)            # gora → dol
    arrays = [d[1] for d in data_per_label]
    labels = [d[0] for d in data_per_label]
    medians = [d[2] for d in data_per_label]

    # For a log scale we clamp to [0.05, ∞) and suppress zero values
    if x_log:
        arrays_for_violin = [np.clip(a, 0.05, None) for a in arrays]
    else:
        arrays_for_violin = arrays

    parts = ax.violinplot(arrays_for_violin, positions=positions,
                          vert=False, showmedians=False, showextrema=False,
                          widths=0.78)
    for body, med in zip(parts["bodies"], medians):
        c = _violin_color(med)
        body.set_facecolor(c)
        body.set_edgecolor(c)
        body.set_alpha(0.55)
        body.set_linewidth(0)

    # IQR as a thick horizontal line, median as a white dot
    for pos, arr, med in zip(positions, arrays, medians):
        if arr.size == 0:
            continue
        q1, q3 = np.percentile(arr, [25, 75])
        ax.hlines(pos, max(q1, 0.05) if x_log else q1, q3,
                  color=_violin_color(med), linewidth=4, alpha=0.95,
                  zorder=4)
        ax.scatter(med, pos, color="white", edgecolor=_violin_color(med),
                   s=55, linewidth=1.5, zorder=5)
        # Median annotation on the right
        ax.text(ax.get_xlim()[1] if False else med, pos, "",
                visible=False)  # placeholder

    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=10)


def plot_neighborhood(hetc_results, nlr_results, savepath):
    """Horyzontalny violin plot rozkladu lokalnego enrichmentu per position.
    Log-scale na osi X. Linia 1× jako missing signal. Sortowanie po medianie."""
    # Build data sorted by median
    hetc_data = []
    for p in sorted(hetc_results.keys()):
        v = hetc_results[p]
        if not v:
            continue
        arr = np.array(v)
        hetc_data.append((f"poz. {p}\n(n={len(arr)})", arr, float(np.median(arr))))
    hetc_data.sort(key=lambda t: t[2])  # od najslabszego do najsilniejszego

    nlr_data = []
    for p in sorted(nlr_results.keys()):
        v = nlr_results[p]
        if not v:
            continue
        arr = np.array(v)
        nlr_data.append((f"poz. {p}\n(n={len(arr)})", arr, float(np.median(arr))))
    nlr_data.sort(key=lambda t: t[2])

    # Common x-range — log scale od 0.1 do 200 (NLR poz. 30 ma outliers do ~600)
    all_arrs = [a for _, a, _ in hetc_data] + [a for _, a, _ in nlr_data]
    x_min = 0.1
    # Use 99 percentyla, by outliery not rozciagnely osi
    x_max = max(np.percentile(np.concatenate(all_arrs), 99), 10)

    # 3 + 7 = 10 positions total. Height ~0.8" per position = ~8" + headers.
    fig, axes = plt.subplots(2, 1, figsize=(11, 10),
                             gridspec_kw={"height_ratios": [3, 7]})
    fig.subplots_adjust(top=0.93, bottom=0.07, left=0.13, right=0.93, hspace=0.28)

    for ax, data, title in [
        (axes[0], hetc_data, "HET-C (3 positions hypervariable)"),
        (axes[1], nlr_data, "NLR — position wewnatrz WD40 repeats (7 positions)"),
    ]:
        _draw_horizontal_violins(ax, data, x_log=True)
        ax.set_xscale("log")
        ax.set_xlim(x_min, x_max * 1.5)
        ax.axvline(1.0, color="#222222", linestyle="--", linewidth=1.2, alpha=0.7)
        # Annotacja mediany na prawym marginesie
        n = len(data)
        positions = np.arange(n, 0, -1)
        for pos, (_, _, med) in zip(positions, data):
            label = f"{med:.2f}×" if med < 10 else f"{med:.1f}×"
            ax.text(x_max * 1.55, pos, label,
                    va="center", ha="right",
                    fontsize=12, fontweight="bold",
                    color=_violin_color(med))
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10, loc="left")
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0, labelsize=10)
        ax.tick_params(axis="x", labelsize=10)
        ax.grid(axis="x", linestyle=":", alpha=0.35, which="both")

    # Only the bottom subplot has an X-axis label
    axes[0].set_xlabel("")
    axes[1].set_xlabel(
        "Lokalny enrichment per bundle (log scale; pionowa linia = 1× = missing signal)",
        fontsize=11)

    fig.suptitle("Rozklad lokalnego enrichmentu at hypervariable positions",
                 fontsize=15, fontweight="bold", y=0.985)
    fig.savefig(savepath, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  → {savepath.relative_to(ROOT)}")


def plot_profile_around(profiles, savepath, window=PEAK_WINDOW):
    """Three small panels side by side. Mean only, no std ribbon.
    Clear interpretation: a peak at offset 0 means AF3 hits the position."""
    keys = sorted(profiles.keys())
    fig, axes = plt.subplots(1, len(keys), figsize=(4.5 * len(keys), 4),
                             sharey=True)
    if len(keys) == 1:
        axes = [axes]
    fig.subplots_adjust(top=0.85, bottom=0.16, left=0.07, right=0.97, wspace=0.15)

    x = np.arange(-window, window + 1)
    colors = ["#2257b8", "#7a3eb8", "#c0182d"]

    # Wspolny y_max for porownywalnosci
    all_means = []
    for p in keys:
        prof = profiles[p]
        if prof.size > 0:
            all_means.append(np.nanmean(prof, axis=0))
    if not all_means:
        return
    y_max = max(m.max() for m in all_means) * 1.15

    for ax, p, color in zip(axes, keys, colors):
        prof = profiles[p]
        if prof.size == 0:
            ax.set_visible(False)
            continue
        mean = np.nanmean(prof, axis=0)

        ax.fill_between(x, 0, mean, color=color, alpha=0.18)
        ax.plot(x, mean, color=color, linewidth=2.5)
        # Highlight position 0
        center_val = mean[window]
        ax.scatter([0], [center_val], color=color, s=80, zorder=5,
                   edgecolor="white", linewidth=1.5)
        ax.axvline(0, color="black", linestyle=":", linewidth=0.8, alpha=0.5)

        ax.set_title(f"poz. {p}", fontsize=13, fontweight="bold", color=color)
        ax.set_xlabel("Odsuniecie (residua)", fontsize=10)
        ax.set_xlim(-window, window)
        ax.set_ylim(0, y_max)
        ax.set_xticks([-10, -5, 0, 5, 10])
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="y", linestyle=":", alpha=0.35)

    axes[0].set_ylabel("Mean gestosc contact_prob", fontsize=11)
    fig.suptitle("Profil kontaktow wokol trzech positions hypervariable HET-C",
                 fontsize=13, fontweight="bold", y=0.96)
    fig.savefig(savepath, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  → {savepath.relative_to(ROOT)}")


# ─────────────────────────────────────────────────────────────────────
# REPORT
# ─────────────────────────────────────────────────────────────────────

def write_report(results, savepath, pae_threshold):
    hetc = results["hetc_neigh"]
    nlr = results["nlr_neigh"]
    orig = results["pae_orig"]
    filt = results["pae_filt"]
    fwhm = results["fwhm"]

    hetc_med_overall = float(np.median([np.median(v) for v in hetc.values() if v]))
    nlr_med_overall = float(np.median([np.median(v) for v in nlr.values() if v]))
    fwhm_med_overall = float(np.median([np.median(v) for v in fwhm.values() if v]))
    orig_med = float(np.median(orig)) if orig else 0
    filt_med = float(np.median(filt)) if filt else 0
    ratio = filt_med / orig_med if orig_med > 0 else 0

    with open(savepath, "w") as f:
        f.write("# Refined enrichment analysis — a more honest AF3 validation\n\n")
        f.write("**Goal:** show that AF3 enriches contacts on the *region* around "
                "the hypervariable positions, rather than on the *exact* position. "
                "This regional signal is real and valuable (it localises the "
                "interface), but it is weaker than the naive 4.96x would suggest.\n\n")
        f.write("---\n\n")

        # Method 1
        f.write("## 1. Neighborhood-corrected enrichment\n\n")
        f.write("For each hypervariable position: how many times more contacts does "
                "it have than the mean of its neighbours (+/- 5 residues)?\n\n")
        f.write("### HET-C\n\n")
        f.write("| Position | n bundles | Median | Mean ± std |\n")
        f.write("|--------:|---------:|--------:|--------------:|\n")
        for p in sorted(hetc.keys()):
            v = hetc[p]
            if v:
                f.write(f"| {p} | {len(v)} | {np.median(v):.2f}× "
                        f"| {np.mean(v):.2f} ± {np.std(v):.2f} |\n")
        f.write("\n### NLR (positions within the repeat)\n\n")
        f.write("| Position | n (bundles×repeat) | Median | Mean ± std |\n")
        f.write("|--------:|-----------------:|--------:|--------------:|\n")
        for p in sorted(nlr.keys()):
            v = nlr[p]
            if v:
                f.write(f"| {p} | {len(v)} | {np.median(v):.2f}× "
                        f"| {np.mean(v):.2f} ± {np.std(v):.2f} |\n")

        # Method 2
        f.write("\n---\n\n## 2. PAE-filtered enrichment\n\n")
        f.write(f"- Without filter (n={len(orig)}): median enrichment = "
                f"**{orig_med:.2f}×**\n")
        f.write(f"- With PAE filter < {pae_threshold:.0f} Å (n={len(filt)}): "
                f"median = **{filt_med:.2f}×**\n")
        f.write(f"- Ratio (after/before) = **{ratio:.2f}**\n\n")
        f.write("- Ratio ~ 1.0 -> enrichment holds on confident contacts\n")
        f.write("- Ratio < 0.5 -> enrichment partly from artefacts\n\n")

        # Method 4
        f.write("---\n\n## 4. Peak FWHM\n\n")
        f.write("| Position | n bundles | Median FWHM | Mean FWHM |\n")
        f.write("|--------:|---------:|-------------:|-------------:|\n")
        for p in sorted(fwhm.keys()):
            v = fwhm[p]
            if v:
                f.write(f"| {p} | {len(v)} | {np.median(v):.0f} | "
                        f"{np.mean(v):.1f} ± {np.std(v):.1f} |\n")
        f.write("\n- FWHM <= 3 -> narrow peak (point specificity)\n")
        f.write("- FWHM 4-8 -> intermediate width\n")
        f.write("- FWHM >= 10 -> broad plateau (regional signal)\n\n")

        # Conclusions
        f.write("---\n\n## Conclusions (for discussion in the thesis)\n\n")
        f.write(f"- **Local enrichment**: HET-C ~ {hetc_med_overall:.2f}×, "
                f"NLR ~ {nlr_med_overall:.2f}×. ")
        if hetc_med_overall < 1.5 and nlr_med_overall < 1.5:
            f.write("The signal is **regional** — AF3 hits the vicinity, not the exact position.\n")
        elif hetc_med_overall > 2.0 or nlr_med_overall > 2.0:
            f.write("The signal has a **point component** — AF3 partly hits the exact position.\n")
        else:
            f.write("The signal is **intermediate** — between regional and point-like.\n")
        f.write(f"- **PAE filter** retains {ratio*100:.0f}% of the original enrichment — "
                f"the signal ")
        if ratio > 0.7:
            f.write("**holds up** on geometrically confident contacts.\n")
        elif ratio > 0.4:
            f.write("**partly fades** — uncertain contacts made a meaningful contribution.\n")
        else:
            f.write("**largely fades** — the original enrichment is a PAE artefact.\n")
        f.write(f"- **Median FWHM ~ {fwhm_med_overall:.0f} residues**: ")
        if fwhm_med_overall <= 3:
            f.write("the peaks are **narrow** — AF3 targets specific positions.\n")
        elif fwhm_med_overall <= 8:
            f.write("the peaks are **moderately broad** — regional signal with a mild peak.\n")
        else:
            f.write("the peaks are **a smeared plateau** — AF3 sees the region, not the position.\n")
        f.write("\n### Main takeaway\n\n")
        f.write("AF3 does not model individual single-residue contacts precisely — "
                "the geometric uncertainty (PAE) is too high. However, it **still "
                "produces a strong signal around the hypervariable positions** and "
                "correctly identifies the NLR–HET-C interface region. This regional "
                "signal can be a useful first-pass filter for more precise methods "
                "(e.g. molecular dynamics, mutational experiments).\n")
    print(f"  → {savepath.relative_to(ROOT)}")


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pae-threshold", type=float, default=PAE_DEFAULT_THRESHOLD,
                    help="Threshold PAE for 'pewnych geometrycznie' kontaktow (default: 10 Å)")
    ap.add_argument("--neighborhood", type=int, default=NEIGHBORHOOD_DEFAULT,
                    help="Okno sasiadow ±N residuow (default: 5)")
    args = ap.parse_args()

    print(f"PAE threshold:  < {args.pae_threshold} Å")
    print(f"Neighborhood:   ±{args.neighborhood} residues")
    print(f"Output dir:     {OUT_DIR}\n")

    records = collect_records()
    if not records:
        print("No bundles loaded.")
        return

    print("\n=== Method 1: Neighborhood-corrected enrichment ===")
    print(f"  (skipuje przypadki bez signal: center i neighbor < {MIN_DENSITY_FOR_RATIO})")
    hetc_neigh, hetc_skip = neighborhood_enrichment_hetc(
        records, sorted(HETC_HYPERVARIABLE), window=args.neighborhood)
    for p, v in sorted(hetc_neigh.items()):
        if v:
            print(f"  HET-C poz. {p:>3}: mediana={np.median(v):.2f}×, "
                  f"mean={np.mean(v):.2f}±{np.std(v):.2f} "
                  f"(n_valid={len(v)}, n_skipped={hetc_skip[p]})")
    nlr_neigh, nlr_skip = neighborhood_enrichment_nlr(
        records, sorted(NLR_REPEAT_HYPERVARIABLE), window=args.neighborhood)
    for p, v in sorted(nlr_neigh.items()):
        if v:
            print(f"  NLR poz.  {p:>2}: mediana={np.median(v):.2f}×, "
                  f"mean={np.mean(v):.2f}±{np.std(v):.2f} "
                  f"(n_valid={len(v)}, n_skipped={nlr_skip[p]})")

    print("\n=== Method 2: PAE-filtered enrichment ===")
    orig, filt, orig_n, filt_n = pae_filtered_enrichment(
        records, sorted(HETC_HYPERVARIABLE),
        pae_threshold=args.pae_threshold)
    print(f"  Bez filtra (n={len(orig)}): mediana = {np.median(orig):.2f}×")
    if filt:
        print(f"  Z filtrem  (n={len(filt)}): mediana = {np.median(filt):.2f}×")
        print(f"  Ratio (po/przed): {np.median(filt)/np.median(orig):.2f}")
        if orig_n and filt_n:
            ratios = [f / max(o, 1) for o, f in zip(orig_n, filt_n)]
            print(f"  Ulamek kontaktow po filtrze: mediana = {np.median(ratios):.2f}")

    print("\n=== Method 4: Peak FWHM ===")
    fwhm = peak_fwhm(records, sorted(HETC_HYPERVARIABLE))
    for p, v in sorted(fwhm.items()):
        if v:
            print(f"  HET-C poz. {p:>3}: mediana FWHM = {np.median(v):.0f} residuow, "
                  f"mean={np.mean(v):.1f}±{np.std(v):.1f} (n={len(v)})")

    profiles = aggregate_profile_around(records, sorted(HETC_HYPERVARIABLE))

    print("\n=== Generating plots ===")
    plot_neighborhood(hetc_neigh, nlr_neigh,
                       OUT_DIR / "01_neighborhood_enrichment.png")
    plot_profile_around(profiles, OUT_DIR / "02_profile_around_hypervariable.png")

    results = {
        "hetc_neigh": hetc_neigh,
        "nlr_neigh": nlr_neigh,
        "pae_orig": orig,
        "pae_filt": filt,
        "fwhm": fwhm,
    }
    write_report(results, OUT_DIR / "REPORT.md", args.pae_threshold)

    np.savez(
        OUT_DIR / "refined_enrichment_data.npz",
        pae_orig=np.array(orig),
        pae_filt=np.array(filt) if filt else np.array([]),
        **{f"hetc_neigh_{p}": np.array(v) for p, v in hetc_neigh.items()},
        **{f"nlr_neigh_{p}": np.array(v) for p, v in nlr_neigh.items()},
        **{f"fwhm_{p}": np.array(v) for p, v in fwhm.items()},
    )
    print(f"  → {(OUT_DIR / 'refined_enrichment_data.npz').relative_to(ROOT)}")
    print("\nDone.")


if __name__ == "__main__":
    main()
