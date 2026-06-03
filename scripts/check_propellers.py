#!/usr/bin/env python3
"""
check_propellers.py — Verify that NLR chains fold into two WD40 beta-propellers.

Standalone script (no local imports) — safe to copy to WCSS as a single file.

This is a quality-control script run on WCSS after AF3 finishes. For each job
it checks every seed×sample CIF and decides by majority vote whether the NLR
chain (A) contains two separate, compact propeller domains.

Strategy:
  1. Parse CIF, extract Cα coordinates of chain A (NLR).
  2. Split at residue 294 (7 repeats × 42 AA = one propeller).
  3. Compute radius of gyration (Rg) for each half — both must be
     within [RG_MIN, RG_MAX] to count as compact globular domains.
  4. Measure distance between centres of mass — must be >= COM_SEPARATION_MIN
     to confirm the two domains are spatially separated.
  5. For CIF selection, use hetc_centrality:
     abs(d(HET-C, P1_COM) - d(HET-C, P2_COM)) — lower means HET-C
     is equidistant from both propellers (sitting in the cleft).
     This metric penalises wrapping, where one propeller moves very close
     to HET-C while the other stays far away (large asymmetry → high score).

Usage:
    python check_propellers.py --output_dir /path/to/af3_outputs
    python check_propellers.py --output_dir /path/to/af3_outputs --copy-best
    python check_propellers.py --output_dir /path/to/af3_outputs --copy-n 3
    python check_propellers.py --cif path/to/file.cif

--copy-best  copies the single best CIF per job as {job}.cif (classic mode).
--copy-n N   copies the top N CIFs per job ranked by hetc_centrality,
             all into a single flat folder as {job}__{seed_sample}.cif.
             The seed is encoded in the filename so collect_selected.py
             can pick the correct JSONs later without re-running the metric.
             Use this when the automatic best pick shows a wrapping artefact.

Author: Kacper Koźmin
"""

from __future__ import annotations

import argparse
import math
import re
import shutil
from pathlib import Path


# ── WD40 / propeller constants ─────────────────────────────────────

REPEAT_LENGTH = 42
REPEATS_PER_PROPELLER = 7
PROPELLER_LENGTH = REPEAT_LENGTH * REPEATS_PER_PROPELLER  # 294 AA

# Propeller geometry thresholds
RG_MIN = 12.0             # Å — minimum radius of gyration for a compact propeller
RG_MAX = 38.0             # Å — maximum radius of gyration
COM_SEPARATION_MIN = 30.0  # Å — min distance between propeller CoMs

# ── Paths ──────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = ROOT / "data" / "af3_outputs"


# ── Argument parsing ───────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Check WD40 propeller count in AF3 CIF outputs")
    p.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
                   help="Directory containing AF3 output job folders")
    p.add_argument("--cif", type=str, default=None,
                   help="Path to a single CIF file (overrides --output_dir)")
    p.add_argument("--split", type=int, default=PROPELLER_LENGTH,
                   help=f"Residue split point for P1/P2 boundary "
                        f"(default: {PROPELLER_LENGTH})")

    copy_group = p.add_mutually_exclusive_group()
    copy_group.add_argument("--copy-best", action="store_true", default=False,
                            help="Copy the single best TWO_PROPELLERS CIF per "
                                 "job to <output_dir>/../two_propellers/")
    copy_group.add_argument("--copy-n", type=int, default=0, metavar="N",
                            help="Copy the top N CIFs per job (ranked by "
                                 "hetc_centrality, TWO_PROPELLERS only) into "
                                 "<output_dir>/../two_propellers/<job>/. "
                                 "Use this to manually review candidates when "
                                 "the automatic best pick shows a wrapping "
                                 "artefact.")
    return p.parse_args()


# ── CIF parsing ────────────────────────────────────────────────────

def parse_ca_coords(cif_path: Path) -> dict[str, list[tuple[int, float, float, float]]]:
    """Extract Cα coordinates per chain from a CIF file.

    Returns {chain_id: [(resnum, x, y, z), ...]} sorted by residue number.
    """
    with open(cif_path) as f:
        lines = f.readlines()

    col_map: dict[str, int] = {}
    col_idx = 0
    in_atom = False
    start_line = len(lines)

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "loop_":
            col_map, col_idx, in_atom = {}, 0, False
            continue
        if stripped.startswith("_atom_site."):
            col_map[stripped] = col_idx
            col_idx += 1
            in_atom = True
            continue
        if in_atom and col_map and not stripped.startswith("_") and stripped and stripped != "#":
            start_line = i
            break

    if not col_map:
        return {}

    grp = col_map.get("_atom_site.group_PDB")
    atm = col_map.get("_atom_site.label_atom_id")
    chn = col_map.get("_atom_site.label_asym_id") or col_map.get("_atom_site.auth_asym_id")
    seq = col_map.get("_atom_site.label_seq_id") or col_map.get("_atom_site.auth_seq_id")
    xi  = col_map.get("_atom_site.Cartn_x")
    yi  = col_map.get("_atom_site.Cartn_y")
    zi  = col_map.get("_atom_site.Cartn_z")

    if None in (grp, atm, chn, seq, xi, yi, zi):
        return {}

    max_col = max(grp, atm, chn, seq, xi, yi, zi)
    result: dict[str, list[tuple[int, float, float, float]]] = {}

    for i in range(start_line, len(lines)):
        line = lines[i].strip()
        if not line or line.startswith("#") or line.startswith("_"):
            break
        parts = line.split()
        if len(parts) <= max_col:
            continue
        try:
            if parts[grp] == "ATOM" and parts[atm] == "CA" and parts[seq] != ".":
                chain = parts[chn]
                resnum = int(parts[seq])
                x, y, z = float(parts[xi]), float(parts[yi]), float(parts[zi])
                result.setdefault(chain, []).append((resnum, x, y, z))
        except (ValueError, IndexError):
            pass

    return {ch: sorted(coords, key=lambda c: c[0]) for ch, coords in result.items()}


# ── Geometry ───────────────────────────────────────────────────────

def centre_of_mass(coords: list[tuple[int, float, float, float]]
                   ) -> tuple[float, float, float]:
    n = len(coords)
    if n == 0:
        return (0.0, 0.0, 0.0)
    return (sum(c[1] for c in coords) / n,
            sum(c[2] for c in coords) / n,
            sum(c[3] for c in coords) / n)


def radius_of_gyration(coords: list[tuple[int, float, float, float]]) -> float:
    if not coords:
        return 0.0
    cx, cy, cz = centre_of_mass(coords)
    return math.sqrt(
        sum((c[1]-cx)**2 + (c[2]-cy)**2 + (c[3]-cz)**2 for c in coords)
        / len(coords)
    )


def distance(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


# ── Model enumeration ──────────────────────────────────────────────

def find_all_cifs(job_dir: Path) -> list[tuple[int, int, Path]]:
    """Return all (seed, sample, cif_path) tuples in a job directory."""
    results = []
    for subdir in job_dir.iterdir():
        if not subdir.is_dir():
            continue
        m = re.match(r"seed-(\d+)_sample-(\d+)$", subdir.name)
        if not m:
            continue
        cifs = list(subdir.glob("*.cif"))
        if cifs:
            results.append((int(m.group(1)), int(m.group(2)), cifs[0]))
    return results


# ── Per-CIF check ──────────────────────────────────────────────────

def check_cif(cif_path: Path, split_at: int = PROPELLER_LENGTH) -> dict:
    """Analyse one CIF file for two-propeller geometry.

    Returns a dict with:
      status:           "OK" or "ERROR"
      verdict:          "TWO_PROPELLERS", "ONE_PROPELLER (...)", "UNCLEAR (...)"
      rg_p1, rg_p2:    radius of gyration for each half (Å)
      com_dist:         distance between propeller CoMs (Å)
      d_hetc_p1/p2:    distance from HET-C CoM to each propeller CoM (Å)
      hetc_centrality: abs(d1 - d2) — lower = more equidistant = less wrapping
    """
    by_chain = parse_ca_coords(cif_path)
    coords = by_chain.get("A", [])
    if not coords:
        return {"status": "ERROR", "reason": "no Cα atoms in chain A"}

    hetc_coords = by_chain.get("B", [])

    p1 = [r for r in coords if r[0] <= split_at]
    p2 = [r for r in coords if r[0] > split_at]

    if not p1 or not p2:
        return {"status": "ERROR",
                "reason": f"split at {split_at} gave {len(p1)}/{len(p2)} residues"}

    rg1 = radius_of_gyration(p1)
    rg2 = radius_of_gyration(p2)
    com_p1 = centre_of_mass(p1)
    com_p2 = centre_of_mass(p2)
    com_dist = distance(com_p1, com_p2)

    p1_ok = RG_MIN <= rg1 <= RG_MAX
    p2_ok = RG_MIN <= rg2 <= RG_MAX
    separated = com_dist >= COM_SEPARATION_MIN

    if p1_ok and p2_ok and separated:
        verdict = "TWO_PROPELLERS"
    elif not separated:
        verdict = "UNCLEAR (propellers not separated)"
    elif not p1_ok:
        verdict = "ONE_PROPELLER (P1 not compact)"
    else:
        verdict = "ONE_PROPELLER (P2 not compact)"

    d_hetc_p1 = None
    d_hetc_p2 = None
    centrality_value = None
    if hetc_coords:
        com_hetc = centre_of_mass(hetc_coords)
        d_hetc_p1 = round(distance(com_hetc, com_p1), 1)
        d_hetc_p2 = round(distance(com_hetc, com_p2), 1)
        # abs(d1 - d2): lower = HET-C equidistant from both propellers.
        # Penalises wrapping (one propeller very close, other far → large diff).
        centrality_value = round(abs(d_hetc_p1 - d_hetc_p2), 1)

    return {
        "status": "OK",
        "verdict": verdict,
        "rg_p1": round(rg1, 1),
        "rg_p2": round(rg2, 1),
        "com_dist": round(com_dist, 1),
        "d_hetc_p1": d_hetc_p1,
        "d_hetc_p2": d_hetc_p2,
        "hetc_centrality": centrality_value,
    }


# ── Job scanning ───────────────────────────────────────────────────

def check_job(job_dir: Path, split_at: int) -> dict | None:
    """Check all seeds for a job. Returns aggregated result with majority vote."""
    entries = find_all_cifs(job_dir)
    if not entries:
        return None

    n_two = 0
    best_centrality = float("inf")
    best_cif_path: Path | None = None
    two_prop_candidates: list[tuple[float, Path]] = []

    for _seed, _sample, cif_path in entries:
        r = check_cif(cif_path, split_at)
        if r["status"] == "ERROR":
            continue

        centrality = (r["hetc_centrality"]
                      if r["hetc_centrality"] is not None
                      else float("inf"))

        if r["verdict"] == "TWO_PROPELLERS":
            n_two += 1
            two_prop_candidates.append((centrality, cif_path))

        if centrality < best_centrality:
            best_centrality = centrality
            best_cif_path = cif_path

    two_prop_candidates.sort(key=lambda x: x[0])

    n_checked = len(entries)
    frac = n_two / n_checked if n_checked > 0 else 0.0

    if frac >= 0.5:
        verdict = "TWO_PROPELLERS"
    elif frac > 0.0:
        verdict = "UNCERTAIN"
    else:
        verdict = "ONE/UNCLEAR"

    best_two_cif = two_prop_candidates[0][1] if two_prop_candidates else None

    return {
        "n_checked": n_checked,
        "n_two": n_two,
        "frac": frac,
        "best_centrality": (round(best_centrality, 1)
                            if best_centrality < float("inf") else None),
        "best_cif": best_cif_path,
        "best_two_cif": best_two_cif,
        "two_prop_candidates": two_prop_candidates,
        "verdict": verdict,
    }


# ── Main ───────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    if args.cif:
        r = check_cif(Path(args.cif), args.split)
        for k, v in r.items():
            print(f"  {k}: {v}")
        return

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"ERROR: {output_dir} does not exist")
        return

    job_dirs = sorted(d for d in output_dir.iterdir() if d.is_dir())
    if not job_dirs:
        print(f"No job directories found in {output_dir}")
        return

    print(f"Checking {len(job_dirs)} jobs — majority vote across all seeds")
    print(f"{'Job':<45} {'2-prop':>7} {'Frac':>6} {'HetC-ctr':>9}  Verdict")
    print("-" * 85)

    all_results = []

    for job_dir in job_dirs:
        r = check_job(job_dir, args.split)
        if r is None:
            print(f"{job_dir.name:<45}  no CIF found")
            continue

        frac_str = f"{r['n_two']}/{r['n_checked']}"
        best_cif_display = r["best_two_cif"] or r["best_cif"]
        ctr = r["best_centrality"]
        ctr_str = f"{ctr:.1f}Å" if ctr is not None else "  n/a"
        print(
            f"{job_dir.name:<45} {frac_str:>7} {r['frac']:>5.0%} {ctr_str:>9}"
            f"  {r['verdict']}"
        )
        if best_cif_display:
            print(f"  → {best_cif_display}")

        all_results.append({"job": job_dir.name, **r})

    if not all_results:
        return

    two = sum(1 for r in all_results if r["verdict"] == "TWO_PROPELLERS")
    unc = sum(1 for r in all_results if r["verdict"] == "UNCERTAIN")
    total = len(all_results)
    print(f"\nSummary: {two}/{total} TWO_PROPELLERS  |  {unc}/{total} UNCERTAIN")

    dest_dir = output_dir.parent / "two_propellers"

    # ── --copy-best: single best CIF per job ──
    if args.copy_best:
        dest_dir.mkdir(exist_ok=True)
        copied = 0
        for r in all_results:
            if r["n_two"] == 0:
                continue
            src = r["best_two_cif"] or r["best_cif"]
            if src is None or not src.exists():
                continue
            dst = dest_dir / f"{r['job']}.cif"
            shutil.copy2(src, dst)
            copied += 1
        print(f"\nCopied {copied} CIFs → {dest_dir}/")
        print(f"\nscp kackoz9157@ui.wcss.pl:{dest_dir}/*.cif .")

    # ── --copy-n N: top N candidates per job for manual review ─────
    elif args.copy_n > 0:
        n = args.copy_n
        dest_dir.mkdir(exist_ok=True)
        copied_jobs = 0
        copied_files = 0

        for r in all_results:
            candidates = r["two_prop_candidates"]
            if not candidates:
                continue

            # Copy top N as flat files named  {job}__{seed_sample}.cif
            # The double underscore separator lets collect_selected.py parse
            # the seed_sample back out of the filename automatically.
            top = candidates[:n]
            for _score, src in top:
                seed_sample = src.parent.name          # e.g. seed-2_sample-0
                dst_name = f"{r['job']}__{seed_sample}.cif"
                shutil.copy2(src, dest_dir / dst_name)
                copied_files += 1

            copied_jobs += 1
            seeds_str = ", ".join(src.parent.name for _, src in top)
            print(f"  {r['job']}: {len(top)} candidate(s) [{seeds_str}]")

        print(f"\nCopied {copied_files} CIFs across {copied_jobs} jobs → {dest_dir}/")
        print(f"\nscp kackoz9157@ui.wcss.pl:{dest_dir}/*.cif .")
        print(f"\nReview the CIFs. The filename encodes the seed, e.g.:")
        print(f"  job__seed-1_sample-0.cif   ← best-ranked (lowest asymmetry)")
        print(f"  job__seed-3_sample-0.cif   ← second-ranked")
        print(f"Copy the good one to wcss/selected/<batch>/ keeping the full filename.")
        print(f"collect_selected.py will read the seed from the filename.")


if __name__ == "__main__":
    main()
