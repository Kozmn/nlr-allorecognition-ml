"""
compute_sasa.py — Batch-compute per-residue SASA for every accepted AF3 job.

Caches one JSON per job so that downstream scripts (validate_af3_contacts_sasa,
refine_enrichment) can consume surface masks without reparsing CIFs.

For each job folder in af3_outputs/, this picks the same CIF that
validate_af3_contacts.py uses (the single flat-bundle CIF) and runs
Bio.PDB.SASA.ShrakeRupley on the complex. The surface mask is computed
at three thresholds (0.15, 0.20, 0.25) so sensitivity analysis is free.

Input:
  data/af3_outputs/<job>/*.cif

Output:
  data/validation/sasa/<job>.json
  data/validation/sasa/_summary.tsv   (one row per job: chain lengths,
                                       surface counts, hypervariable∩surface)

Usage:
  cd thesis/
  python scripts/compute_sasa.py
  python scripts/compute_sasa.py --output_dir data/af3_outputs --threshold 0.20
  python scripts/compute_sasa.py --jobs d1_chehdap_conf_vs_c1_l0   # single job

Author: Kacper Koźmin
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from constants import (
    HETC_HYPERVARIABLE,
    NLR_REPEAT_HYPERVARIABLE,
)
from cif_utils import find_repeat_boundaries
from sasa_utils import (
    SASA_DEFAULT_THRESHOLDS,
    compute_sasa,
    compute_sasa_apo,
    dump_sasa_cache,
    surface_mask,
)
from validate_af3_contacts import (
    AF3_INPUTS_LABELED,
    AF3_OUTPUTS,
    collect_job_folders,
    find_model_dir,
    load_sequences,
)


# ── Defaults ─────────────────────────────────────────────────────────────────

SASA_CACHE_DIR = Path("data/validation/sasa")


def hypervariable_positions_nlr(nlr_seq: str) -> set[int]:
    """Return all 1-indexed NLR residue positions that are hypervariable.

    "Hypervariable" = intra-repeat position in NLR_REPEAT_HYPERVARIABLE,
    restricted to residues that fall within a detected WD40 repeat.
    """
    out: set[int] = set()
    for start, end in find_repeat_boundaries(nlr_seq):
        for r in range(start, end + 1):
            intra = r - start + 1
            if intra in NLR_REPEAT_HYPERVARIABLE:
                out.add(r)
    return out


def summary_row(job_name: str,
                sasa_data: dict,
                threshold: float,
                nlr_seq: str,
                ) -> dict:
    """Build a one-row summary for _summary.tsv."""
    mask = surface_mask(sasa_data, threshold=threshold)
    surf_a = mask.get("A", set())
    surf_b = mask.get("B", set())

    hyper_a = hypervariable_positions_nlr(nlr_seq)
    hyper_b = set(HETC_HYPERVARIABLE)

    # Length per chain = # entries in per-residue list
    len_a = len(sasa_data.get("A", []))
    len_b = len(sasa_data.get("B", []))

    return {
        "job": job_name,
        "nlr_length": len_a,
        "hetc_length": len_b,
        "threshold": f"{threshold:.2f}",
        "nlr_surface": len(surf_a),
        "nlr_surface_frac": round(len(surf_a) / len_a, 3) if len_a else 0.0,
        "hetc_surface": len(surf_b),
        "hetc_surface_frac": round(len(surf_b) / len_b, 3) if len_b else 0.0,
        "nlr_hyper_total": len(hyper_a),
        "nlr_hyper_on_surface": len(hyper_a & surf_a),
        "hetc_hyper_total": len(hyper_b),
        "hetc_hyper_on_surface": len(hyper_b & surf_b),
        # Baseline ratios: old vs new
        "nlr_baseline_old_pct": (round(len(hyper_a) / len_a * 100, 2)
                                 if len_a else 0.0),
        "nlr_baseline_new_pct": (round(len(hyper_a & surf_a) / len(surf_a) * 100, 2)
                                 if surf_a else 0.0),
        "hetc_baseline_old_pct": (round(len(hyper_b) / len_b * 100, 2)
                                  if len_b else 0.0),
        "hetc_baseline_new_pct": (round(len(hyper_b & surf_b) / len(surf_b) * 100, 2)
                                  if surf_b else 0.0),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Precompute per-residue SASA and surface masks for all "
                    "accepted AF3 jobs.")
    p.add_argument("--output_dir", type=Path, default=AF3_OUTPUTS,
                   help=f"AF3 outputs root (default: {AF3_OUTPUTS})")
    p.add_argument("--cache_dir", type=Path, default=SASA_CACHE_DIR,
                   help=f"SASA cache folder (default: {SASA_CACHE_DIR})")
    p.add_argument("--inputs_dir", type=Path, default=AF3_INPUTS_LABELED,
                   help=f"AF3 inputs (for sequence lookup) "
                        f"(default: {AF3_INPUTS_LABELED})")
    p.add_argument("--threshold", type=float, default=0.20,
                   help="Relative-SASA cutoff for the summary TSV "
                        "(default: 0.20; all three thresholds are still "
                        "written into each JSON)")
    p.add_argument("--jobs", nargs="*", default=None,
                   help="Restrict to specific job folder names (default: all)")
    p.add_argument("--probe_radius", type=float, default=1.4,
                   help="Solvent probe radius in Å (default: 1.4)")
    p.add_argument("--n_points", type=int, default=100,
                   help="Shrake-Rupley points per atom (default: 100)")
    p.add_argument("--force", action="store_true",
                   help="Recompute even if cache JSON already exists")
    p.add_argument("--apo", dest="apo", action="store_true", default=True,
                   help="Additionally compute apo SASA (each chain in "
                        "isolation) and store it under per_residue_apo / "
                        "surface_masks_apo. Default: True.")
    p.add_argument("--no-apo", dest="apo", action="store_false",
                   help="Skip the apo SASA pass (faster, but the cache "
                        "will not support state='apo' downstream).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    folders = collect_job_folders(args.output_dir)
    if args.jobs:
        keep = set(args.jobs)
        folders = [f for f in folders if f.name in keep]
    print(f"Found {len(folders)} job folders to process")
    if not folders:
        print("Nothing to compute.")
        return

    summary_rows: list[dict] = []
    t0 = time.time()
    for i, job_folder in enumerate(folders, 1):
        cache_path = args.cache_dir / f"{job_folder.name}.json"
        if cache_path.exists() and not args.force:
            # Still regenerate summary row from cache to keep TSV consistent
            try:
                with open(cache_path) as fh:
                    cached = json.load(fh)
                sasa_data = cached.get("per_residue", {})
                nlr_seq, _ = load_sequences(job_folder, args.inputs_dir)
                summary_rows.append(
                    summary_row(job_folder.name, sasa_data,
                                args.threshold, nlr_seq)
                )
                print(f"  [{i:3d}/{len(folders)}] {job_folder.name}  (cached)")
                continue
            except Exception as exc:
                print(f"  WARN: cache for {job_folder.name} unreadable "
                      f"({exc}); recomputing")

        best = find_model_dir(job_folder)
        if best is None:
            print(f"  SKIP {job_folder.name}: no CIF + confidences found")
            continue

        cifs = list(best.glob("*.cif"))
        if not cifs:
            print(f"  SKIP {job_folder.name}: no CIF in {best.name}")
            continue
        cif_path = cifs[0]

        try:
            sasa_data = compute_sasa(
                cif_path,
                probe_radius=args.probe_radius,
                n_points=args.n_points,
            )
        except Exception as exc:
            print(f"  FAIL {job_folder.name}: {exc}")
            continue

        sasa_data_apo = None
        if args.apo:
            try:
                sasa_data_apo = compute_sasa_apo(
                    cif_path,
                    probe_radius=args.probe_radius,
                    n_points=args.n_points,
                )
            except Exception as exc:
                print(f"  FAIL apo for {job_folder.name}: {exc}")
                # Fall through with complex-only cache rather than aborting

        dump_sasa_cache(
            sasa_data, cache_path,
            thresholds=SASA_DEFAULT_THRESHOLDS,
            sasa_data_apo=sasa_data_apo,
            extra={
                "cif": str(cif_path.relative_to(args.output_dir.parent.parent)
                           if cif_path.is_absolute() else cif_path),
                "probe_radius": args.probe_radius,
                "n_points": args.n_points,
                "has_apo": sasa_data_apo is not None,
            },
        )

        nlr_seq, _ = load_sequences(job_folder, args.inputs_dir)
        summary_rows.append(
            summary_row(job_folder.name, sasa_data, args.threshold, nlr_seq)
        )

        elapsed = time.time() - t0
        eta = elapsed / i * (len(folders) - i)
        print(f"  [{i:3d}/{len(folders)}] {job_folder.name}  "
              f"(elapsed {elapsed:.0f}s, ETA {eta:.0f}s)")

    # Write summary TSV
    if summary_rows:
        tsv = args.cache_dir / "_summary.tsv"
        with open(tsv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()),
                               delimiter="\t")
            w.writeheader()
            w.writerows(summary_rows)
        print(f"\nWrote {tsv}")

        # Aggregate baseline summary
        import numpy as np
        nlr_old = np.mean([r["nlr_baseline_old_pct"] for r in summary_rows])
        nlr_new = np.mean([r["nlr_baseline_new_pct"] for r in summary_rows])
        hetc_old = np.mean([r["hetc_baseline_old_pct"] for r in summary_rows])
        hetc_new = np.mean([r["hetc_baseline_new_pct"] for r in summary_rows])
        nlr_surf_frac = np.mean([r["nlr_surface_frac"] for r in summary_rows])
        hetc_surf_frac = np.mean([r["hetc_surface_frac"] for r in summary_rows])

        print("\n── BASELINE CHANGE (at rel_SASA ≥ "
              f"{args.threshold:.2f}) ─────────────────")
        print(f"  Chain A surface fraction:   {nlr_surf_frac:.1%}  (n_jobs={len(summary_rows)})")
        print(f"  Chain B surface fraction:   {hetc_surf_frac:.1%}")
        print(f"  NLR baseline:   old {nlr_old:.2f}%   →   new {nlr_new:.2f}%")
        print(f"  HET-C baseline: old {hetc_old:.2f}%   →   new {hetc_new:.2f}%")

    print("\nDone.")


if __name__ == "__main__":
    main()
