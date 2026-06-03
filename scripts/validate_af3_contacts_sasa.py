"""
validate_af3_contacts_sasa.py — SASA-filtered contact enrichment test.

Permutation test for enrichment of interchain contacts on hypervariable
positions, with the random pool restricted to solvent-exposed residues
(surface positions only). Surface masks come from compute_sasa.py; the
``--state`` flag selects apo or complex SASA.

Input:
  data/af3_outputs/<job>/            AF3 model bundle
  data/validation/sasa/<job>.json    precomputed SASA masks

Output:
  data/validation/reports/contact_validation_sasa[_apo].tsv
  data/validation/reports/summary_sasa[_apo].txt

Usage:
  python scripts/compute_sasa.py
  python scripts/validate_af3_contacts_sasa.py --state apo --threshold 0.20
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from constants import (
    CONTACT_PROB_THRESHOLD,
    HETC_HYPERVARIABLE,
    NLR_REPEAT_HYPERVARIABLE,
    REPEAT_LENGTH,
)
from cif_utils import (
    find_repeat_boundaries,
    classify_nlr_residue,
)
from sasa_utils import load_surface_mask

# Re-use the plumbing from the classic validator — only the statistical
# tests need to change. Everything that loads contacts and sequences is
# identical.
from validate_af3_contacts import (
    AF3_INPUTS_LABELED,
    AF3_OUTPUTS,
    VALIDATION_DIR,
    collect_job_folders,
    extract_interchain_contacts,
    find_model_dir,
    load_contacts_data,
    load_sequences,
    load_summary_confidences,
    parse_job_name,
)

SASA_CACHE_DIR = Path("data/validation/sasa")


# ═════════════════════════════════════════════════════════════════════════════
# SURFACE-RESTRICTED PERMUTATION TESTS
# ═════════════════════════════════════════════════════════════════════════════

def permutation_test_nlr_surface(nlr_contact_residues: list[int],
                                 boundaries: list[tuple[int, int]],
                                 surface_a: set[int],
                                 n_perm: int,
                                 rng: np.random.Generator,
                                 ) -> tuple[int, int, float, int, int]:
    """NLR permutation test restricted to surface residues inside repeats.

    Pool = {residues that lie inside some WD40 repeat} ∩ surface_a.
    Null = uniform choice of `n` residues (with replacement) from pool.
    Hypervariable positions inside this pool are the success states.

    Returns:
        (observed_hits, n_contacts_in_pool, p_value,
         pool_size, pool_hyper_count)

    p_value uses Laplace (+1) smoothing to avoid p = 0 pathology:
        (count_ge + 1) / (n_perm + 1)
    """
    pool: list[int] = []
    hyper_in_pool: list[bool] = []
    for start, end in boundaries:
        for r in range(start, end + 1):
            if r not in surface_a:
                continue
            pool.append(r)
            intra = r - start + 1
            hyper_in_pool.append(intra in NLR_REPEAT_HYPERVARIABLE)

    pool_size = len(pool)
    pool_hyper = int(sum(hyper_in_pool))
    if pool_size == 0:
        return (0, 0, 1.0, 0, 0)

    hyper_arr = np.asarray(hyper_in_pool, dtype=bool)

    pool_set = set(pool)
    in_pool = [r for r in nlr_contact_residues if r in pool_set]
    n = len(in_pool)
    if n == 0:
        return (0, 0, 1.0, pool_size, pool_hyper)

    pool_to_hyper = {r: h for r, h in zip(pool, hyper_in_pool)}
    observed = sum(1 for r in in_pool if pool_to_hyper[r])

    sims = rng.choice(pool_size, size=(n_perm, n), replace=True)
    sim_hits = hyper_arr[sims].sum(axis=1)
    p_value = float(((sim_hits >= observed).sum() + 1) / (n_perm + 1))
    return (observed, n, p_value, pool_size, pool_hyper)


def permutation_test_hetc_surface(hetc_contact_residues: list[int],
                                  hetc_length: int,
                                  surface_b: set[int],
                                  n_perm: int,
                                  rng: np.random.Generator,
                                  ) -> tuple[int, int, float, int, int]:
    """HET-C permutation test restricted to surface residues.

    Pool = surface_b ∩ {1..L_B}. Null = uniform choice of `n` residues
    (with replacement) from pool. Hypervariable positions on HET-C that
    are ALSO on the surface are the success states.
    """
    pool = sorted(p for p in surface_b if 1 <= p <= hetc_length)
    pool_size = len(pool)
    pool_hyper = sum(1 for p in pool if p in HETC_HYPERVARIABLE)
    if pool_size == 0:
        return (0, 0, 1.0, 0, 0)

    pool_set = set(pool)
    in_pool = [r for r in hetc_contact_residues if r in pool_set]
    n = len(in_pool)
    observed = sum(1 for r in in_pool if r in HETC_HYPERVARIABLE)
    if n == 0:
        return (0, 0, 1.0, pool_size, pool_hyper)

    pool_arr = np.asarray(pool, dtype=np.int32)
    hyper_mask = np.asarray([p in HETC_HYPERVARIABLE for p in pool], dtype=bool)

    sims_idx = rng.integers(low=0, high=pool_size, size=(n_perm, n))
    sim_hits = hyper_mask[sims_idx].sum(axis=1)
    p_value = float(((sim_hits >= observed).sum() + 1) / (n_perm + 1))
    return (observed, n, p_value, pool_size, pool_hyper)


# ═════════════════════════════════════════════════════════════════════════════
# PER-JOB ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════

def analyze_job_sasa(contacts: list[dict],
                     hetc_length: int,
                     boundaries: list[tuple[int, int]],
                     surface_a: set[int],
                     surface_b: set[int],
                     n_perm: int,
                     rng: np.random.Generator,
                     ) -> dict:
    """Compute the SASA-filtered enrichment statistics for a single job."""
    total = len(contacts)
    if total == 0:
        return {
            "total_contacts": 0,
            "nlr_hyper_contacts": 0, "nlr_hyper_pct": 0.0,
            "nlr_contacts_in_pool": 0, "nlr_pool_size": 0,
            "nlr_pool_hyper": 0, "nlr_baseline_pct": 0.0,
            "nlr_enrichment_x": 0.0, "nlr_hyper_pvalue": 1.0,
            "hetc_hyper_contacts": 0, "hetc_hyper_pct": 0.0,
            "hetc_contacts_in_pool": 0, "hetc_pool_size": 0,
            "hetc_pool_hyper": 0, "hetc_baseline_pct": 0.0,
            "hetc_enrichment_x": 0.0, "hetc_hyper_pvalue": 1.0,
        }

    nlr_residues = [c["nlr_res"] for c in contacts]
    hetc_residues = [c["hetc_res"] for c in contacts]

    # ── NLR ────────────────────────────────────────────────────────────
    nlr_obs, nlr_in_pool, nlr_p, nlr_pool, nlr_pool_hyper = \
        permutation_test_nlr_surface(
            nlr_residues, boundaries, surface_a,
            n_perm=n_perm, rng=rng)
    nlr_baseline = (nlr_pool_hyper / nlr_pool * 100) if nlr_pool else 0.0
    nlr_obs_pct = (nlr_obs / nlr_in_pool * 100) if nlr_in_pool else 0.0
    nlr_enrich = (nlr_obs_pct / nlr_baseline) if nlr_baseline > 0 else 0.0

    # ── HET-C ──────────────────────────────────────────────────────────
    hetc_obs, hetc_in_pool, hetc_p, hetc_pool, hetc_pool_hyper = \
        permutation_test_hetc_surface(
            hetc_residues, hetc_length, surface_b,
            n_perm=n_perm, rng=rng)
    hetc_baseline = (hetc_pool_hyper / hetc_pool * 100) if hetc_pool else 0.0
    hetc_obs_pct = (hetc_obs / hetc_in_pool * 100) if hetc_in_pool else 0.0
    hetc_enrich = (hetc_obs_pct / hetc_baseline) if hetc_baseline > 0 else 0.0

    return {
        "total_contacts": total,
        # NLR
        "nlr_hyper_contacts": nlr_obs,
        "nlr_hyper_pct": round(nlr_obs_pct, 2),
        "nlr_contacts_in_pool": nlr_in_pool,
        "nlr_pool_size": nlr_pool,
        "nlr_pool_hyper": nlr_pool_hyper,
        "nlr_baseline_pct": round(nlr_baseline, 2),
        "nlr_enrichment_x": round(nlr_enrich, 2),
        "nlr_hyper_pvalue": round(nlr_p, 5),
        # HET-C
        "hetc_hyper_contacts": hetc_obs,
        "hetc_hyper_pct": round(hetc_obs_pct, 2),
        "hetc_contacts_in_pool": hetc_in_pool,
        "hetc_pool_size": hetc_pool,
        "hetc_pool_hyper": hetc_pool_hyper,
        "hetc_baseline_pct": round(hetc_baseline, 2),
        "hetc_enrichment_x": round(hetc_enrich, 2),
        "hetc_hyper_pvalue": round(hetc_p, 5),
    }


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SASA-filtered enrichment test: hypervariable vs. "
                    "surface positions, not vs. whole sequence.")
    p.add_argument("--output_dir", type=Path, default=AF3_OUTPUTS,
                   help=f"AF3 outputs root (default: {AF3_OUTPUTS})")
    p.add_argument("--sasa_dir", type=Path, default=SASA_CACHE_DIR,
                   help=f"Precomputed SASA cache "
                        f"(default: {SASA_CACHE_DIR})")
    p.add_argument("--report_dir", type=Path, default=VALIDATION_DIR,
                   help=f"Report output folder (default: {VALIDATION_DIR})")
    p.add_argument("--threshold", type=float, default=0.20,
                   help="Relative-SASA cutoff to call a residue "
                        "'surface-exposed' (default: 0.20)")
    p.add_argument("--state", choices=["apo", "complex"], default="apo",
                   help="Which SASA state to use for the surface filter. "
                        "'apo' (default) uses SASA of each chain in "
                        "isolation — methodologically correct for "
                        "interchain contact tests because it does not "
                        "exclude interfacial residues from the candidate "
                        "pool. 'complex' uses SASA in the bound state "
                        "(legacy, kept for comparison only).")
    p.add_argument("--contact_threshold", type=float,
                   default=CONTACT_PROB_THRESHOLD,
                   help=f"Contact probability cutoff "
                        f"(default: {CONTACT_PROB_THRESHOLD})")
    p.add_argument("--n_permutations", type=int, default=10000,
                   help="Number of permutations (default: 10000)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed (default: 42)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    folders = collect_job_folders(args.output_dir)
    print(f"Found {len(folders)} job folders in {args.output_dir}")
    if not folders:
        print("Nothing to validate.")
        return

    if not args.sasa_dir.exists():
        raise SystemExit(
            f"ERROR: SASA cache {args.sasa_dir} missing. "
            f"Run `python scripts/compute_sasa.py` first.")

    rows: list[dict] = []
    skipped_no_sasa: list[str] = []
    for job_folder in folders:
        job_info = parse_job_name(job_folder.name)

        best = find_model_dir(job_folder)
        if best is None:
            print(f"  SKIP {job_folder.name}: no CIF + confidences found")
            continue

        data = load_contacts_data(best)
        if data is None:
            print(f"  SKIP {job_folder.name}: no contacts JSON")
            continue

        sasa_path = args.sasa_dir / f"{job_folder.name}.json"
        if not sasa_path.exists():
            skipped_no_sasa.append(job_folder.name)
            continue
        try:
            surface = load_surface_mask(sasa_path, threshold=args.threshold,
                                        state=args.state)
        except KeyError as e:
            print(f"  SKIP {job_folder.name}: {e}")
            skipped_no_sasa.append(job_folder.name)
            continue
        surface_a = surface.get("A", set())
        surface_b = surface.get("B", set())

        conf = load_summary_confidences(best)
        nlr_seq, _hetc_seq = load_sequences(job_folder)
        boundaries = find_repeat_boundaries(nlr_seq)

        contacts = extract_interchain_contacts(
            data, threshold=args.contact_threshold)
        stats = analyze_job_sasa(
            contacts, data["chain_b_len"], boundaries,
            surface_a, surface_b,
            n_perm=args.n_permutations, rng=rng)

        row = {
            "job": job_folder.name,
            "model": best.name,
            "phenotype": job_info["phenotype"],
            "strain": job_info["strain"],
            "confidence": job_info["confidence"],
            "hetc_allele": job_info["hetc_allele"],
            "label": job_info["label"],
            "iptm": conf.get("iptm", ""),
            "ptm": conf.get("ptm", ""),
            "ranking_score": conf.get("ranking_score", ""),
            "nlr_length": data["chain_a_len"],
            "hetc_length": data["chain_b_len"],
            "nlr_surface": len(surface_a),
            "hetc_surface": len(surface_b),
            "sasa_threshold": args.threshold,
            "sasa_state": args.state,
            **stats,
        }
        rows.append(row)

        flag = ""
        if stats["total_contacts"] == 0:
            flag = " NO_CONTACTS"
        else:
            if stats["nlr_hyper_pvalue"] < 0.05:
                flag += " NLR_SIG"
            if stats["hetc_hyper_pvalue"] < 0.05:
                flag += " HETC_SIG"

        print(f"  {job_folder.name}: {stats['total_contacts']} contacts | "
              f"NLR {stats['nlr_hyper_pct']:.1f}% vs base "
              f"{stats['nlr_baseline_pct']:.1f}% "
              f"(x{stats['nlr_enrichment_x']:.2f}, p={stats['nlr_hyper_pvalue']:.3f}) | "
              f"HET-C {stats['hetc_hyper_pct']:.1f}% vs "
              f"{stats['hetc_baseline_pct']:.1f}% "
              f"(x{stats['hetc_enrichment_x']:.2f}, p={stats['hetc_hyper_pvalue']:.3f})"
              f"{flag}")

    if skipped_no_sasa:
        print(f"\nNOTE: {len(skipped_no_sasa)} jobs have no SASA cache; "
              f"run compute_sasa.py to include them. First few: "
              f"{skipped_no_sasa[:3]}")

    if not rows:
        print("No jobs produced output.")
        return

    # ── TSV ─────────────────────────────────────────────────────────────
    suffix = "" if args.state == "complex" else f"_{args.state}"
    tsv = args.report_dir / f"contact_validation_sasa{suffix}.tsv"
    with open(tsv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {tsv}")

    # ── Summary ─────────────────────────────────────────────────────────
    with_c = [r for r in rows if r["total_contacts"] > 0]
    sig_nlr = [r for r in with_c if r["nlr_hyper_pvalue"] < 0.05]
    sig_hetc = [r for r in with_c if r["hetc_hyper_pvalue"] < 0.05]

    summary_path = args.report_dir / f"summary_sasa{suffix}.txt"
    with open(summary_path, "w") as f:
        f.write(f"AF3 Contact Validation Summary "
                f"(SASA-filtered baseline, state={args.state})\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"SASA threshold:          rel_SASA >= {args.threshold:.2f}\n")
        f.write(f"SASA state:              {args.state}  "
                f"({'each chain in isolation' if args.state == 'apo' else 'in bound complex'})\n")
        f.write(f"Contact threshold:       contact_prob >= "
                f"{args.contact_threshold}\n")
        f.write(f"Permutations per test:   {args.n_permutations}\n")
        f.write(f"Significance threshold:  p < 0.05\n\n")
        f.write(f"Jobs analyzed:           {len(rows)}\n")
        f.write(f"Jobs with contacts:      {len(with_c)}\n")
        f.write(f"Jobs without contacts:   {len(rows) - len(with_c)}\n\n")

        f.write(f"NLR   hypervariable enrichment p<0.05: "
                f"{len(sig_nlr)}/{len(with_c)} "
                f"({len(sig_nlr) / max(len(with_c), 1) * 100:.0f}%)\n")
        f.write(f"HET-C hypervariable enrichment p<0.05: "
                f"{len(sig_hetc)}/{len(with_c)} "
                f"({len(sig_hetc) / max(len(with_c), 1) * 100:.0f}%)\n\n")

        if with_c:
            # Weighted aggregate baseline: average of per-job baselines.
            avg_nlr_obs = float(np.mean([r["nlr_hyper_pct"] for r in with_c]))
            avg_nlr_base = float(np.mean([r["nlr_baseline_pct"] for r in with_c]))
            avg_nlr_enrich = (avg_nlr_obs / avg_nlr_base
                              if avg_nlr_base > 0 else 0.0)
            avg_hetc_obs = float(np.mean([r["hetc_hyper_pct"] for r in with_c]))
            avg_hetc_base = float(np.mean([r["hetc_baseline_pct"] for r in with_c]))
            avg_hetc_enrich = (avg_hetc_obs / avg_hetc_base
                               if avg_hetc_base > 0 else 0.0)

            f.write("Observed vs baseline (per-job mean, %):\n")
            f.write(f"  NLR:   observed {avg_nlr_obs:.2f}   "
                    f"surface-baseline {avg_nlr_base:.2f}   "
                    f"→ enrichment {avg_nlr_enrich:.2f}×\n")
            f.write(f"  HET-C: observed {avg_hetc_obs:.2f}   "
                    f"surface-baseline {avg_hetc_base:.2f}   "
                    f"→ enrichment {avg_hetc_enrich:.2f}×\n\n")

            f.write("Note on observed%:\n")
            f.write("  'observed' here = (hyper contacts ∩ surface pool) /\n"
                    "                    (all contacts ∩ surface pool) × 100.\n"
                    "  This is DIFFERENT from the unfiltered observed% reported by\n"
                    "  validate_af3_contacts.py (which uses (hyper contacts) /\n"
                    "  (all contacts), so the denominator includes contacts that\n"
                    "  fall on buried residues). Do NOT divide the surface-filtered\n"
                    "  observed by the classic uniform baseline — the two\n"
                    "  numerator/denominator pairs come from different pools and\n"
                    "  the ratio is methodologically meaningless.\n\n"
                    "  For the genuine classic baseline (uniform over whole\n"
                    "  sequence), refer to data/validation/reports/summary.txt\n"
                    "  produced by validate_af3_contacts.py.\n\n")

            f.write("Interpretation note:\n")
            f.write("  The surface-filtered (apo-SASA) baseline is the honest\n"
                    "  null for interchain contacts: residues that are buried\n"
                    "  even when the chain is isolated cannot physically form\n"
                    "  an interface. The classic uniform baseline includes\n"
                    "  such core residues and is therefore a useful but\n"
                    "  permissive reference.\n")

    print(f"Wrote {summary_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
