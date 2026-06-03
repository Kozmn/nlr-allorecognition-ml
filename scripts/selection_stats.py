"""
selection_stats.py — Per-NLR acceptance statistics with batch progression.

After each AF3 batch (original, rerun_1, rerun_2, rerun_3), CIFs are manually
reviewed and sorted under wcss/selected/{original,rerun_1,rerun_2,rerun_3}/
on the local machine. This script tracks which batch rescued each job and
reports per-NLR acceptance rates with a batch-by-batch progression view.

Input:
  data/af3_inputs/labeled/*.json              expected job list (275 jobs)
  wcss/selected/original/<job>.cif            batch 0 — original run
  wcss/selected/rerun_1/<job>[__seed-*].cif   batch 1 — rerun_1
  wcss/selected/rerun_2/<job>[__seed-*].cif   batch 2 — rerun_2
  wcss/selected/rerun_3/<job>[__seed-*].cif   batch 3 — rerun_3

Output (printed + TSV):
  data/validation/reports/selection_stats.tsv

Usage:
  cd thesis/
  python scripts/selection_stats.py
  python scripts/selection_stats.py --selected ../wcss/selected/original ../wcss/selected/rerun_1
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from constants import EXCLUDED_NLR_PREFIXES


# ── Defaults (paths relative to the thesis/ folder) ─────────────────────────

AF3_INPUTS = Path("data/af3_inputs/labeled")
DEFAULT_SELECTED_DIRS = [
    Path("../wcss/selected/original"),
    Path("../wcss/selected/rerun_1"),
    Path("../wcss/selected/rerun_2"),
    Path("../wcss/selected/rerun_3"),
]
REPORT_PATH = Path("data/validation/reports/selection_stats.tsv")

BATCH_LABELS = ["original", "rerun_1", "rerun_2", "rerun_3"]


# ── Job name parsing ────────────────────────────────────────────────────────

def strip_seed_suffix(stem: str) -> str:
    """Remove __seed-N_sample-M suffix if present.

    'd1_chehdap_conf_vs_c2_l1__seed-3_sample-0' → 'd1_chehdap_conf_vs_c2_l1'
    'd1_chehdap_conf_vs_c2_l1'                  → 'd1_chehdap_conf_vs_c2_l1'
    """
    if "__seed-" in stem:
        return stem.split("__seed-", 1)[0]
    if "__" in stem:
        return stem.split("__", 1)[0]
    return stem


def split_canonical(canonical_stem: str) -> tuple[str, str] | None:
    """Split a canonical job name into (nlr_key, hetc_key).

    The canonical stem comes from af3_inputs filenames, which preserve the
    intended casing (e.g. 'D2_CsDfP_conf_vs_C10_L1'). We split on '_vs_'
    and normalize the phenotype prefix (D1/D2/D3/E1-E4) to uppercase.
    Returns None if the format is unrecognized.
    """
    if "_vs_" not in canonical_stem:
        return None
    nlr_raw, hetc_key = canonical_stem.split("_vs_", 1)
    tokens = nlr_raw.split("_", 1)
    tokens[0] = tokens[0].upper()
    nlr_key = "_".join(tokens)
    return (nlr_key, hetc_key)


def phenotype_of(nlr_key: str) -> str:
    """Return just the top-level phenotype label (D1/D2/D3/E1-E4...)."""
    return nlr_key.split("_", 1)[0].upper() if nlr_key else "?"


# ── Folder scanning ─────────────────────────────────────────────────────────

def list_canonical_jobs(folder: Path, suffix: str) -> dict[str, str]:
    """Return a {lowercase_stem: canonical_stem} map from files in folder."""
    if not folder.exists():
        return {}
    out: dict[str, str] = {}
    for p in folder.iterdir():
        if p.is_file() and p.name.lower().endswith(suffix):
            stem = p.stem
            out[stem.lower()] = stem
    return out


def list_accepted_per_batch(selected_dirs: list[Path]) -> list[set[str]]:
    """Return a list of sets, one per batch, of accepted lowercase job stems.

    Filenames may contain __seed-* suffixes which are stripped before matching.
    Each batch only includes jobs that were NEW in that batch (not already
    accepted in an earlier batch).
    """
    batches: list[set[str]] = []
    already_accepted: set[str] = set()
    for selected_dir in selected_dirs:
        batch_jobs: set[str] = set()
        if selected_dir.exists():
            for p in selected_dir.iterdir():
                if p.is_file() and p.name.lower().endswith(".cif"):
                    job_stem = strip_seed_suffix(p.stem).lower()
                    if job_stem not in already_accepted:
                        batch_jobs.add(job_stem)
        already_accepted |= batch_jobs
        batches.append(batch_jobs)
    return batches


# ── Stats aggregation ───────────────────────────────────────────────────────

def is_excluded(lowercase_stem: str) -> bool:
    return any(lowercase_stem.startswith(p) for p in EXCLUDED_NLR_PREFIXES)


def aggregate(expected: dict[str, str],
              batches: list[set[str]],
              ) -> tuple[dict, dict, int]:
    """Build per-NLR and per-phenotype summaries with batch breakdown.

    Returns (by_nlr, by_phen, excluded_count).
    """
    n_batches = len(batches)
    all_accepted = set()
    for b in batches:
        all_accepted |= b

    def make_row():
        return {
            "total": 0,
            "accepted": 0,
            "still_bad": 0,
            **{f"batch_{i}": 0 for i in range(n_batches)},
            "hetc_accepted": [],
            "hetc_still_bad": [],
        }

    by_nlr: dict[str, dict] = defaultdict(make_row)
    by_phen: dict[str, dict] = defaultdict(make_row)

    excluded = 0
    for lc_stem, canonical in expected.items():
        if is_excluded(lc_stem):
            excluded += 1
            continue
        parsed = split_canonical(canonical)
        if parsed is None:
            continue
        nlr_key, hetc_key = parsed
        phen = phenotype_of(nlr_key)

        by_nlr[nlr_key]["total"] += 1
        by_phen[phen]["total"] += 1

        accepted_in_batch = None
        for i, batch_set in enumerate(batches):
            if lc_stem in batch_set:
                accepted_in_batch = i
                break

        if accepted_in_batch is not None:
            by_nlr[nlr_key]["accepted"] += 1
            by_nlr[nlr_key][f"batch_{accepted_in_batch}"] += 1
            by_nlr[nlr_key]["hetc_accepted"].append(hetc_key)
            by_phen[phen]["accepted"] += 1
            by_phen[phen][f"batch_{accepted_in_batch}"] += 1
        else:
            by_nlr[nlr_key]["still_bad"] += 1
            by_nlr[nlr_key]["hetc_still_bad"].append(hetc_key)
            by_phen[phen]["still_bad"] += 1

    return by_nlr, by_phen, excluded


# ── Output ──────────────────────────────────────────────────────────────────

def print_table(by_nlr: dict, by_phen: dict,
                excluded_count: int, batch_labels: list[str],
                n_batches: int) -> None:
    # Header
    batch_hdrs = "  ".join(f"{bl:>8}" for bl in batch_labels[:n_batches])
    print()
    print(f"{'NLR variant':<32} {'total':>5}  {batch_hdrs}  "
          f"{'accepted':>8}  {'bad':>4}  {'rate':>7}")
    print("-" * (60 + 10 * n_batches))

    for nlr_key in sorted(by_nlr):
        d = by_nlr[nlr_key]
        batch_vals = "  ".join(
            f"{d.get(f'batch_{i}', 0):>8}" for i in range(n_batches))
        pct = f"{100 * d['accepted'] / d['total']:.0f}%" if d["total"] else "-"
        print(f"{nlr_key:<32} {d['total']:>5}  {batch_vals}  "
              f"{d['accepted']:>8}  {d['still_bad']:>4}  "
              f"{d['accepted']}/{d['total']} ({pct})")

    print("-" * (60 + 10 * n_batches))
    print(f"{'PER PHENOTYPE':<32}")
    for phen in sorted(by_phen):
        d = by_phen[phen]
        batch_vals = "  ".join(
            f"{d.get(f'batch_{i}', 0):>8}" for i in range(n_batches))
        pct = f"{100 * d['accepted'] / d['total']:.0f}%" if d["total"] else "-"
        print(f"{phen:<32} {d['total']:>5}  {batch_vals}  "
              f"{d['accepted']:>8}  {d['still_bad']:>4}  "
              f"{d['accepted']}/{d['total']} ({pct})")

    # Grand totals
    total_all = sum(d["total"] for d in by_phen.values())
    total_acc = sum(d["accepted"] for d in by_phen.values())
    total_bad = sum(d["still_bad"] for d in by_phen.values())
    batch_totals = "  ".join(
        f"{sum(d.get(f'batch_{i}', 0) for d in by_phen.values()):>8}"
        for i in range(n_batches))
    pct = f"{100 * total_acc / total_all:.0f}%" if total_all else "-"
    print("-" * (60 + 10 * n_batches))
    print(f"{'TOTAL':<32} {total_all:>5}  {batch_totals}  "
          f"{total_acc:>8}  {total_bad:>4}  "
          f"{total_acc}/{total_all} ({pct})")

    if excluded_count:
        print(f"\n(Excluded {excluded_count} jobs — see EXCLUDED_NLR_PREFIXES)")

    # Batch progression summary
    print(f"\nBatch progression:")
    cumulative = 0
    for i in range(n_batches):
        rescued = sum(d.get(f"batch_{i}", 0) for d in by_phen.values())
        cumulative += rescued
        remaining = total_all - cumulative
        print(f"  {batch_labels[i]:<12}  +{rescued:>3} rescued  "
              f"→ {cumulative:>3}/{total_all} accepted  "
              f"({remaining:>3} remaining)")


def print_fun_facts(by_nlr: dict, n_batches: int,
                    batch_labels: list[str]) -> None:
    """Print fun facts: jobs that failed across all batches, total executions."""
    print(f"\n{'='*60}")
    print("Fun facts")
    print(f"{'='*60}")

    # Jobs still bad after all batches (per NLR)
    stubborn_nlrs: list[tuple[str, int, list[str]]] = []
    for nlr_key in sorted(by_nlr):
        d = by_nlr[nlr_key]
        if d["still_bad"] > 0:
            stubborn_nlrs.append((nlr_key, d["still_bad"], d["hetc_still_bad"]))

    if stubborn_nlrs:
        total_bad = sum(n for _, n, _ in stubborn_nlrs)
        print(f"\nStill unresolved after {n_batches} batch(es): "
              f"{total_bad} jobs across {len(stubborn_nlrs)} NLR variants")
        for nlr_key, n_bad, hetc_list in stubborn_nlrs:
            alleles = ", ".join(sorted(hetc_list))
            # Each job was run n_batches times with 4 seeds = n_batches*4 total AF3 executions
            total_exec = n_bad * n_batches * 4
            print(f"  {nlr_key}: {n_bad} job(s) × {n_batches} batches × 4 seeds "
                  f"= {total_exec} AF3 executions, 0 accepted")
            print(f"    alleles: {alleles}")

    # Grand total: how many AF3 GPU-hours were spent on jobs that never worked?
    total_wasted = sum(n * n_batches * 4 for _, n, _ in stubborn_nlrs)
    print(f"\n  Total wasted AF3 executions (still-bad jobs): {total_wasted}")
    print(f"  (~{total_wasted * 2:.0f} GPU-hours at ~2h/execution)")


def write_tsv(out_path: Path, by_nlr: dict, by_phen: dict,
              n_batches: int, batch_labels: list[str]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    batch_cols = [f"batch_{bl}" for bl in batch_labels[:n_batches]]
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["level", "key", "total", *batch_cols,
                    "accepted", "still_bad", "accepted_pct",
                    "hetc_accepted", "hetc_still_bad"])
        for nlr_key in sorted(by_nlr):
            d = by_nlr[nlr_key]
            pct = round(100 * d["accepted"] / d["total"], 1) if d["total"] else 0.0
            batch_vals = [d.get(f"batch_{i}", 0) for i in range(n_batches)]
            w.writerow([
                "nlr", nlr_key, d["total"], *batch_vals,
                d["accepted"], d["still_bad"], pct,
                ",".join(sorted(d["hetc_accepted"])),
                ",".join(sorted(d["hetc_still_bad"])),
            ])
        for phen in sorted(by_phen):
            d = by_phen[phen]
            pct = round(100 * d["accepted"] / d["total"], 1) if d["total"] else 0.0
            batch_vals = [d.get(f"batch_{i}", 0) for i in range(n_batches)]
            w.writerow([
                "phenotype", phen, d["total"], *batch_vals,
                d["accepted"], d["still_bad"], pct, "", "",
            ])


# ── Main ────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Per-NLR acceptance statistics with batch progression")
    p.add_argument("--af3_inputs", type=Path, default=AF3_INPUTS,
                   help=f"Folder with expected job JSONs (default: {AF3_INPUTS})")
    p.add_argument("--selected", type=Path, nargs="+", default=None,
                   help="Ordered list of selected-CIF folders (batch 0, 1, 2, ...). "
                        "Default: ../wcss/selected/original ../wcss/selected/rerun_1 "
                        "../wcss/selected/rerun_2 ../wcss/selected/rerun_3")
    p.add_argument("--report", type=Path, default=REPORT_PATH,
                   help=f"Output TSV path (default: {REPORT_PATH})")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    selected_dirs = args.selected if args.selected else DEFAULT_SELECTED_DIRS

    # Filter to only existing dirs (don't error on future rerun_N folders)
    existing_dirs = [d for d in selected_dirs if d.exists()]
    n_batches = len(existing_dirs)

    expected = list_canonical_jobs(args.af3_inputs, ".json")
    batches = list_accepted_per_batch(existing_dirs)

    n_excl = sum(1 for lc in expected if is_excluded(lc))
    n_active = len(expected) - n_excl
    all_accepted = set()
    for b in batches:
        all_accepted |= b

    print(f"Expected jobs (af3_inputs): {len(expected)}  "
          f"(excl. {n_excl} → {n_active} active)")
    print(f"Selected folders found: {n_batches}")
    for i, d in enumerate(existing_dirs):
        print(f"  {BATCH_LABELS[i]:<12} ({d}): {len(batches[i])} new accepts")
    print(f"Total accepted: {len(all_accepted)}")
    print(f"Still unresolved: {n_active - len(all_accepted)}")

    by_nlr, by_phen, excluded = aggregate(expected, batches)
    print_table(by_nlr, by_phen, excluded, BATCH_LABELS, n_batches)
    print_fun_facts(by_nlr, n_batches, BATCH_LABELS)

    write_tsv(args.report, by_nlr, by_phen, n_batches, BATCH_LABELS)
    print(f"\nWrote {args.report}")


if __name__ == "__main__":
    main()
