"""
validate_af3_contacts.py — Validate AF3 predictions against known biology.

For each AF3 job this script checks whether inter-chain contacts between
the NLR (chain A, HET-E/HET-D) and HET-C (chain B) land on positions
under diversifying selection:

  - NLR side: positions 10, 11, 12, 14, 30, 32, 39 within each WD40 repeat
    (Paoletti et al. 2007; Ament-Velásquez 2025)
  - HET-C side: positions 118, 133, 153
    (Bastiaans et al. 2014)

For each job we report the observed enrichment of hypervariable contacts
and its permutation p-value. The null model is uniform placement of the
observed number of contacts over the same residue space (NLR residues that
fall within any detected repeat for the NLR test, and HET-C residues 1..L_B
for the HET-C test).

Input:
  data/af3_outputs/<job>/     flat bundle: CIF + JSONs per job folder

Output:
  data/validation/reports/contact_validation.tsv
  data/validation/reports/summary.txt

Usage:
  cd thesis/
  python scripts/validate_af3_contacts.py
  python scripts/validate_af3_contacts.py --output_dir /path/to/af3_outputs \\
      --n_permutations 20000 --seed 42

Author: Kacper Koźmin
"""

import argparse
import csv
import json
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


# ── Defaults ─────────────────────────────────────────────────────────────────

AF3_OUTPUTS = Path("data/af3_outputs")
AF3_INPUTS_LABELED = Path("data/af3_inputs/labeled")
VALIDATION_DIR = Path("data/validation/reports")


# ── Job name parsing ────────────────────────────────────────────────────────

def parse_job_name(folder_name: str) -> dict:
    """Parse job folder name like 'd3_pazp_conf_vs_c1_l0'."""
    parts = folder_name.lower().split("_vs_")
    if len(parts) != 2:
        return {"phenotype": "?", "strain": "?", "confidence": "?",
                "hetc_allele": "?", "label": "?", "raw": folder_name}
    nlr_tokens = parts[0].split("_")
    hetc_tokens = parts[1].split("_")
    phenotype = nlr_tokens[0].upper() if nlr_tokens else "?"
    if phenotype in ("D3", "E4"):
        phenotype = phenotype[0].lower() + phenotype[1]
    return {
        "phenotype": phenotype,
        "strain": nlr_tokens[1] if len(nlr_tokens) > 1 else "?",
        "confidence": nlr_tokens[2] if len(nlr_tokens) > 2 else "?",
        "hetc_allele": hetc_tokens[0].upper() if hetc_tokens else "?",
        "label": hetc_tokens[1] if len(hetc_tokens) > 1 else "?",
        "raw": folder_name,
    }


# ── AF3 output loading ──────────────────────────────────────────────────────

def find_model_dir(job_folder: Path) -> Path | None:
    """Return job_folder if it contains a CIF and a confidences JSON.

    Expects flat bundle layout from collect_selected.py.
    """
    has_cif = bool(list(job_folder.glob("*.cif")))
    has_conf = bool([p for p in job_folder.glob("*confidences*.json")
                     if "summary" not in p.name])
    return job_folder if (has_cif and has_conf) else None


def load_contacts_data(model_dir: Path) -> dict | None:
    """Load contact_probs, PAE, and chain lengths from *confidences*.json."""
    files = [p for p in model_dir.glob("*confidences*.json")
             if "summary" not in p.name]
    if not files:
        return None
    with open(files[0]) as f:
        data = json.load(f)
    if "contact_probs" not in data or "pae" not in data or "token_chain_ids" not in data:
        return None
    chain_ids = data["token_chain_ids"]
    return {
        "contact_probs": data["contact_probs"],
        "pae": data["pae"],
        "chain_a_len": sum(1 for c in chain_ids if c == "A"),
        "chain_b_len": sum(1 for c in chain_ids if c == "B"),
    }


def load_summary_confidences(model_dir: Path) -> dict:
    """Load summary confidence scores (ipTM, pTM, ranking_score)."""
    files = list(model_dir.glob("*summary_confidences*.json"))
    if not files:
        return {}
    with open(files[0]) as f:
        return json.load(f)


def _parse_af3_json(path: Path) -> tuple[str, str]:
    """Extract (nlr_seq, hetc_seq) from an AF3 input/request JSON."""
    with open(path) as f:
        job = json.load(f)
    if isinstance(job, list):
        job = job[0]
    seqs = job.get("sequences", [])

    def _seq(entry: dict) -> str:
        for key in ("protein", "proteinChain"):
            if key in entry:
                return entry[key].get("sequence", "")
        return ""

    nlr = _seq(seqs[0]) if len(seqs) > 0 else ""
    hetc = _seq(seqs[1]) if len(seqs) > 1 else ""
    return nlr, hetc


def load_sequences(job_folder: Path,
                   af3_inputs_dir: Path = AF3_INPUTS_LABELED
                   ) -> tuple[str, str]:
    """Load NLR and HET-C sequences for a job.

    Priority:
      1. `*job_request.json` inside the bundle folder (legacy AF3 Server layout).
      2. `data/af3_inputs/labeled/<job>.json` — the canonical input JSON that
         generate_af3_inputs.py produces. Matched case-insensitively so that
         the lowercase folder name from WCSS (e.g. `d1_chehdap_conf_vs_c2_l1`)
         finds the mixed-case input file (`D1_CheHdaP_conf_vs_C2_L1.json`).

    Supports both 'protein' and 'proteinChain' key dialects.
    """
    candidates = (list(job_folder.glob("*job_request.json"))
                  + list(job_folder.parent.glob(
                         f"{job_folder.name}*job_request.json")))
    if candidates:
        return _parse_af3_json(candidates[0])

    if af3_inputs_dir.exists():
        target = job_folder.name.lower()
        for f in af3_inputs_dir.glob("*.json"):
            if f.stem.lower() == target:
                return _parse_af3_json(f)

    return "", ""


# ── Contact extraction ──────────────────────────────────────────────────────

def extract_interchain_contacts(data: dict,
                                threshold: float = CONTACT_PROB_THRESHOLD
                                ) -> list[dict]:
    """Return contacts above `threshold` as dicts with 1-indexed positions."""
    cp = data["contact_probs"]
    pae = data["pae"]
    n_a = data["chain_a_len"]
    n_b = data["chain_b_len"]
    out: list[dict] = []
    for i in range(n_a):
        row = cp[i]
        for j in range(n_a, n_a + n_b):
            if row[j] >= threshold:
                out.append({
                    "nlr_res": i + 1,
                    "hetc_res": j - n_a + 1,
                    "contact_prob": row[j],
                    "pae": pae[i][j],
                })
    return out


# ── Statistical tests ───────────────────────────────────────────────────────

def permutation_test_nlr(nlr_contact_residues: list[int],
                         boundaries: list[tuple[int, int]],
                         n_perm: int,
                         rng: np.random.Generator
                         ) -> tuple[int, int, float]:
    """Permutation test: are NLR contacts enriched on hypervariable positions?

    Null model: each contact lands uniformly at random on any residue that
    falls within *some* detected WD40 repeat. We only consider contacts whose
    residue is within a detected repeat (contacts outside repeats are ignored).

    Returns (observed_hits, total_in_repeats, p_value). p_value uses
    Laplace (+1) smoothing: (count_ge + 1) / (n_perm + 1).
    """
    pool: list[int] = []
    hyper_pool_mask: list[bool] = []
    for start, end in boundaries:
        for r in range(start, end + 1):
            pool.append(r)
            intra = r - start + 1
            hyper_pool_mask.append(intra in NLR_REPEAT_HYPERVARIABLE)

    if not pool:
        return (0, 0, 1.0)

    pool_arr = np.asarray(pool, dtype=np.int32)
    hyper_arr = np.asarray(hyper_pool_mask, dtype=bool)

    in_repeat = [r for r in nlr_contact_residues if r in set(pool)]
    n = len(in_repeat)
    if n == 0:
        return (0, 0, 1.0)

    pool_to_hyper = {r: h for r, h in zip(pool, hyper_pool_mask)}
    observed = sum(1 for r in in_repeat if pool_to_hyper[r])

    sims = rng.choice(len(pool_arr), size=(n_perm, n), replace=True)
    sim_hits = hyper_arr[sims].sum(axis=1)
    p_value = float(((sim_hits >= observed).sum() + 1) / (n_perm + 1))

    return (observed, n, p_value)


def permutation_test_hetc(hetc_contact_residues: list[int],
                          hetc_length: int,
                          n_perm: int,
                          rng: np.random.Generator
                          ) -> tuple[int, int, float]:
    """Permutation test for HET-C hypervariable contact enrichment.

    Null: each contact picks a uniform random position in 1..L_B.
    """
    n = len(hetc_contact_residues)
    if n == 0 or hetc_length <= 0:
        return (0, 0, 1.0)

    observed = sum(1 for r in hetc_contact_residues if r in HETC_HYPERVARIABLE)

    hyper_mask = np.zeros(hetc_length + 1, dtype=bool)
    for p in HETC_HYPERVARIABLE:
        if 1 <= p <= hetc_length:
            hyper_mask[p] = True

    sims = rng.integers(low=1, high=hetc_length + 1, size=(n_perm, n))
    sim_hits = hyper_mask[sims].sum(axis=1)
    p_value = float(((sim_hits >= observed).sum() + 1) / (n_perm + 1))

    return (observed, n, p_value)


# ── Analysis per job ────────────────────────────────────────────────────────

def analyze_job(contacts: list[dict],
                nlr_sequence: str,
                hetc_length: int,
                boundaries: list[tuple[int, int]],
                n_perm: int,
                rng: np.random.Generator
                ) -> dict:
    """Compute contact enrichment statistics and permutation p-values."""
    total = len(contacts)
    if total == 0:
        return {
            "total_contacts": 0,
            "nlr_hyper_contacts": 0, "nlr_hyper_pct": 0.0,
            "nlr_contacts_in_repeats": 0, "nlr_hyper_pvalue": 1.0,
            "hetc_hyper_contacts": 0, "hetc_hyper_pct": 0.0,
            "hetc_hyper_pvalue": 1.0,
            "both_hyper_contacts": 0,
            "hetc_118_contacts": 0, "hetc_133_contacts": 0,
            "hetc_153_contacts": 0,
            "contacted_repeats": "", "n_repeats_contacted": 0,
            "n_repeats_total": len(boundaries),
        }

    nlr_residues = [c["nlr_res"] for c in contacts]
    hetc_residues = [c["hetc_res"] for c in contacts]

    nlr_hyper = 0
    both_hyper = 0
    contacted_repeats: set[int] = set()
    hetc_pos_counts = {118: 0, 133: 0, 153: 0}

    for c in contacts:
        is_nlr_hyper, repeat_num, _intra = classify_nlr_residue(
            c["nlr_res"], boundaries)
        if repeat_num is not None:
            contacted_repeats.add(repeat_num)
        if is_nlr_hyper:
            nlr_hyper += 1
        if c["hetc_res"] in HETC_HYPERVARIABLE:
            if is_nlr_hyper:
                both_hyper += 1
        if c["hetc_res"] in hetc_pos_counts:
            hetc_pos_counts[c["hetc_res"]] += 1

    hetc_hyper = sum(1 for r in hetc_residues if r in HETC_HYPERVARIABLE)

    nlr_obs, nlr_in_rep, nlr_p = permutation_test_nlr(
        nlr_residues, boundaries, n_perm=n_perm, rng=rng)
    _hetc_obs, _hetc_n, hetc_p = permutation_test_hetc(
        hetc_residues, hetc_length, n_perm=n_perm, rng=rng)

    return {
        "total_contacts": total,
        "nlr_hyper_contacts": nlr_hyper,
        "nlr_hyper_pct": round(nlr_hyper / total * 100, 2),
        "nlr_contacts_in_repeats": nlr_in_rep,
        "nlr_hyper_pvalue": round(nlr_p, 5),
        "hetc_hyper_contacts": hetc_hyper,
        "hetc_hyper_pct": round(hetc_hyper / total * 100, 2),
        "hetc_hyper_pvalue": round(hetc_p, 5),
        "both_hyper_contacts": both_hyper,
        "hetc_118_contacts": hetc_pos_counts[118],
        "hetc_133_contacts": hetc_pos_counts[133],
        "hetc_153_contacts": hetc_pos_counts[153],
        "contacted_repeats": ",".join(str(r) for r in sorted(contacted_repeats)),
        "n_repeats_contacted": len(contacted_repeats),
        "n_repeats_total": len(boundaries),
    }


# ── Folder collection ───────────────────────────────────────────────────────

def collect_job_folders(output_dir: Path) -> list[Path]:
    """Return subdirectories that contain a CIF file (flat bundle layout)."""
    folders: list[Path] = []
    if not output_dir.exists():
        return folders
    for job_dir in sorted(output_dir.iterdir()):
        if not job_dir.is_dir() or job_dir.name.startswith("."):
            continue
        if list(job_dir.glob("*.cif")):
            folders.append(job_dir)
    return folders


# ── Main ────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validate AF3 NLR×HET-C contacts against "
                    "diversifying-selection sites")
    p.add_argument("--output_dir", type=Path, default=AF3_OUTPUTS,
                   help=f"AF3 outputs root (default: {AF3_OUTPUTS})")
    p.add_argument("--report_dir", type=Path, default=VALIDATION_DIR,
                   help=f"Report output folder (default: {VALIDATION_DIR})")
    p.add_argument("--threshold", type=float, default=CONTACT_PROB_THRESHOLD,
                   help=f"Contact probability cutoff "
                        f"(default: {CONTACT_PROB_THRESHOLD})")
    p.add_argument("--n_permutations", type=int, default=10000,
                   help="Number of permutations (default: 10000)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for permutations (default: 42)")
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

    rows: list[dict] = []
    for job_folder in folders:
        job_info = parse_job_name(job_folder.name)

        best = find_model_dir(job_folder)
        if best is None:
            print(f"  SKIP {job_folder.name}: no CIF + confidences found")
            continue

        data = load_contacts_data(best)
        if data is None:
            print(f"  SKIP {job_folder.name}: no contacts JSON in {best.name}")
            continue

        conf = load_summary_confidences(best)
        nlr_seq, _hetc_seq = load_sequences(job_folder)
        boundaries = find_repeat_boundaries(nlr_seq)

        contacts = extract_interchain_contacts(data, threshold=args.threshold)
        stats = analyze_job(contacts, nlr_seq, data["chain_b_len"],
                            boundaries, n_perm=args.n_permutations, rng=rng)

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

        print(f"  {job_folder.name}: {stats['total_contacts']} contacts, "
              f"NLR hyper={stats['nlr_hyper_pct']:.1f}% "
              f"(p={stats['nlr_hyper_pvalue']:.3f}), "
              f"HET-C hyper={stats['hetc_hyper_pct']:.1f}% "
              f"(p={stats['hetc_hyper_pvalue']:.3f}){flag}")

    if not rows:
        print("No jobs produced output.")
        return

    # TSV
    tsv_path = args.report_dir / "contact_validation.tsv"
    with open(tsv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()),
                                delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {tsv_path}")

    # Summary
    with_contacts = [r for r in rows if r["total_contacts"] > 0]
    sig_nlr = [r for r in with_contacts if r["nlr_hyper_pvalue"] < 0.05]
    sig_hetc = [r for r in with_contacts if r["hetc_hyper_pvalue"] < 0.05]

    summary_path = args.report_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write("AF3 Contact Validation Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Jobs analyzed:           {len(rows)}\n")
        f.write(f"Jobs with contacts >={args.threshold}: "
                f"{len(with_contacts)}\n")
        f.write(f"Jobs without contacts:   "
                f"{len(rows) - len(with_contacts)}\n\n")
        f.write(f"Permutations per test:   {args.n_permutations}\n")
        f.write(f"Significance threshold:  p < 0.05\n\n")
        f.write(f"NLR hypervariable enrichment (significant jobs): "
                f"{len(sig_nlr)}/{len(with_contacts)}\n")
        f.write(f"HET-C hypervariable enrichment (significant jobs): "
                f"{len(sig_hetc)}/{len(with_contacts)}\n\n")

        if with_contacts:
            avg_nlr = np.mean([r["nlr_hyper_pct"] for r in with_contacts])
            avg_hetc = np.mean([r["hetc_hyper_pct"] for r in with_contacts])
            exp_nlr = len(NLR_REPEAT_HYPERVARIABLE) / REPEAT_LENGTH * 100
            exp_hetc_vals = [len(HETC_HYPERVARIABLE) / r["hetc_length"] * 100
                             for r in with_contacts if r["hetc_length"]]
            exp_hetc = (float(np.mean(exp_hetc_vals))
                        if exp_hetc_vals else 0.0)

            f.write("Average observed vs expected (uniform) hit percentage:\n")
            f.write(f"  NLR:   observed {avg_nlr:.2f}%   "
                    f"expected {exp_nlr:.2f}%   "
                    f"ratio {avg_nlr / exp_nlr:.2f}x\n")
            f.write(f"  HET-C: observed {avg_hetc:.2f}%   "
                    f"expected {exp_hetc:.2f}%   "
                    f"ratio {avg_hetc / exp_hetc if exp_hetc else 0:.2f}x\n\n")

            avg_118 = np.mean([r["hetc_118_contacts"] for r in with_contacts])
            avg_133 = np.mean([r["hetc_133_contacts"] for r in with_contacts])
            avg_153 = np.mean([r["hetc_153_contacts"] for r in with_contacts])
            f.write("HET-C per-position average contacts:\n")
            f.write(f"  Position 118: {avg_118:.2f}\n")
            f.write(f"  Position 133: {avg_133:.2f}\n")
            f.write(f"  Position 153: {avg_153:.2f}\n")

    print(f"Wrote {summary_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
