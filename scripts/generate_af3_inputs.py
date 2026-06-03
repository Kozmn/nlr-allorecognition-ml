"""
generate_af3_inputs.py — Generate AlphaFold 3 input JSON files for all
HET-E/HET-D × HET-C pairs (alleles C1-C11).

Reads:
  data/sequences/het_ed_wd40/het_ed_sequences.fasta   NLR sequences (from parse_nlr_fasta.py)
  data/sequences/het_c/HET-C_C*.fasta               HET-C alleles (from download_hetc_sequences.py)

Writes:
  data/af3_inputs/labeled/     one JSON per pair with a known interaction label
  data/sequences/het_c/interaction_matrix_full.tsv   full C1-C11 matrix

Interaction labels  (1 = incompatible / cell death, 0 = compatible):
  All 11 alleles (C1-C11) have known interaction patterns.
  Sources:
    C1-C4  : Ament-Velásquez et al. 2025, Fig. 1b
    C5-C11 : Bastiaans et al. 2014 (Mol. Biol. Evol.), Fig. 1

Author: Kacper Koźmin
"""

import csv
import json
import re
from pathlib import Path
from itertools import product


# ── Paths ──────────────────────────────────────────────────────────────────────

HETDE_FASTA   = Path("data/sequences/het_ed_wd40/het_ed_sequences.fasta")
HETC_DIR      = Path("data/sequences/het_c")
LABELED_DIR   = Path("data/af3_inputs/labeled")
MATRIX_TSV    = Path("data/sequences/het_c/interaction_matrix_full.tsv")

# ── Full interaction matrix C1-C11 ─────────────────────────────────────────────
# 1 = incompatible (cell death)
# 0 = compatible
# Sources: Ament-Velásquez 2025 Fig.1b (C1-C4), Bastiaans 2014 Fig.1 (C5-C11)

INTERACTION_MATRIX: dict[tuple[str, str], int] = {
    # ── C1: incompatible with E2, E3 ─────────────────────────────────────────
    ("E1", "C1"): 0, ("E2", "C1"): 1, ("E3", "C1"): 1, ("e4", "C1"): 0,
    ("D1", "C1"): 0, ("D2", "C1"): 0, ("d3", "C1"): 0,
    # ── C2: incompatible with E1, D1 ─────────────────────────────────────────
    ("E1", "C2"): 1, ("E2", "C2"): 0, ("E3", "C2"): 0, ("e4", "C2"): 0,
    ("D1", "C2"): 1, ("D2", "C2"): 0, ("d3", "C2"): 0,
    # ── C3: incompatible with E2 ─────────────────────────────────────────────
    ("E1", "C3"): 0, ("E2", "C3"): 1, ("E3", "C3"): 0, ("e4", "C3"): 0,
    ("D1", "C3"): 0, ("D2", "C3"): 0, ("d3", "C3"): 0,
    # ── C4: incompatible with E2, E3, D1, D2 ─────────────────────────────────
    ("E1", "C4"): 0, ("E2", "C4"): 1, ("E3", "C4"): 1, ("e4", "C4"): 0,
    ("D1", "C4"): 1, ("D2", "C4"): 1, ("d3", "C4"): 0,
    # ── C5: incompatible with E2, E3 (= C1 class) ───────────────────────────
    ("E1", "C5"): 0, ("E2", "C5"): 1, ("E3", "C5"): 1, ("e4", "C5"): 0,
    ("D1", "C5"): 0, ("D2", "C5"): 0, ("d3", "C5"): 0,
    # ── C6: incompatible with E2 (= C3 class) ───────────────────────────────
    ("E1", "C6"): 0, ("E2", "C6"): 1, ("E3", "C6"): 0, ("e4", "C6"): 0,
    ("D1", "C6"): 0, ("D2", "C6"): 0, ("d3", "C6"): 0,
    # ── C7: incompatible with E1, D1, D2 ─────────────────────────────────────
    ("E1", "C7"): 1, ("E2", "C7"): 0, ("E3", "C7"): 0, ("e4", "C7"): 0,
    ("D1", "C7"): 1, ("D2", "C7"): 1, ("d3", "C7"): 0,
    # ── C8: incompatible with E1, E2, E3, D1 ─────────────────────────────────
    ("E1", "C8"): 1, ("E2", "C8"): 1, ("E3", "C8"): 1, ("e4", "C8"): 0,
    ("D1", "C8"): 1, ("D2", "C8"): 0, ("d3", "C8"): 0,
    # ── C9: incompatible with E2, D1, D2 ─────────────────────────────────────
    ("E1", "C9"): 0, ("E2", "C9"): 1, ("E3", "C9"): 0, ("e4", "C9"): 0,
    ("D1", "C9"): 1, ("D2", "C9"): 1, ("d3", "C9"): 0,
    # ── C10: incompatible with E2, E3, D1, D2 (= C4 class) ──────────────────
    ("E1", "C10"): 0, ("E2", "C10"): 1, ("E3", "C10"): 1, ("e4", "C10"): 0,
    ("D1", "C10"): 1, ("D2", "C10"): 1, ("d3", "C10"): 0,
    # ── C11: incompatible with E2, E3, D1, D2 (= C4 class) ──────────────────
    ("E1", "C11"): 0, ("E2", "C11"): 1, ("E3", "C11"): 1, ("e4", "C11"): 0,
    ("D1", "C11"): 1, ("D2", "C11"): 1, ("d3", "C11"): 0,
}

ALL_ALLELES       = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11"]
PHENOTYPE_ORDER   = ["E1", "E2", "E3", "e4", "D1", "D2", "d3"]


# ── FASTA readers ──────────────────────────────────────────────────────────────

def read_multifasta(filepath: Path) -> list[dict]:
    records, header, parts = [], None, []
    with open(filepath) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if header:
                    records.append({"header": header, "seq": "".join(parts)})
                header, parts = line[1:], []
            else:
                parts.append(line)
    if header:
        records.append({"header": header, "seq": "".join(parts)})
    return records


def load_hetc_fastas(directory: Path) -> list[dict]:
    records   = []
    allele_re = re.compile(r"^HET-C_([A-Za-z0-9]+)\.fasta$")
    for fp in sorted(directory.glob("HET-C_*.fasta")):
        m      = allele_re.match(fp.name)
        allele = m.group(1) if m else "unknown"
        recs   = read_multifasta(fp)
        if recs:
            records.append({"allele": allele,
                             "header": recs[0]["header"],
                             "seq":    recs[0]["seq"]})
    return records


def parse_nlr_header(header: str) -> dict:
    """Split '{phenotype}__{strain}__{confidence}' header."""
    parts = header.split("__")
    return {
        "phenotype":  parts[0] if len(parts) > 0 else "unknown",
        "strain":     parts[1] if len(parts) > 1 else "unknown",
        "confidence": parts[2] if len(parts) > 2 else "unknown",
    }


# ── AF3 JSON builder ───────────────────────────────────────────────────────────

def make_af3_json(nlr_name, nlr_seq, hetc_name, hetc_seq,
                  label, job_name) -> dict:
    return {
        "name": job_name,
        "sequences": [
            {"proteinChain": {"sequence": nlr_seq,  "count": 1}},
            {"proteinChain": {"sequence": hetc_seq, "count": 1}},
        ],
        "_pipeline_meta": {
            "receptor":          nlr_name,
            "ligand":            hetc_name,
            "interaction_label": label if label is not None else "unknown",
        },
        "dialect": "alphafold3",
        "version": 1,
    }


# ── Matrix TSV writer ──────────────────────────────────────────────────────────

def write_full_matrix_tsv(path: Path) -> None:
    """Write full C1-C11 interaction matrix for future ESM2 use."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["phenotype"] + ALL_ALLELES)
        for ph in PHENOTYPE_ORDER:
            row = [ph]
            for allele in ALL_ALLELES:
                val = INTERACTION_MATRIX.get((ph, allele))
                row.append(val)
            w.writerow(row)
    print(f"  [matrix] full C1-C11 matrix → {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    LABELED_DIR.mkdir(parents=True, exist_ok=True)

    if not HETDE_FASTA.exists():
        print(f"ERROR: {HETDE_FASTA} not found — run parse_nlr_fasta.py first.")
        return

    nlr_records = read_multifasta(HETDE_FASTA)
    for rec in nlr_records:
        rec.update(parse_nlr_header(rec["header"]))

    hetc_records = load_hetc_fastas(HETC_DIR)
    if not hetc_records:
        print(f"ERROR: No HET-C FASTAs in {HETC_DIR} — run download_hetc_sequences.py first.")
        return

    print(f"NLR sequences  : {len(nlr_records)}")
    allele_names = [r["allele"] for r in hetc_records]
    print(f"HET-C alleles  : {allele_names}")
    print(f"Total pairs    : {len(nlr_records) * len(hetc_records)}\n")

    n_labeled = n_pos = n_neg = 0

    for nlr, hetc in product(nlr_records, hetc_records):
        phenotype  = nlr["phenotype"]
        strain     = nlr["strain"]
        confidence = nlr["confidence"]
        allele     = hetc["allele"]

        label    = INTERACTION_MATRIX.get((phenotype, allele))
        conf_tag = "conf" if confidence == "confirmed" else "inf"
        label_tag = f"L{label}"

        job_name = f"{phenotype}_{strain}_{conf_tag}_vs_{allele}_{label_tag}"
        job_name = job_name[:50]

        af3 = make_af3_json(
            nlr_name  = nlr["header"],
            nlr_seq   = nlr["seq"],
            hetc_name = hetc["header"],
            hetc_seq  = hetc["seq"],
            label     = label,
            job_name  = job_name,
        )

        out_path = LABELED_DIR / f"{job_name}.json"
        n_labeled += 1
        if label == 1:
            n_pos += 1
        else:
            n_neg += 1

        with open(out_path, "w") as fh:
            json.dump(af3, fh, indent=2)

    write_full_matrix_tsv(MATRIX_TSV)

    print(f"Labeled   : {n_labeled} files → {LABELED_DIR}/")
    print(f"  L1 (incompatible) : {n_pos}")
    print(f"  L0 (compatible)   : {n_neg}")
    print(f"\nDone. {n_labeled} total pairs.")


if __name__ == "__main__":
    main()
