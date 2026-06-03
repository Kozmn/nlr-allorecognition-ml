"""
parse_nlr_fasta.py — Parse aligned HET-E/HET-D FASTA into clean sequences.

Input:
  data/sequences/het_ed_wd40/raw/wd40_raw.fa       aligned FASTA (gaps as '-')

Output:
  data/sequences/het_ed_wd40/het_ed_sequences.fasta   clean sequences, one per record
  data/sequences/het_ed_wd40/interaction_matrix.tsv   phenotype × het-c allele labels (C1-C4)
  data/sequences/het_ed_wd40/sequence_metadata.tsv    per-sequence metadata table

FASTA header written:
  >{phenotype}__{strain}__{confidence}
  e.g.  >E1__ChEhDa__confirmed
        >E3__PaWa46__inferred    (inferred from Ament-Velásquez 2025 Fig. 4)

Excluded: hnwd*, het-r, nwd* (different ligand or unknown function).
Excluded: sequences shorter than MIN_LENGTH after gap removal.

Source:
  Ament-Velásquez et al. 2025, Fig. 1b (interaction matrix C1-C4)
  Ament-Velásquez et al. 2025, Fig. 4a-b (phenotype inference)

Author: Kacper Koźmin
"""

import re
import csv
from pathlib import Path
from collections import Counter
from Bio.Seq import Seq


# ── Configuration ─────────────────────────────────────────────────────────────

INPUT_FASTA     = "data/sequences/het_ed_wd40/raw/wd40_raw.fa"
OUTPUT_FASTA    = Path("data/sequences/het_ed_wd40/het_ed_sequences.fasta")
INTERACTION_TSV = Path("data/sequences/het_ed_wd40/interaction_matrix.tsv")
METADATA_TSV    = Path("data/sequences/het_ed_wd40/sequence_metadata.tsv")

MIN_LENGTH = 100       # discard near-empty sequences after gap removal (nucleotides)
MIN_WD40_REPEATS = 10  # require at least 10 TLEGH repeats (≥1 full β-propeller worth of material)
                       # Ament-Velásquez 2025: functional alleles have 10–12 HIC repeats;
                       # fewer repeats = incomplete propeller → unreliable AF3 features

_WD40_REPEAT_RE = re.compile(r"TLEGH")  # core motif present in every HIC WD40 repeat

# ── Interaction matrix (Ament-Velásquez 2025, Fig. 1b) ────────────────────────
# 1 = incompatible (cell death), 0 = compatible
# Rows: het-e/het-d phenotypes.  Columns: het-c alleles C1-C4.

HET_C_ALLELES = ["C1", "C2", "C3", "C4"]

INTERACTION_MATRIX = {
    #          C1  C2  C3  C4    source: Ament-Velásquez 2025 Fig. 1b
    "E1":  [    0,  1,  0,  0],  # reactive with C2
    "E2":  [    1,  0,  1,  1],  # reactive with C1, C3, C4
    "E3":  [    1,  0,  0,  1],  # reactive with C1, C4
    "e4":  [    0,  0,  0,  0],  # non-reactive (null allele)
    "D1":  [    0,  1,  0,  1],  # reactive with C2, C4
    "D2":  [    0,  0,  0,  0],  # non-reactive
    "d3":  [    0,  0,  0,  0],  # non-reactive (disrupted by stop codons)
}

# ── Header regexes ─────────────────────────────────────────────────────────────

# Primary: GENE__STRAIN[.TECH]_chromosome_CHROM_START[_-]END
HEADER_RE = re.compile(
    r"^(?P<gene>[\w][\w-]*)"
    r"__"
    r"(?P<strain_full>[^_]+)"
    r"_chromosome_(?P<chrom>[^_]+)"
    r"[_](?P<start>\d+)"
    r"[_-](?P<end>\d+)$"
)

# Fallback: GENE__ACCESSION_DESCRIPTION  (GenBank, no coords)
HEADER_RE_ACCESSION = re.compile(
    r"^(?P<gene>[\w][\w-]*)"
    r"__"
    r"(?P<accession>\w+)"
    r"_(?P<description>.+)$"
)


def parse_header(header: str) -> dict:
    """Extract gene, strain, coordinates from a raw FASTA header."""
    header = header.lstrip(">").strip()

    m = HEADER_RE.match(header)
    if m:
        d = m.groupdict()
        d["raw"] = header
        sf = d.pop("strain_full")
        if "." in sf:
            parts = sf.rsplit(".", 1)
            d["strain"] = parts[0]
            d["tech"]   = parts[1]
        else:
            d["strain"] = sf
            d["tech"]   = "unknown"
        d["gene_clean"] = d["gene"].replace("het-", "").replace("Het-", "")
        return d

    m2 = HEADER_RE_ACCESSION.match(header)
    if m2:
        d = m2.groupdict()
        d["raw"]        = header
        d["strain"]     = d.pop("accession")
        d["tech"]       = "unknown"
        d["chrom"]      = "?"
        d["start"]      = "?"
        d["end"]        = "?"
        d["gene_clean"] = d["gene"].replace("het-", "").replace("Het-", "")
        return d

    return {
        "raw": header, "gene": "unknown", "gene_clean": "unknown",
        "strain": "unknown", "tech": "unknown",
        "chrom": "?", "start": "?", "end": "?",
    }


# ── Phenotype lookup ───────────────────────────────────────────────────────────

# Gene names that directly encode their phenotype
GENE_PHENOTYPE = {
    "D1":  ("D1", "confirmed"),
    "D2":  ("D2", "confirmed"),
    "d3":  ("d3", "confirmed"),
    "E1":  ("E1", "confirmed"),
    "E1A": ("E1", "confirmed"),   # FJ897789 GenBank reference
    "E2":  ("E2", "confirmed"),
    "E3":  ("E3", "confirmed"),
    "e4":  ("e4", "confirmed"),
}

# (gene_type_upper, strain_prefix) → (phenotype, confidence)
# Inferred from Ament-Velásquez 2025, Fig. 4a (het-d) and 4b (het-e).
STRAIN_PHENOTYPE = {
    # ── het-e ────────────────────────────────────────────────
    ("E", "CmEmm"):   ("E1", "confirmed"),
    ("E", "ChEhDa"):  ("E1", "confirmed"),
    ("E", "PaYp"):    ("E1", "confirmed"),
    ("E", "CoEcp"):   ("E2", "confirmed"),
    ("E", "PaZp"):    ("E3", "confirmed"),
    ("E", "CoEfp"):   ("E3", "confirmed"),
    ("E", "CaDam"):   ("e4", "confirmed"),
    ("E", "PaWa46"):  ("E3", "inferred"),
    ("E", "PaWa58"):  ("E3", "inferred"),
    ("E", "PaWa100"): ("E1", "inferred"),
    ("E", "PaWa87"):  ("E1", "inferred"),
    ("E", "PaWa63"):  ("e4", "confirmed"),
    ("E", "PaWa137"): ("e4", "inferred"),
    ("E", "PaTgp"):   ("e4", "inferred"),
    ("E", "PaWa21"):  ("e4", "inferred"),
    ("E", "PaWa53"):  ("e4", "inferred"),
    ("E", "PaWa28"):  ("e4", "inferred"),
    ("E", "Podan2"):  ("e4", "inferred"),
    # ── het-d ────────────────────────────────────────────────
    ("D", "ChEhDa"):  ("D1", "confirmed"),
    ("D", "CaDam"):   ("D1", "confirmed"),
    ("D", "CsDfp"):   ("D2", "confirmed"),
    ("D", "PaYp"):    ("D2", "confirmed"),
    ("D", "PaZp"):    ("d3", "confirmed"),
    ("D", "CoEfp"):   ("d3", "confirmed"),
    ("D", "CmEmm"):   ("d3", "confirmed"),
    ("D", "PaWa100"): ("d3", "inferred"),
    ("D", "PaTgp"):   ("D2", "inferred"),
    ("D", "PaWa87"):  ("d3", "inferred"),
    ("D", "PaWa21"):  ("d3", "inferred"),
    ("D", "PaWa28"):  ("d3", "inferred"),
    ("D", "PaWa46"):  ("d3", "inferred"),
    ("D", "PaWa53"):  ("d3", "inferred"),
    ("D", "PaWa58"):  ("d3", "inferred"),
    ("D", "PaWa63"):  ("d3", "inferred"),
    ("D", "PaWa137"): ("d3", "inferred"),
    ("D", "Podan2"):  ("d3", "inferred"),
}


def infer_phenotype(gene_clean: str, strain: str) -> tuple[str, str]:
    """Return (phenotype, confidence) for a sequence."""
    if gene_clean in GENE_PHENOTYPE:
        return GENE_PHENOTYPE[gene_clean]

    if not gene_clean or gene_clean == "unknown":
        return ("unknown", "unknown")

    gene_type = gene_clean[0].upper()
    if gene_type not in ("D", "E"):
        return ("n/a", "n/a")

    for (gt, sp), result in STRAIN_PHENOTYPE.items():
        if gt == gene_type and strain.startswith(sp):
            return result

    return ("unknown", "unknown")


# ── FASTA parser ───────────────────────────────────────────────────────────────

def _build_record(header: str, raw_parts: list[str]) -> dict:
    """Translate one aligned nucleotide sequence into an amino acid record."""
    raw   = "".join(raw_parts)
    clean = raw.replace("-", "").replace(".", "")
    translated = str(Seq(clean).translate())
    has_internal_stop = "*" in translated.rstrip("*")
    aa = translated.split("*")[0]
    return {
        "header": header, "raw_seq": raw, "clean_seq": clean,
        "aa_seq": aa, "length": len(clean),
        "has_internal_stop": has_internal_stop,
    }


def parse_aligned_fasta(filepath: str) -> list[dict]:
    records, current_header, current_parts = [], None, []
    with open(filepath) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if current_header is not None:
                    records.append(_build_record(current_header, current_parts))
                current_header, current_parts = line, []
            else:
                current_parts.append(line)
    if current_header:
        records.append(_build_record(current_header, current_parts))
    return records


def enrich_and_filter(records: list[dict]) -> list[dict]:
    """Add phenotype fields; discard non het-e/het-d and too-short sequences."""
    kept = []
    for rec in records:
        meta = parse_header(rec["header"])
        gc   = meta.get("gene_clean", "unknown")

        if not gc or gc == "unknown":
            continue
        if gc[0].upper() not in ("D", "E"):
            continue

        phenotype, confidence = infer_phenotype(gc, meta.get("strain", ""))

        if rec["length"] < MIN_LENGTH:
            continue
        if rec.get("has_internal_stop"):
            continue  # truncated by stop codon: misassembly or disrupted allele
        if len(_WD40_REPEAT_RE.findall(rec["aa_seq"])) < MIN_WD40_REPEATS:
            continue  # too few WD40 repeats: incomplete propeller, unreliable AF3 structure

        rec["gene"]       = meta.get("gene", "unknown")
        rec["gene_clean"] = gc
        rec["strain"]     = meta.get("strain", "unknown")
        rec["tech"]       = meta.get("tech", "unknown")
        rec["chrom"]      = meta.get("chrom", "?")
        rec["phenotype"]  = phenotype
        rec["confidence"] = confidence
        rec["interacts"]  = INTERACTION_MATRIX.get(phenotype, [0, 0, 0, 0])
        kept.append(rec)

    return kept


# ── Writers ────────────────────────────────────────────────────────────────────

def write_fasta(records: list[dict], path: Path) -> None:
    """Write amino acid sequences (translated from nucleotide CDS)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for rec in records:
            fh.write(f">{rec['phenotype']}__{rec['strain']}__{rec['confidence']}\n")
            seq = rec["aa_seq"]
            for i in range(0, len(seq), 80):
                fh.write(seq[i:i+80] + "\n")
    print(f"  [fasta]  {len(records)} sequences → {path}")


def write_interaction_matrix(path: Path) -> None:
    """Write phenotype × C1-C4 interaction matrix as TSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    notes = {
        "E1": "reactive – C2",
        "E2": "reactive – C1, C3, C4",
        "E3": "reactive – C1, C4",
        "e4": "non-reactive",
        "D1": "reactive – C2, C4",
        "D2": "non-reactive",
        "d3": "non-reactive",
    }
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["phenotype"] + HET_C_ALLELES + ["notes"])
        for phenotype, row in INTERACTION_MATRIX.items():
            w.writerow([phenotype] + row + [notes.get(phenotype, "")])
    print(f"  [matrix] interaction matrix (C1-C4) → {path}")


def write_metadata(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["phenotype", "confidence", "gene", "gene_clean", "strain",
              "tech", "chrom", "length", "aa_length"] + HET_C_ALLELES + ["header"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, delimiter="\t",
                           extrasaction="ignore")
        w.writeheader()
        for rec in records:
            row = dict(rec)
            row["aa_length"] = len(rec["aa_seq"])
            for i, allele in enumerate(HET_C_ALLELES):
                row[allele] = rec["interacts"][i]
            w.writerow(row)
    print(f"  [meta]   metadata → {path}")


def print_summary(records: list[dict]) -> None:
    print("\n── Sequences kept ───────────────────────────────────────────────")
    counts = Counter((r["phenotype"], r["confidence"]) for r in records)
    for (ph, conf), n in sorted(counts.items()):
        tag   = "  [inferred]" if conf == "inferred" else ""
        ints  = INTERACTION_MATRIX.get(ph, [])
        c_str = ", ".join(HET_C_ALLELES[i] for i, v in enumerate(ints) if v)
        react = f"→ reacts with {c_str}" if c_str else "→ non-reactive"
        print(f"  {ph:>4}  ({n:2d} seq) {react:<30}{tag}")
    n_conf = sum(n for (_, c), n in counts.items() if c == "confirmed")
    n_inf  = sum(n for (_, c), n in counts.items() if c == "inferred")
    print(f"\n  Total: {len(records)}  ({n_conf} confirmed, {n_inf} inferred)")
    unknown = [r for r in records if r["phenotype"] == "unknown"]
    if unknown:
        print(f"\n  WARNING: {len(unknown)} sequences with unknown phenotype:")
        for r in unknown:
            print(f"    {r['header']}")
    print("─────────────────────────────────────────────────────────────────\n")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print(f"Reading: {INPUT_FASTA}")
    raw     = parse_aligned_fasta(INPUT_FASTA)
    print(f"  {len(raw)} raw sequences in file")

    records = enrich_and_filter(raw)
    print(f"  {len(records)} het-e/het-d sequences after filtering")
    print(f"  (skipped {len(raw) - len(records)}: hnwd/het-r/nwd/too-short)")

    print_summary(records)

    write_fasta(records, OUTPUT_FASTA)
    write_interaction_matrix(INTERACTION_TSV)
    write_metadata(records, METADATA_TSV)

    print("\nDone.")


if __name__ == "__main__":
    main()
