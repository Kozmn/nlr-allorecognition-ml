"""
download_hetc_sequences.py — Fetch all HET-C allele sequences (C1-C11) from NCBI.

Downloads:
  C1-C4  : protein records (AAA20541-44)   via NCBI protein DB
  C5-C11 : nucleotide CDS translated to AA (KF951052-58) via NCBI nuccore

Output:
  data/sequences/het_c/HET-C_C1.fasta  …  HET-C_C11.fasta   (one file per allele, 208 aa each)


Author: Kacper Koźmin
"""

import ssl
import time
import urllib.request
from pathlib import Path

try:
    import certifi
    SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    SSL_CONTEXT = ssl.create_default_context()

OUTPUT_DIR = Path("data/sequences/het_c")
NCBI_BASE  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

# ── Accessions ─────────────────────────────────────────────────────────────────
# C1-C4: protein accessions (db=protein, rettype=fasta)
# C5-C11: nucleotide accessions → CDS translated to AA (db=nuccore, rettype=fasta_cds_aa)

PROTEIN_ACCESSIONS = {
    # Source: Saupe et al. 1994/1995 (deposited NCBI 1994)
    # C1, C3, C4: unpublished deposit (locus PANVIPB, PANVIPD, PANVIPE)
    # C2: AAA20542.1 = protein record explicitly named HET-C2; identical to
    #     U05236.1 (Saupe et al. 1994, PNAS 91:5927)
    "C1": "AAA33626.1",
    "C2": "AAA20542.1",
    "C3": "AAA33628.1",
    "C4": "AAA33629.1",
}

NUCLEOTIDE_ACCESSIONS = {
    # C5-C11: Chevanne et al. 2010, PMC3969566
    "C5":  "KF951052",
    "C6":  "KF951053",
    "C7":  "KF951054",
    "C8":  "KF951055",
    "C9":  "KF951056",
    "C10": "KF951057",
    "C11": "KF951058",
}

FETCH_DELAY = 0.4  # seconds between NCBI requests (NCBI rate limit: 3/s)


# ── NCBI fetch helpers ─────────────────────────────────────────────────────────

def fetch_url(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, context=SSL_CONTEXT, timeout=30) as resp:
        return resp.read().decode("utf-8")


def fetch_protein_fasta(accession: str) -> str:
    url = (f"{NCBI_BASE}?db=protein&id={accession}"
           f"&rettype=fasta&retmode=text")
    return fetch_url(url)


def fetch_nucleotide_cds_aa(accession: str) -> str:
    """Fetch nucleotide record and return the CDS translated to amino acids."""
    url = (f"{NCBI_BASE}?db=nuccore&id={accession}"
           f"&rettype=fasta_cds_aa&retmode=text")
    return fetch_url(url)


# ── FASTA cleaning ─────────────────────────────────────────────────────────────

def clean_fasta(raw: str) -> tuple[str, str]:
    """Return (header_line, sequence) from a raw FASTA string."""
    lines  = [l.strip() for l in raw.strip().splitlines() if l.strip()]
    header = next((l for l in lines if l.startswith(">")), None)
    seq    = "".join(l for l in lines if not l.startswith(">"))
    seq    = seq.replace("-", "").replace(" ", "").upper()
    return header or ">unknown", seq


def write_fasta(allele: str, header: str, seq: str, output_dir: Path) -> None:
    path = output_dir / f"HET-C_{allele}.fasta"
    with open(path, "w") as fh:
        fh.write(f">{allele}_HET-C\n")
        for i in range(0, len(seq), 80):
            fh.write(seq[i:i+80] + "\n")


# ── Download routines ──────────────────────────────────────────────────────────

def download_allele(allele: str, accession: str, fetch_fn) -> tuple[str, int]:
    """Download one allele. Returns (status_message, sequence_length)."""
    try:
        raw = fetch_fn(accession)
        time.sleep(FETCH_DELAY)
        header, seq = clean_fasta(raw)
        if not seq or len(seq) < 50:
            return f"  [ERROR] {allele} ({accession}): empty or too short", 0
        write_fasta(allele, header, seq, OUTPUT_DIR)
        return f"  [ok]    {allele} ({accession})  {len(seq)} aa", len(seq)
    except Exception as exc:
        return f"  [ERROR] {allele} ({accession}): {exc}", 0


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    errors = []

    print("Downloading C1-C4 (protein records, Saupe et al. 1995)...")
    for allele, acc in PROTEIN_ACCESSIONS.items():
        msg, length = download_allele(allele, acc, fetch_protein_fasta)
        print(msg)
        if length == 0:
            errors.append(allele)

    print("\nDownloading C5-C11 (nucleotide CDS, Chevanne et al. 2010)...")
    for allele, acc in NUCLEOTIDE_ACCESSIONS.items():
        msg, length = download_allele(allele, acc, fetch_nucleotide_cds_aa)
        print(msg)
        if length == 0:
            errors.append(allele)

    print()
    if errors:
        print(f"Failed: {', '.join(errors)}")
    else:
        print(f"All 11 alleles written to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
