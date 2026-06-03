"""
compute_embeddings_per_residue.py — ESM-C 600M per-residue embeddings.

A variant of `compute_embeddings.py` that does NOT pool. For each sequence
it keeps the full matrix (L × d) — needed to extract embeddings at specific
positions (e.g. the hypervariable {10,11,12,14,30,32,39} in NLR WD40 repeats
and {118,133,153} in HET-C).

Reads:
  - data/sequences/het_ed_wd40/het_ed_sequences.fasta   (25 NLR WD40)
  - data/sequences/het_c/HET-C_C*.fasta                 (11 HET-C alleles)

Writes:
  - data/embeddings/esm_c_600m_per_residue.npz
       hetc_{allele}              (L, 1152)
       nlr_{seq_id}               (L, 1152)
       _meta_keys                 (N,)  array of key names
       _meta_kinds                (N,)  'hetc' or 'nlr'
       _meta_names                (N,)  sequence name
       _meta_lengths              (N,)  length L
  - data/embeddings/per_residue_index.tsv  (human-readable index)

Sanity-check: the mean over axis 0 for a given sequence should be identical
to the corresponding `*_mean` in `esm_c_600m.npz` (up to floating-point error).

Requires the torch and esm packages.

Usage:
  python scripts/compute_embeddings_per_residue.py
  python scripts/compute_embeddings_per_residue.py --device cpu
  python scripts/compute_embeddings_per_residue.py --model esmc_300m
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
HETC_DIR = ROOT / "data" / "sequences" / "het_c"
NLR_FASTA = ROOT / "data" / "sequences" / "het_ed_wd40" / "het_ed_sequences.fasta"
OUT_DIR = ROOT / "data" / "embeddings"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── FASTA parsing (no biopython) ─────────────────────────────────────────────

def parse_fasta(path: Path) -> list[tuple[str, str]]:
    records = []
    header, chunks = None, []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(chunks)))
                header = line[1:].split()[0]
                chunks = []
            elif line:
                chunks.append(line.strip())
    if header is not None:
        records.append((header, "".join(chunks)))
    return records


def collect_hetc_sequences() -> list[tuple[str, str]]:
    recs = []
    for p in sorted(HETC_DIR.glob("HET-C_C*.fasta")):
        parsed = parse_fasta(p)
        if len(parsed) != 1:
            raise RuntimeError(f"{p.name}: expected 1 sequence, got {len(parsed)}")
        header, seq = parsed[0]
        allele = header.split("_")[0]  # 'C1_HET-C' → 'C1'
        recs.append((allele, seq))
    return recs


def collect_nlr_sequences() -> list[tuple[str, str]]:
    return parse_fasta(NLR_FASTA)


# ── Model ─────────────────────────────────────────────────────────────────────

def choose_device(requested: str) -> str:
    import torch
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_esm_c(model_name: str, device: str):
    try:
        from esm.models.esmc import ESMC
    except ImportError as e:
        raise ImportError(
            "Potrzebna paczka `esm` (v3+). Zainstaluj: pip install esm"
        ) from e
    print(f"Loading {model_name} na {device}... (first time downloads ~2.4 GB)")
    t0 = time.time()
    model = ESMC.from_pretrained(model_name).to(device)
    model.eval()
    print(f"  model gotowy w {time.time() - t0:.1f}s")
    return model


def embed_sequence(model, sequence: str, device: str) -> np.ndarray:
    """Returns matrix (L, d) per-residue, bez BOS/EOS."""
    import torch
    from esm.sdk.api import ESMProtein, LogitsConfig

    protein = ESMProtein(sequence=sequence)
    tokens = model.encode(protein)
    with torch.no_grad():
        out = model.logits(
            tokens,
            LogitsConfig(sequence=True, return_embeddings=True),
        )
    emb = out.embeddings.squeeze(0).cpu().numpy()  # (L+2, d)
    return emb[1:-1, :].astype(np.float32)         # trim BOS/EOS


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="esmc_600m",
                    choices=["esmc_300m", "esmc_600m", "esmc_6b"])
    ap.add_argument("--device", default="auto",
                    choices=["auto", "cpu", "mps", "cuda"])
    ap.add_argument("--out", default=str(OUT_DIR / "esm_c_600m_per_residue.npz"))
    args = ap.parse_args()

    print(f"Root dir: {ROOT}")
    print(f"Output:   {args.out}\n")

    hetc_records = collect_hetc_sequences()
    nlr_records = collect_nlr_sequences()
    print(f"HET-C alleles: {len(hetc_records)}")
    for name, seq in hetc_records:
        print(f"  {name:>4s}  len={len(seq):>4d}")
    print(f"\nNLR WD40 sequence: {len(nlr_records)}")
    for name, seq in nlr_records:
        print(f"  {name:<30s}  len={len(seq):>4d}")

    device = choose_device(args.device)
    print(f"\nDevice: {device}")
    model = load_esm_c(args.model, device)

    arrays: dict[str, np.ndarray] = {}
    keys, kinds, names, lengths = [], [], [], []

    all_records = [("hetc", n, s) for n, s in hetc_records] + \
                  [("nlr",  n, s) for n, s in nlr_records]

    print(f"\n=== Embedding {len(all_records)} sequences ===")
    for i, (kind, name, seq) in enumerate(all_records, 1):
        t0 = time.time()
        emb = embed_sequence(model, seq, device)  # (L, d)
        if emb.shape[0] != len(seq):
            print(f"  WARN: {kind}_{name}: emb len {emb.shape[0]} != seq len {len(seq)}")

        key = f"{kind}_{name}"
        arrays[key] = emb
        keys.append(key)
        kinds.append(kind)
        names.append(name)
        lengths.append(emb.shape[0])

        print(f"  [{i:>2d}/{len(all_records)}] {key:<35s}  "
              f"shape={emb.shape}  ({time.time()-t0:.1f}s)")

    # ── Save ──────────────────────────────────────────────────────────────────
    arrays["_meta_keys"]    = np.array(keys,    dtype=np.str_)
    arrays["_meta_kinds"]   = np.array(kinds,   dtype=np.str_)
    arrays["_meta_names"]   = np.array(names,   dtype=np.str_)
    arrays["_meta_lengths"] = np.array(lengths, dtype=np.int32)

    print(f"\nZapisuje do {args.out}...")
    np.savez_compressed(args.out, **arrays)
    out_path = Path(args.out)
    print(f"  rozmiar: {out_path.stat().st_size / 1e6:.1f} MB")

    # Plik indeksu w TSV (do podgladu)
    idx_path = OUT_DIR / "per_residue_index.tsv"
    with open(idx_path, "w") as fh:
        fh.write("kind\tname\tkey\taa_len\tdim\n")
        d = arrays[keys[0]].shape[1]
        for k, kd, nm, L in zip(keys, kinds, names, lengths):
            fh.write(f"{kd}\t{nm}\t{k}\t{L}\t{d}\n")
    print(f"  index: {idx_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
