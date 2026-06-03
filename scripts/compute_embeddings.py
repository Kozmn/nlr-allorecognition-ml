"""
compute_embeddings.py — ESM-C 600M per-sequence embeddings for NLR WD40
and HET-C sequences.

For each input sequence the ESM-C 600M model produces a per-residue
matrix (L × 1152), reduced to one vector per sequence by mean pooling
and max pooling.

Input:
  data/sequences/het_ed_wd40/het_ed_sequences.fasta   (25 NLR WD40)
  data/sequences/het_c/HET-C_C*.fasta                 (11 HET-C alleles)

Output:
  data/embeddings/esm_c_600m.npz       per-sequence mean/max vectors
  data/embeddings/sequence_index.tsv   metadata table

Requires the torch and esm packages. The first run downloads the model
weights (~2.4 GB) to the local cache.

Usage:
  python scripts/compute_embeddings.py
  python scripts/compute_embeddings.py --device cpu
  python scripts/compute_embeddings.py --model esmc_300m
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
HETC_DIR = ROOT / "data" / "sequences" / "het_c"
NLR_FASTA = ROOT / "data" / "sequences" / "het_ed_wd40" / "het_ed_sequences.fasta"
METADATA_TSV = ROOT / "data" / "sequences" / "het_ed_wd40" / "sequence_metadata.tsv"
OUT_DIR = ROOT / "data" / "embeddings"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── FASTA parsing (no biopython — to keep dependencies minimal) ─────────────

def parse_fasta(path: Path) -> list[tuple[str, str]]:
    """Return a list of (header, sequence). Header without '>' or whitespace."""
    records = []
    header, chunks = None, []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(chunks)))
                header = line[1:].split()[0]  # first word after '>'
                chunks = []
            elif line:
                chunks.append(line.strip())
    if header is not None:
        records.append((header, "".join(chunks)))
    return records


def collect_hetc_sequences() -> list[tuple[str, str]]:
    """Load all HET-C_C*.fasta → [(allele, seq), ...]."""
    recs = []
    for p in sorted(HETC_DIR.glob("HET-C_C*.fasta")):
        parsed = parse_fasta(p)
        if len(parsed) != 1:
            raise RuntimeError(f"{p.name}: expected 1 sequence, got {len(parsed)}")
        header, seq = parsed[0]
        # header has format "C1_HET-C" — extract only "C1"
        allele = header.split("_")[0]
        recs.append((allele, seq))
    return recs


def collect_nlr_sequences() -> list[tuple[str, str]]:
    """Load het_ed_sequences.fasta → [(seq_id, seq), ...].
    seq_id is e.g. 'E1__PaYp__confirmed' — we keep the full identifier."""
    return parse_fasta(NLR_FASTA)


# ── Model loading ───────────────────────────────────────────────────────────

def choose_device(requested: str) -> str:
    """auto: MPS if dostepne, inaczej CUDA, inaczej CPU."""
    import torch
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_esm_c(model_name: str, device: str):
    """Laduje model ESM-C. Requires paczki `esm` (v3+, z EvolutionaryScale)."""
    try:
        from esm.models.esmc import ESMC
    except ImportError as e:
        raise ImportError(
            "Potrzebna paczka `esm` (v3+). Instalacja:\n"
            "    pip install esm\n"
            "Albo z repo Meta/EvolutionaryScale if pip not ma najnowszej."
        ) from e
    print(f"Loading {model_name} na {device}... (pierwsze uruchomienie downloads ~2.4 GB)")
    t0 = time.time()
    model = ESMC.from_pretrained(model_name).to(device)
    model.eval()
    print(f"  model gotowy w {time.time() - t0:.1f}s")
    return model


# ── Embedding extraction ─────────────────────────────────────────────────────

def embed_sequence(model, sequence: str, device: str) -> np.ndarray:
    """Returns matrix (L, d) — embedding per-residuum, bez tokenow specjalnych.
    d = 1152 for 600M, 960 for 300M, 2560 for 6B."""
    import torch
    from esm.sdk.api import ESMProtein, LogitsConfig

    protein = ESMProtein(sequence=sequence)
    tokens = model.encode(protein)
    with torch.no_grad():
        out = model.logits(
            tokens,
            LogitsConfig(sequence=True, return_embeddings=True),
        )
    emb = out.embeddings.squeeze(0).cpu().numpy()  # (L+2, d) — z BOS/EOS
    # Trim tokeny BOS i EOS — we keep only L amino acids
    return emb[1:-1, :].astype(np.float32)


def pool(emb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (mean_pool, max_pool) — oba (d,)."""
    return emb.mean(axis=0), emb.max(axis=0)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="esmc_600m",
                    choices=["esmc_300m", "esmc_600m", "esmc_6b"])
    ap.add_argument("--device", default="auto",
                    choices=["auto", "cpu", "mps", "cuda"])
    ap.add_argument("--out", default=str(OUT_DIR / "esm_c_600m.npz"),
                    help="Sciezka wyjsciowa NPZ. Dla modelu 300m dopisz '_300m'.")
    args = ap.parse_args()

    print(f"Root dir: {ROOT}")
    print(f"HET-C dir: {HETC_DIR}")
    print(f"NLR fasta: {NLR_FASTA}")
    print(f"Output:    {args.out}\n")

    # ── Load sequences ──────────────────────────────────────────────────
    hetc_records = collect_hetc_sequences()
    nlr_records = collect_nlr_sequences()
    print(f"HET-C alleles: {len(hetc_records)}")
    for name, seq in hetc_records:
        print(f"  {name:>4s}  len={len(seq):>4d}")
    print(f"\nNLR WD40 sequence: {len(nlr_records)}")
    for name, seq in nlr_records:
        print(f"  {name:<30s}  len={len(seq):>4d}")

    # ── Zaladuj model ───────────────────────────────────────────────────────
    device = choose_device(args.device)
    print(f"\nDevice: {device}")
    model = load_esm_c(args.model, device)

    # ── Embeddingi ──────────────────────────────────────────────────────────
    arrays: dict[str, np.ndarray] = {}
    index_rows: list[dict] = []

    all_records = [("hetc", n, s) for n, s in hetc_records] + \
                  [("nlr",  n, s) for n, s in nlr_records]

    for i, (kind, name, seq) in enumerate(all_records, 1):
        t0 = time.time()
        emb = embed_sequence(model, seq, device)
        mean_vec, max_vec = pool(emb)
        arrays[f"{kind}_{name}_mean"] = mean_vec
        arrays[f"{kind}_{name}_max"]  = max_vec
        index_rows.append({
            "kind": kind, "name": name, "aa_len": len(seq),
            "emb_dim": int(emb.shape[1]),
            "sec_per_seq": round(time.time() - t0, 2),
        })
        print(f"  [{i:>2d}/{len(all_records)}] {kind} {name:<30s} "
              f"L={len(seq):>4d}  t={time.time()-t0:.1f}s")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_npz = Path(args.out)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_npz, **arrays)
    print(f"\nZapisano {len(arrays)} tablic do {out_npz.relative_to(ROOT)}")

    # Index TSV
    index_tsv = out_npz.with_suffix(".tsv")
    with open(index_tsv, "w") as f:
        f.write("kind\tname\taa_len\temb_dim\tsec_per_seq\n")
        for r in index_rows:
            f.write(f"{r['kind']}\t{r['name']}\t{r['aa_len']}\t"
                    f"{r['emb_dim']}\t{r['sec_per_seq']}\n")
    print(f"Index: {index_tsv.relative_to(ROOT)}")
    print("Done.")


if __name__ == "__main__":
    main()
