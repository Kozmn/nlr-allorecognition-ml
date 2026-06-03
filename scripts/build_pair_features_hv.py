"""
build_pair_features_hv.py — Pair features at hypervariable positions (HV).

A variant of `build_pair_features.py` that, instead of mean+max over the
whole ESM-C sequence, pools ONLY over the hypervariable positions:

  NLR:    {10, 11, 12, 14, 30, 32, 39} INTRA-REPEAT
          → all copies of these positions in every detected
            WD40 repeat (find_repeat_boundaries from cif_utils.py)
  HET-C:  {118, 133, 153} (positions 1-indexed in the full sequence)

Hypothesis: the entire compatibility signal resides in these ~50 (NLR) / 3
(HET-C) positions. Mean/max over the whole sequence dilutes it by averaging
with hundreds of "background" positions.

The pair-feature dimensionality stays 4608 (the same as whole-seq) — an A/B
test against the existing Model 1.

Inputs:
  data/embeddings/esm_c_600m_per_residue.npz       (from compute_embeddings_per_residue.py)
  data/sequences/het_c/interaction_matrix_full.tsv
  data/sequences/het_ed_wd40/sequence_metadata.tsv

Outputs:
  data/models/features_hv_mean.npz
  data/models/features_hv_max.npz
  data/models/pair_index_hv.csv
  data/models/hv_extraction_report.tsv  — per-sequence diagnostics:
       how many repeats were detected, how many HV positions found, any gaps

Usage:
  cd thesis/
  python scripts/build_pair_features_hv.py

Author: Kacper Koźmin
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EMB = ROOT / "data" / "embeddings" / "esm_c_600m_per_residue.npz"
INTERACTION_TSV = ROOT / "data" / "sequences" / "het_c" / "interaction_matrix_full.tsv"
METADATA_TSV = ROOT / "data" / "sequences" / "het_ed_wd40" / "sequence_metadata.tsv"
OUT_DIR = ROOT / "data" / "models"

# Re-use constants from the shared module — guarantees consistency with AF3 validation
sys.path.insert(0, str(Path(__file__).parent))
from constants import (  # noqa: E402
    NLR_REPEAT_HYPERVARIABLE,
    HETC_HYPERVARIABLE,
    REPEAT_LENGTH,
)
from cif_utils import find_repeat_boundaries  # noqa: E402


# ── Helpers for extracting HV embeddings ─────────────────────────────────────

def hv_positions_nlr(nlr_seq: str) -> list[int]:
    """Return 1-indexed hypervariable positions in the FULL NLR sequence.

    For each detected WD40 repeat (TLEGH motif) and each intra-repeat
    position in NLR_REPEAT_HYPERVARIABLE = {10,11,12,14,30,32,39}, returns
    the corresponding absolute position in the sequence.

    Skips positions that fall beyond the sequence length (e.g. the last
    short repeat).
    """
    bounds = find_repeat_boundaries(nlr_seq)
    if not bounds:
        return []
    L = len(nlr_seq)
    out: list[int] = []
    for (start, _end) in bounds:  # start and end are 1-indexed
        for intra in sorted(NLR_REPEAT_HYPERVARIABLE):
            abs_pos = start + intra - 1     # intra=1 → start itself
            if 1 <= abs_pos <= L:
                out.append(abs_pos)
    return out


def hv_positions_hetc(hc_seq: str) -> list[int]:
    """1-indexed positions in the HET-C sequence (already absolute)."""
    L = len(hc_seq)
    return [p for p in sorted(HETC_HYPERVARIABLE) if 1 <= p <= L]


def pool_hv(emb_per_residue: np.ndarray, positions_1based: list[int]
            ) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean, max) over the selected positions.

    emb_per_residue: (L, d)
    positions_1based: list of 1-indexed positions to select
    """
    if not positions_1based:
        d = emb_per_residue.shape[1]
        return np.zeros(d, dtype=np.float32), np.zeros(d, dtype=np.float32)
    idx0 = np.array([p - 1 for p in positions_1based], dtype=np.int64)
    sub = emb_per_residue[idx0, :]   # (k, d)
    return sub.mean(axis=0).astype(np.float32), sub.max(axis=0).astype(np.float32)


# ── Wczytywanie data ───────────────────────────────────────────────────────

def load_interaction_matrix() -> pd.DataFrame:
    df = pd.read_csv(INTERACTION_TSV, sep="\t", index_col="phenotype")
    return df.astype(int)


def load_nlr_metadata() -> pd.DataFrame:
    df = pd.read_csv(METADATA_TSV, sep="\t")
    df["seq_id"] = df["phenotype"] + "__" + df["strain"] + "__" + df["confidence"]
    return df.set_index("seq_id")[["phenotype", "confidence"]]


def load_per_residue_embeddings(path: Path) -> dict[str, np.ndarray]:
    """Returns {key: (L, d)} bez tablic _meta_*."""
    npz = np.load(path, allow_pickle=False)
    out = {k: npz[k] for k in npz.files if not k.startswith("_meta_")}
    npz.close()
    return out


def load_sequences() -> tuple[dict[str, str], dict[str, str]]:
    """Load sequences from FASTA — needed for find_repeat_boundaries.
    Returns (hetc_seqs, nlr_seqs) as name→seq maps."""
    hetc_dir = ROOT / "data" / "sequences" / "het_c"
    nlr_fasta = ROOT / "data" / "sequences" / "het_ed_wd40" / "het_ed_sequences.fasta"

    def parse_fasta(p: Path) -> list[tuple[str, str]]:
        records, header, chunks = [], None, []
        with open(p) as fh:
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

    hetc_seqs: dict[str, str] = {}
    for p in sorted(hetc_dir.glob("HET-C_C*.fasta")):
        for h, s in parse_fasta(p):
            allele = h.split("_")[0]
            hetc_seqs[allele] = s

    nlr_seqs: dict[str, str] = {h: s for h, s in parse_fasta(nlr_fasta)}
    return hetc_seqs, nlr_seqs


# ── Budowa pairs ───────────────────────────────────────────────────────────────

def build_features_for_pair(nlr_pool: np.ndarray, hc_pool: np.ndarray) -> np.ndarray:
    """Exactly ten sam schemat co w build_pair_features.py:
    [a; b; |a-b|; a*b] → (4d,)"""
    return np.concatenate([
        nlr_pool,
        hc_pool,
        np.abs(nlr_pool - hc_pool),
        nlr_pool * hc_pool,
    ]).astype(np.float32)


def build_matrix(emb_per_res: dict, hetc_seqs: dict, nlr_seqs: dict,
                 interaction: pd.DataFrame, nlr_meta: pd.DataFrame
                 ) -> tuple[dict, pd.DataFrame]:
    """Build (X_mean, y, meta) and (X_max, y, meta) simultaneously, in a single
    pass over the per-residue embeddings. Also returns a diagnostic report."""
    # All HET-C and NLR keys from the embeddings
    hetc_keys = sorted(k for k in emb_per_res if k.startswith("hetc_"))
    nlr_keys  = sorted(k for k in emb_per_res if k.startswith("nlr_"))

    # name → key
    hetc_name_to_key = {k.removeprefix("hetc_"): k for k in hetc_keys}
    nlr_name_to_key  = {k.removeprefix("nlr_"):  k for k in nlr_keys}

    hetc_ids = sorted(hetc_name_to_key.keys(), key=lambda x: int(x[1:]))
    nlr_ids  = sorted(nlr_name_to_key.keys())

    # ── Pre-compute HV pools per sequence ────────────────────────────────────
    nlr_pool_mean: dict[str, np.ndarray] = {}
    nlr_pool_max:  dict[str, np.ndarray] = {}
    diag_rows: list[dict] = []

    print("=== NLR — ekstrakcja embeddingow HV ===")
    print(f"{'NLR':<35s}  {'L':>4s}  {'reps':>4s}  {'HV pos':>6s}")
    for nm in nlr_ids:
        if nm not in nlr_seqs:
            raise KeyError(f"NLR '{nm}' not ma sequence w FASTA")
        seq = nlr_seqs[nm]
        bounds = find_repeat_boundaries(seq)
        positions = hv_positions_nlr(seq)
        emb = emb_per_res[nlr_name_to_key[nm]]
        if emb.shape[0] != len(seq):
            print(f"  WARN: {nm} emb len {emb.shape[0]} != seq len {len(seq)}")

        m, mx = pool_hv(emb, positions)
        nlr_pool_mean[nm] = m
        nlr_pool_max[nm]  = mx
        print(f"  {nm:<35s}  {len(seq):>4d}  {len(bounds):>4d}  {len(positions):>6d}")
        diag_rows.append({
            "kind": "nlr", "name": nm, "aa_len": len(seq),
            "n_repeats": len(bounds), "n_hv_positions": len(positions),
        })

    hetc_pool_mean: dict[str, np.ndarray] = {}
    hetc_pool_max:  dict[str, np.ndarray] = {}
    print("\n=== HET-C — ekstrakcja embeddingow HV ===")
    print(f"{'HET-C':<6s}  {'L':>4s}  {'HV pos':>6s}")
    for nm in hetc_ids:
        if nm not in hetc_seqs:
            raise KeyError(f"HET-C '{nm}' not ma sequence w FASTA")
        seq = hetc_seqs[nm]
        positions = hv_positions_hetc(seq)
        emb = emb_per_res[hetc_name_to_key[nm]]
        if emb.shape[0] != len(seq):
            print(f"  WARN: {nm} emb len {emb.shape[0]} != seq len {len(seq)}")

        m, mx = pool_hv(emb, positions)
        hetc_pool_mean[nm] = m
        hetc_pool_max[nm]  = mx
        print(f"  {nm:<6s}  {len(seq):>4d}  {len(positions):>6d}")
        diag_rows.append({
            "kind": "hetc", "name": nm, "aa_len": len(seq),
            "n_repeats": np.nan, "n_hv_positions": len(positions),
        })

    # ── Zbuduj macierze X for mean i max ──────────────────────────────────────
    n_pairs = len(nlr_ids) * len(hetc_ids)
    d = next(iter(nlr_pool_mean.values())).shape[0]
    X_mean = np.zeros((n_pairs, 4 * d), dtype=np.float32)
    X_max  = np.zeros((n_pairs, 4 * d), dtype=np.float32)
    y = np.zeros(n_pairs, dtype=np.int8)
    rows = []

    idx = 0
    for nlr_id in nlr_ids:
        if nlr_id not in nlr_meta.index:
            raise KeyError(
                f"NLR '{nlr_id}' has no entry in sequence_metadata.tsv. "
                f"Check that the FASTA headers match seq_id in the metadata.")
        phenotype = nlr_meta.loc[nlr_id, "phenotype"]
        confidence = nlr_meta.loc[nlr_id, "confidence"]
        if phenotype not in interaction.index:
            raise KeyError(
                f"Phenotype '{phenotype}' (NLR {nlr_id}) has no row "
                f"in interaction_matrix_full.tsv")

        for hc_id in hetc_ids:
            label = int(interaction.loc[phenotype, hc_id])
            X_mean[idx] = build_features_for_pair(
                nlr_pool_mean[nlr_id], hetc_pool_mean[hc_id])
            X_max[idx]  = build_features_for_pair(
                nlr_pool_max[nlr_id],  hetc_pool_max[hc_id])
            y[idx] = label
            rows.append({
                "nlr_id": nlr_id, "hetc_id": hc_id,
                "phenotype": phenotype, "confidence": confidence,
                "label": label,
            })
            idx += 1

    meta = pd.DataFrame(rows)
    diag = pd.DataFrame(diag_rows)

    return ({"X_mean": X_mean, "X_max": X_max, "y": y, "meta": meta,
             "hetc_ids": hetc_ids, "nlr_ids": nlr_ids},
            diag)


def save_pack(X: np.ndarray, y: np.ndarray, meta: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        X=X, y=y,
        nlr_id=np.asarray(meta["nlr_id"].tolist(), dtype=np.str_),
        hetc_id=np.asarray(meta["hetc_id"].tolist(), dtype=np.str_),
        phenotype=np.asarray(meta["phenotype"].tolist(), dtype=np.str_),
        confidence=np.asarray(meta["confidence"].tolist(), dtype=np.str_),
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", default=str(DEFAULT_EMB),
                    help="NPZ z per-residue embeddingami")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    emb_path = Path(args.embeddings)
    out_dir = Path(args.out_dir)
    if not emb_path.exists():
        raise FileNotFoundError(
            f"Missing {emb_path}. Run first: "
            f"python scripts/compute_embeddings_per_residue.py")

    print(f"Per-residue embeddings: {emb_path}")
    emb_per_res = load_per_residue_embeddings(emb_path)
    print(f"  wczytano {len(emb_per_res)} sequence "
          f"(d = {next(iter(emb_per_res.values())).shape[1]})\n")

    hetc_seqs, nlr_seqs = load_sequences()
    print(f"FASTA sequence: {len(hetc_seqs)} HET-C, {len(nlr_seqs)} NLR\n")

    interaction = load_interaction_matrix()
    print(f"Interaction matrix: {interaction.shape}")

    nlr_meta = load_nlr_metadata()
    print(f"NLR metadata: {len(nlr_meta)} sequence\n")

    pack, diag = build_matrix(
        emb_per_res, hetc_seqs, nlr_seqs, interaction, nlr_meta)

    pos = int((pack["y"] == 1).sum())
    neg = int((pack["y"] == 0).sum())
    print(f"\n=== Result ===")
    print(f"  pairs: {len(pack['y'])}  ({pos} pozytywnych, {neg} negatywnych)")
    print(f"  X_mean: {pack['X_mean'].shape}")
    print(f"  X_max:  {pack['X_max'].shape}")

    # Save features_hv_{mean,max}.npz in the same format as
    # features_{mean,max}.npz — so train_lono.py works on it
    # without modification.
    out_mean = out_dir / "features_hv_mean.npz"
    out_max  = out_dir / "features_hv_max.npz"
    save_pack(pack["X_mean"], pack["y"], pack["meta"], out_mean)
    save_pack(pack["X_max"],  pack["y"], pack["meta"], out_max)
    print(f"\n  → {out_mean.relative_to(ROOT)}")
    print(f"  → {out_max.relative_to(ROOT)}")

    # Pair index (wspolny for mean i max)
    pair_idx = out_dir / "pair_index_hv.csv"
    pack["meta"].to_csv(pair_idx, index=False)
    print(f"  → {pair_idx.relative_to(ROOT)}")

    # HV extraction diagnostics (per sequence)
    diag_path = out_dir / "hv_extraction_report.tsv"
    diag.to_csv(diag_path, sep="\t", index=False)
    print(f"  → {diag_path.relative_to(ROOT)}  (HV diagnostics: repeats found, positions found)")

    print("\nDone. Then run (z --out-dir zeby NIE overwrite whole-seq results):")
    print("  mkdir -p data/models_hv")
    print("  python scripts/train_lono.py \\")
    print("      --features-mean data/models/features_hv_mean.npz \\")
    print("      --features-max  data/models/features_hv_max.npz \\")
    print("      --out-dir       data/models_hv")


if __name__ == "__main__":
    main()
