"""
build_pair_features.py — builds the feature matrix for (NLR, HET-C) pairs.

Inputs:
  data/embeddings/esm_c_600m.npz              (per-sequence mean/max pool)
  data/embeddings/esm_c_600m.tsv              (sequence metadata)
  data/sequences/het_c/interaction_matrix_full.tsv  (label per phenotype × allele)
  data/sequences/het_ed_wd40/sequence_metadata.tsv  (seq_id → phenotype mapping)

For each of the 275 pairs (NLR_i, HETC_j) it generates:
  features = [ e_NLR ; e_HC ; |e_NLR - e_HC| ; e_NLR ⊙ e_HC ]   → 4 × 1152 = 4608
  label    = interaction_matrix[phenotype_of_NLR_i][HETC_j]     → 0 or 1

Generates two files — separately for mean-pool and max-pool:
  data/models/features_mean.npz
  data/models/features_max.npz

Each NPZ contains:
  X       : (275, 4608)   float32
  y       : (275,)        int8
  nlr_id  : (275,)        str    — NLR identifier
  hetc_id : (275,)        str    — HET-C identifier
  pheno   : (275,)        str    — NLR phenotype
  conf    : (275,)        str    — confirmed/inferred

Usage:
  cd thesis/
  python scripts/build_pair_features.py
  python scripts/build_pair_features.py --embeddings data/embeddings/esm_c_300m.npz

Author: Kacper Koźmin
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EMB = ROOT / "data" / "embeddings" / "esm_c_600m.npz"
INTERACTION_TSV = ROOT / "data" / "sequences" / "het_c" / "interaction_matrix_full.tsv"
METADATA_TSV = ROOT / "data" / "sequences" / "het_ed_wd40" / "sequence_metadata.tsv"
OUT_DIR = ROOT / "data" / "models"


def load_interaction_matrix() -> pd.DataFrame:
    """Return a DataFrame indexed by phenotype, columns C1..C11, values 0/1."""
    df = pd.read_csv(INTERACTION_TSV, sep="\t", index_col="phenotype")
    # Make sure the values are integers
    return df.astype(int)


def load_nlr_metadata() -> pd.DataFrame:
    """Metadata NLR: mapa header (z FASTA) → phenotype + confidence.
    Returns DataFrame z indeksem = seq_id (np. 'E1__PaYp__confirmed')."""
    df = pd.read_csv(METADATA_TSV, sep="\t")
    # seq_id reconstruct z kolumn phenotype, strain, confidence
    # Format FASTA: {phenotype}__{strain}__{confidence}
    df["seq_id"] = df["phenotype"] + "__" + df["strain"] + "__" + df["confidence"]
    return df.set_index("seq_id")[["phenotype", "confidence"]]


def build_features(e_nlr: np.ndarray, e_hc: np.ndarray) -> np.ndarray:
    """Sklada pair features for single pairs.
    [a; b; |a-b|; a*b] → (4d,)"""
    return np.concatenate([
        e_nlr,
        e_hc,
        np.abs(e_nlr - e_hc),
        e_nlr * e_hc,
    ]).astype(np.float32)


def build_matrix(embeddings: dict, pool: str,
                 interaction: pd.DataFrame,
                 nlr_meta: pd.DataFrame) -> dict:
    """Build (X, y, meta) for a single pooling type."""
    # Zbierz nazwy
    hetc_ids = sorted({k.split("_")[1] for k in embeddings.keys()
                       if k.startswith("hetc_") and k.endswith(f"_{pool}")},
                      key=lambda x: int(x[1:]))  # C1, C2, ..., C10, C11
    nlr_ids = sorted({"_".join(k.split("_")[1:-1]) for k in embeddings.keys()
                      if k.startswith("nlr_") and k.endswith(f"_{pool}")})

    # Cache per sequence
    hc_emb = {h: embeddings[f"hetc_{h}_{pool}"] for h in hetc_ids}
    nl_emb = {n: embeddings[f"nlr_{n}_{pool}"] for n in nlr_ids}

    n_pairs = len(nlr_ids) * len(hetc_ids)
    d = next(iter(hc_emb.values())).shape[0]
    X = np.zeros((n_pairs, 4 * d), dtype=np.float32)
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
            X[idx] = build_features(nl_emb[nlr_id], hc_emb[hc_id])
            y[idx] = label
            rows.append({
                "nlr_id": nlr_id, "hetc_id": hc_id,
                "phenotype": phenotype, "confidence": confidence,
                "label": label,
            })
            idx += 1

    meta = pd.DataFrame(rows)
    return {"X": X, "y": y, "meta": meta, "hetc_ids": hetc_ids, "nlr_ids": nlr_ids}


def save_pack(pack: dict, path: Path):
    """NPZ: X, y, plus metadata columns as separate arrays.
    Strings are stored as fixed-width unicode (dtype='<U...'), not as
    object arrays — so they can be loaded with allow_pickle=False."""
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = pack["meta"]
    np.savez(
        path,
        X=pack["X"],
        y=pack["y"],
        nlr_id=np.asarray(meta["nlr_id"].tolist(), dtype=np.str_),
        hetc_id=np.asarray(meta["hetc_id"].tolist(), dtype=np.str_),
        phenotype=np.asarray(meta["phenotype"].tolist(), dtype=np.str_),
        confidence=np.asarray(meta["confidence"].tolist(), dtype=np.str_),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", default=str(DEFAULT_EMB))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    emb_path = Path(args.embeddings)
    out_dir = Path(args.out_dir)
    if not emb_path.exists():
        raise FileNotFoundError(
            f"Missing {emb_path}. Run first: python scripts/compute_embeddings.py")

    print(f"Embeddings: {emb_path}")
    npz = np.load(emb_path, allow_pickle=False)
    embeddings = {k: npz[k] for k in npz.files}
    npz.close()
    print(f"  wczytano {len(embeddings)} tablic")

    interaction = load_interaction_matrix()
    print(f"Interaction matrix: {interaction.shape} "
          f"({interaction.index.tolist()} × {interaction.columns.tolist()})")

    nlr_meta = load_nlr_metadata()
    print(f"NLR metadata: {len(nlr_meta)} sequence")

    for pool in ("mean", "max"):
        print(f"\n=== Building features for '{pool}' pool ===")
        pack = build_matrix(embeddings, pool, interaction, nlr_meta)
        pos = int((pack["y"] == 1).sum())
        neg = int((pack["y"] == 0).sum())
        print(f"  X shape: {pack['X'].shape}")
        print(f"  labels:  {pos} pozytywnych, {neg} negatywnych ({pos/(pos+neg)*100:.1f}% +)")
        # Class balance per phenotype
        print(f"  rozklad per phenotype:")
        for ph, grp in pack["meta"].groupby("phenotype"):
            p = int((grp["label"] == 1).sum())
            n = len(grp)
            print(f"    {ph:<5s}  n={n:>3d}  pozytywow={p:>2d}  ({p/n*100:.0f}%)")

        out = out_dir / f"features_{pool}.npz"
        save_pack(pack, out)
        print(f"  → {out.relative_to(ROOT)}")

    # A single CSV with pair metadata (useful for debugging)
    meta_csv = out_dir / "pair_index.csv"
    pack["meta"].to_csv(meta_csv, index=False)
    print(f"\nPair index: {meta_csv.relative_to(ROOT)}")
    print("Done.")


if __name__ == "__main__":
    main()
