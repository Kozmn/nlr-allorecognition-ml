"""
sasa_utils.py — per-residue SASA and surface-exposure utilities.

Computes solvent-accessible surface area (Shrake-Rupley algorithm via
Bio.PDB.SASA) and derives per-chain surface masks. Relative SASA is
normalised against Tien et al. 2013 empirical maxima
(doi:10.1371/journal.pone.0080635).

Public API:
    compute_sasa(cif_path)       — per-residue SASA in the bound complex
    compute_sasa_apo(cif_path)   — per-residue SASA, each chain in isolation
    surface_mask(sasa_data, threshold=0.20)
    dump_sasa_cache / load_surface_mask
"""
from __future__ import annotations

import json
from pathlib import Path

# Bio.PDB is imported lazily inside compute_sasa / compute_sasa_apo so that
# downstream consumers that only need cached masks (validate_af3_contacts_sasa,
# refine_enrichment) can run without biopython installed.


# ═════════════════════════════════════════════════════════════════════════════
# MAX-ACCESSIBLE SURFACE AREA (Tien et al. 2013, empirical)
# ═════════════════════════════════════════════════════════════════════════════
# Ala-X-Ala tripeptide in a fully extended conformation — the "empirical"
# reference. Values in Å². Gly uses its actual SASA (it has no side chain).
# Reference: Tien MZ, Meyer AG, Sydykova DK, Spielman SJ, Wilke CO (2013).
#            Maximum Allowed Solvent Accessibilities of Residues in Proteins.
#            PLoS ONE 8(11):e80635.

MAX_ASA_TIEN_2013: dict[str, float] = {
    "ALA":  121.0, "ARG":  265.0, "ASN":  187.0, "ASP":  187.0,
    "CYS":  148.0, "GLN":  214.0, "GLU":  214.0, "GLY":   97.0,
    "HIS":  216.0, "ILE":  195.0, "LEU":  191.0, "LYS":  230.0,
    "MET":  203.0, "PHE":  228.0, "PRO":  154.0, "SER":  143.0,
    "THR":  163.0, "TRP":  264.0, "TYR":  255.0, "VAL":  165.0,
}

# One-letter ↔ three-letter convenience (used only for logging)
AA3_TO_AA1: dict[str, str] = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


# ═════════════════════════════════════════════════════════════════════════════
# CORE COMPUTATION
# ═════════════════════════════════════════════════════════════════════════════

def compute_sasa(cif_path: Path,
                 probe_radius: float = 1.4,
                 n_points: int = 100
                 ) -> dict[str, list[dict]]:
    """Compute per-residue SASA for every chain in a CIF file.

    Uses Bio.PDB.SASA.ShrakeRupley at residue level. The complex is scored
    as-is (both chains present), so interfacial residues have lower SASA
    than they would in isolation — this is the physiologically meaningful
    value for defining "surface" in the bound state.

    Args:
        cif_path: Path to an AlphaFold 3 .cif model file.
        probe_radius: Solvent probe radius in Å (default 1.4 = water).
        n_points: Number of Shrake-Rupley test points per atom
                  (default 100; trades accuracy for speed).

    Returns:
        {chain_id: [{"resnum": int, "aa": "ALA", "abs_sasa": float,
                     "rel_sasa": float}, ...]}
        where:
          - resnum is 1-indexed label_seq_id as parsed from CIF,
          - abs_sasa is Å² from Shrake-Rupley on the complex,
          - rel_sasa is abs_sasa / MAX_ASA_TIEN_2013[aa], clipped to [0, 1].

        Non-standard residues (not in MAX_ASA_TIEN_2013) get rel_sasa = None
        and are excluded from surface masks.
    """
    from Bio.PDB import MMCIFParser
    from Bio.PDB.SASA import ShrakeRupley
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(cif_path.stem, str(cif_path))

    # Bio.PDB's ShrakeRupley works in place: it attaches .sasa attributes
    # to atoms and residues.
    sr = ShrakeRupley(probe_radius=probe_radius, n_points=n_points)
    sr.compute(structure, level="R")  # residue level

    out: dict[str, list[dict]] = {}
    # AF3 produces a single model (model 0). Iterate that only.
    for model in structure:
        for chain in model:
            chain_id = chain.id
            entries: list[dict] = []
            for residue in chain:
                hetflag, resnum, _icode = residue.id
                if hetflag != " ":
                    # Skip HETATM records (waters, ligands). AF3 rarely emits
                    # these for protein-only jobs, but defend anyway.
                    continue
                aa3 = residue.get_resname().upper()
                abs_sasa = float(getattr(residue, "sasa", 0.0))
                max_asa = MAX_ASA_TIEN_2013.get(aa3)
                rel = (min(abs_sasa / max_asa, 1.0)
                       if max_asa and max_asa > 0 else None)
                entries.append({
                    "resnum": int(resnum),
                    "aa": aa3,
                    "abs_sasa": abs_sasa,
                    "rel_sasa": rel,
                })
            # Sort by residue number — Bio.PDB usually returns in order,
            # but defend against surprises.
            entries.sort(key=lambda e: e["resnum"])
            out[chain_id] = entries
        break  # AF3 single model; don't double-count
    return out


def compute_sasa_apo(cif_path: Path,
                     probe_radius: float = 1.4,
                     n_points: int = 100
                     ) -> dict[str, list[dict]]:
    """Compute per-residue SASA for every chain *in isolation* (apo state).

    For each chain in the CIF, removes all other chains from the structure
    and computes SASA on what remains. This gives the SASA the chain would
    have if it were never bound to anything — the relevant baseline for
    deciding whether a residue is "surface-exposed" in the unbound state.

    Critical for permutation tests of interchain contacts: residues that
    are buried in the complex BECAUSE of the partner (interfacial residues)
    are exposed in apo state. Using complex SASA for the surface mask
    incorrectly excludes them from the candidate pool, biasing the test.

    Returns: same shape as ``compute_sasa``, but ``rel_sasa`` reflects the
    chain alone, not the complex.
    """
    from Bio.PDB import MMCIFParser
    from Bio.PDB.SASA import ShrakeRupley
    parser = MMCIFParser(QUIET=True)
    structure_template = parser.get_structure(cif_path.stem, str(cif_path))

    # Take chain ids from model 0
    chain_ids: list[str] = []
    for model in structure_template:
        for chain in model:
            chain_ids.append(chain.id)
        break

    out: dict[str, list[dict]] = {}
    for target_chain_id in chain_ids:
        # Fresh parse so we don't mutate a structure already missing chains
        structure = parser.get_structure(cif_path.stem, str(cif_path))
        model = next(iter(structure))
        # Detach all chains except the target
        to_remove = [c.id for c in model if c.id != target_chain_id]
        for cid in to_remove:
            model.detach_child(cid)

        sr = ShrakeRupley(probe_radius=probe_radius, n_points=n_points)
        sr.compute(structure, level="R")

        entries: list[dict] = []
        for residue in model[target_chain_id]:
            hetflag, resnum, _icode = residue.id
            if hetflag != " ":
                continue
            aa3 = residue.get_resname().upper()
            abs_sasa = float(getattr(residue, "sasa", 0.0))
            max_asa = MAX_ASA_TIEN_2013.get(aa3)
            rel = (min(abs_sasa / max_asa, 1.0)
                   if max_asa and max_asa > 0 else None)
            entries.append({
                "resnum": int(resnum),
                "aa": aa3,
                "abs_sasa": abs_sasa,
                "rel_sasa": rel,
            })
        entries.sort(key=lambda e: e["resnum"])
        out[target_chain_id] = entries
    return out


def surface_mask(sasa_data: dict[str, list[dict]],
                 threshold: float = 0.20
                 ) -> dict[str, set[int]]:
    """Derive the set of surface-exposed residue numbers per chain.

    Residues with rel_sasa ≥ threshold are counted as surface. Residues
    with rel_sasa = None (non-standard AA) are skipped.

    Args:
        sasa_data: output of compute_sasa().
        threshold: minimum relative SASA. 0.20 is the canonical cutoff
                   (Miller et al. 1987; used in PDB's surface definition).

    Returns:
        {chain_id: {resnum, ...}} with 1-indexed positions.
    """
    out: dict[str, set[int]] = {}
    for chain_id, entries in sasa_data.items():
        surface: set[int] = set()
        for e in entries:
            if e["rel_sasa"] is not None and e["rel_sasa"] >= threshold:
                surface.add(e["resnum"])
        out[chain_id] = surface
    return out


# ═════════════════════════════════════════════════════════════════════════════
# ON-DISK CACHE
# ═════════════════════════════════════════════════════════════════════════════

SASA_DEFAULT_THRESHOLDS: tuple[float, ...] = (0.15, 0.20, 0.25)


def dump_sasa_cache(sasa_data: dict[str, list[dict]],
                    out_path: Path,
                    thresholds: tuple[float, ...] = SASA_DEFAULT_THRESHOLDS,
                    extra: dict | None = None,
                    sasa_data_apo: dict[str, list[dict]] | None = None,
                    ) -> None:
    """Serialize per-residue SASA and precomputed surface masks to JSON.

    Supports two SASA states:
      - "complex" (default, always written): SASA computed with both chains
        present. Stored under ``per_residue`` and masks under ``surface_masks``.
      - "apo" (optional): SASA of each chain in isolation. When provided,
        stored under ``per_residue_apo`` and masks under ``surface_masks_apo``.

    The JSON contains:
      - "per_residue": raw complex sasa_data,
      - "surface_masks": {str(threshold): {chain: [sorted resnums]}}  (complex),
      - "per_residue_apo": raw apo sasa_data  (only if provided),
      - "surface_masks_apo": {str(threshold): {chain: [sorted resnums]}}  (apo),
      - "metadata": free-form provenance.
    """
    payload: dict = {
        "per_residue": sasa_data,
        "surface_masks": {},
    }
    for thr in thresholds:
        mask = surface_mask(sasa_data, threshold=thr)
        payload["surface_masks"][f"{thr:.2f}"] = {
            ch: sorted(positions) for ch, positions in mask.items()
        }
    if sasa_data_apo is not None:
        payload["per_residue_apo"] = sasa_data_apo
        payload["surface_masks_apo"] = {}
        for thr in thresholds:
            mask = surface_mask(sasa_data_apo, threshold=thr)
            payload["surface_masks_apo"][f"{thr:.2f}"] = {
                ch: sorted(positions) for ch, positions in mask.items()
            }
    if extra:
        payload["metadata"] = extra
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)


def load_surface_mask(cache_path: Path, threshold: float = 0.20,
                      state: str = "complex"
                      ) -> dict[str, set[int]]:
    """Load a precomputed surface mask for a given threshold.

    Args:
        cache_path: per-job JSON written by ``dump_sasa_cache``.
        threshold: rel-SASA cutoff. If the exact threshold is not cached,
                   it is recomputed from the per-residue table.
        state: either ``"complex"`` (default; SASA in the bound state) or
               ``"apo"`` (SASA of each chain in isolation, required for
               an honest surface filter when testing interchain contacts).
               If apo data is not present in the cache a KeyError is raised.
    """
    with open(cache_path) as f:
        payload = json.load(f)
    key = f"{threshold:.2f}"
    if state == "complex":
        masks = payload.get("surface_masks", {})
        per_res = payload.get("per_residue", {})
    elif state == "apo":
        masks = payload.get("surface_masks_apo")
        per_res = payload.get("per_residue_apo")
        if masks is None or per_res is None:
            raise KeyError(
                f"Apo SASA not present in {cache_path}. "
                f"Re-run compute_sasa.py with --apo (or --force) to add it.")
    else:
        raise ValueError(f"state must be 'complex' or 'apo', got {state!r}")
    if key not in masks:
        return surface_mask(per_res, threshold=threshold)
    return {ch: set(positions) for ch, positions in masks[key].items()}
