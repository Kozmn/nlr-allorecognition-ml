"""
cif_utils.py — Shared CIF parsing, geometry, and WD40 repeat utilities.

Centralises all CIF file handling so that the three scripts that read
structural data (check_propellers, collect_selected, extract_features)
use identical parsing logic. Before this module existed, each script had
its own CIF parser — subtle differences could silently produce different
residue sets from the same file.

Functions:

  CIF parsing:
    parse_ca_coords(cif_path)           Cα (x, y, z) per chain
    parse_ca_plddt(cif_path)            Cα B-factor (= pLDDT in AF3) per chain

  Geometry:
    centre_of_mass(coords)              CoM of Cα coordinate list
    radius_of_gyration(coords)          Rg of Cα coordinate list
    euclidean_distance(a, b)            distance between two 3D points

  Model selection:
    hetc_centrality(cif_path, split_at) HET-C proximity metric (lower = better)
    find_all_cifs(job_dir)              enumerate seed×sample CIFs in a job folder

  WD40 repeats:
    find_repeat_boundaries(nlr_seq)     detect repeat start/end using TLEGH
    classify_nlr_residue(res, bounds)   check if residue is on a hypervariable position

Author: Kacper Koźmin
"""

from __future__ import annotations

import math
import re
from pathlib import Path

from constants import (
    PROPELLER_LENGTH,
    NLR_REPEAT_HYPERVARIABLE,
    TLEGH_RE,
    REPEAT_LENGTH,
)


# ═════════════════════════════════════════════════════════════════════════════
# CIF PARSING
# ═════════════════════════════════════════════════════════════════════════════

def _parse_atom_site_columns(lines: list[str]) -> tuple[dict[str, int], int]:
    """Find the _atom_site loop and return (column_map, first_data_line_index).

    Scans forward from the loop_ header through all _atom_site.* column
    definitions, returning the column name → index mapping and the line
    number where actual ATOM/HETATM data starts.
    """
    col_map: dict[str, int] = {}
    col_idx = 0
    i = 0
    # Find the loop_ that precedes _atom_site columns
    while i < len(lines):
        if lines[i].strip() == "loop_":
            j = i + 1
            while j < len(lines) and lines[j].strip().startswith("_atom_site."):
                col_map[lines[j].strip()] = col_idx
                col_idx += 1
                j += 1
            if col_map:
                return col_map, j
        i += 1
    return {}, len(lines)


def parse_ca_coords(cif_path: Path) -> dict[str, list[tuple[int, float, float, float]]]:
    """Extract Cα coordinates per chain from a CIF file.

    Returns {chain_id: [(resnum, x, y, z), ...]} sorted by residue number.
    Only ATOM records with atom name CA and a numeric label_seq_id are kept.
    """
    with open(cif_path) as f:
        lines = f.readlines()

    col_map, start = _parse_atom_site_columns(lines)
    if not col_map:
        return {}

    grp_col = col_map.get("_atom_site.group_PDB")
    atm_col = col_map.get("_atom_site.label_atom_id")
    chn_col = col_map.get("_atom_site.label_asym_id") or col_map.get("_atom_site.auth_asym_id")
    seq_col = col_map.get("_atom_site.label_seq_id") or col_map.get("_atom_site.auth_seq_id")
    x_col = col_map.get("_atom_site.Cartn_x")
    y_col = col_map.get("_atom_site.Cartn_y")
    z_col = col_map.get("_atom_site.Cartn_z")

    if None in (grp_col, atm_col, chn_col, seq_col, x_col, y_col, z_col):
        return {}

    max_col = max(grp_col, atm_col, chn_col, seq_col, x_col, y_col, z_col)
    result: dict[str, list[tuple[int, float, float, float]]] = {}

    for i in range(start, len(lines)):
        line = lines[i].strip()
        if not line or line.startswith("#") or line.startswith("_"):
            break
        parts = line.split()
        if len(parts) <= max_col:
            continue
        try:
            if parts[grp_col] == "ATOM" and parts[atm_col] == "CA" and parts[seq_col] != ".":
                chain = parts[chn_col]
                resnum = int(parts[seq_col])
                x = float(parts[x_col])
                y = float(parts[y_col])
                z = float(parts[z_col])
                result.setdefault(chain, []).append((resnum, x, y, z))
        except (ValueError, IndexError):
            pass

    # Sort each chain by residue number
    return {ch: sorted(coords, key=lambda c: c[0]) for ch, coords in result.items()}


def parse_ca_plddt(cif_path: Path) -> dict[str, list[float]]:
    """Extract per-residue pLDDT from CIF Cα B-factor column.

    AF3 writes pLDDT into the B_iso_or_equiv field. Returns
    {chain_id: [plddt, ...]} with residues ordered by label_seq_id.

    Why B-factor and not atom_plddts from JSON?
      The confidences JSON contains atom_plddts (one value per *atom*, not
      per residue). Naively slicing it as per-residue is a common bug that
      gives wrong values. The CIF B-factor at Cα is guaranteed to be
      one value per residue and directly contains the pLDDT.
    """
    with open(cif_path) as f:
        lines = f.readlines()

    col_map, start = _parse_atom_site_columns(lines)
    if not col_map:
        return {}

    grp_col = col_map.get("_atom_site.group_PDB")
    atm_col = col_map.get("_atom_site.label_atom_id")
    chn_col = col_map.get("_atom_site.label_asym_id") or col_map.get("_atom_site.auth_asym_id")
    seq_col = col_map.get("_atom_site.label_seq_id") or col_map.get("_atom_site.auth_seq_id")
    bf_col = col_map.get("_atom_site.B_iso_or_equiv")

    if None in (grp_col, atm_col, chn_col, seq_col, bf_col):
        return {}

    max_col = max(grp_col, atm_col, chn_col, seq_col, bf_col)
    raw: dict[str, list[tuple[int, float]]] = {}

    for i in range(start, len(lines)):
        line = lines[i].strip()
        if not line or line.startswith("#") or line.startswith("_"):
            break
        parts = line.split()
        if len(parts) <= max_col:
            continue
        try:
            if parts[grp_col] == "ATOM" and parts[atm_col] == "CA" and parts[seq_col] != ".":
                chain = parts[chn_col]
                resnum = int(parts[seq_col])
                plddt = float(parts[bf_col])
                raw.setdefault(chain, []).append((resnum, plddt))
        except (ValueError, IndexError):
            pass

    return {ch: [p for _r, p in sorted(vals)] for ch, vals in raw.items()}


# ═════════════════════════════════════════════════════════════════════════════
# GEOMETRY
# ═════════════════════════════════════════════════════════════════════════════

def centre_of_mass(coords: list[tuple[int, float, float, float]]
                   ) -> tuple[float, float, float]:
    """Compute the centre of mass of Cα coordinates.

    Each element is (resnum, x, y, z); resnum is ignored for the calculation.
    """
    n = len(coords)
    if n == 0:
        return (0.0, 0.0, 0.0)
    return (
        sum(c[1] for c in coords) / n,
        sum(c[2] for c in coords) / n,
        sum(c[3] for c in coords) / n,
    )


def radius_of_gyration(coords: list[tuple[int, float, float, float]]) -> float:
    """Compute the radius of gyration of Cα coordinates."""
    if not coords:
        return 0.0
    cx, cy, cz = centre_of_mass(coords)
    return math.sqrt(
        sum((c[1] - cx) ** 2 + (c[2] - cy) ** 2 + (c[3] - cz) ** 2
            for c in coords)
        / len(coords)
    )


def euclidean_distance(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """Euclidean distance between two points of equal dimension."""
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


# ═════════════════════════════════════════════════════════════════════════════
# MODEL SELECTION
# ═════════════════════════════════════════════════════════════════════════════

def hetc_centrality(cif_path: Path,
                    split_at: int = PROPELLER_LENGTH) -> float | None:
    """Score how symmetrically HET-C sits between the two NLR propellers.

    Metric: abs(d(HET-C_COM, P1_COM) - d(HET-C_COM, P2_COM)).
    Lower is better: HET-C is equidistant from both propellers,
    consistent with sitting in the cleft between them.

    Why abs(d1 - d2) and not max(d1, d2)?
      max(d1, d2) is minimised when HET-C is physically close to both
      propeller centres — but a model where one propeller rotates and
      wraps around HET-C also brings that propeller's CoM very close,
      making max(d1, d2) small even though the geometry is wrong.
      abs(d1 - d2) measures *asymmetry*: wrapping creates a large
      difference (d_wrapped ≈ 0, d_other = normal), so those models
      score poorly. Correct two-cleft binding gives d1 ≈ d2 → score ≈ 0.

    The NLR chain (A) is split at residue `split_at` into propeller 1
    (residues 1..split_at) and propeller 2 (split_at+1..end).

    Returns None if coordinates cannot be extracted.
    """
    by_chain = parse_ca_coords(cif_path)
    chain_a = by_chain.get("A", [])
    chain_b = by_chain.get("B", [])
    if not chain_a or not chain_b:
        return None

    p1 = [c for c in chain_a if c[0] <= split_at]
    p2 = [c for c in chain_a if c[0] > split_at]
    if not p1 or not p2:
        return None

    com_p1 = centre_of_mass(p1)
    com_p2 = centre_of_mass(p2)
    com_b = centre_of_mass(chain_b)

    d1 = euclidean_distance(com_b, com_p1)
    d2 = euclidean_distance(com_b, com_p2)
    return abs(d1 - d2)


def find_all_cifs(job_dir: Path) -> list[tuple[int, int, Path]]:
    """Return all (seed, sample, cif_path) tuples in an AF3 job directory.

    Looks for subdirectories named seed-N_sample-M containing *.cif files.
    """
    results = []
    for subdir in job_dir.iterdir():
        if not subdir.is_dir():
            continue
        m = re.match(r"seed-(\d+)_sample-(\d+)$", subdir.name)
        if not m:
            continue
        cifs = list(subdir.glob("*.cif"))
        if cifs:
            results.append((int(m.group(1)), int(m.group(2)), cifs[0]))
    return results


# ═════════════════════════════════════════════════════════════════════════════
# WD40 REPEAT DETECTION
# ═════════════════════════════════════════════════════════════════════════════

def find_repeat_boundaries(nlr_sequence: str) -> list[tuple[int, int]]:
    """Detect WD40 repeat boundaries using the conserved TLEGH motif.

    Returns list of (start, end) tuples (1-indexed, inclusive) for each
    detected repeat. TLEGH sits at positions 5-9 within each ~42 AA repeat,
    so the repeat start is 4 residues before the T of TLEGH.

    The last repeat extends to the next repeat's start (or +41 AA from
    its own start if it is the final repeat).
    """
    starts = [max(0, m.start() - 4) for m in TLEGH_RE.finditer(nlr_sequence)]
    boundaries: list[tuple[int, int]] = []
    for i, s in enumerate(starts):
        end = (starts[i + 1] - 1 if i + 1 < len(starts)
               else min(s + REPEAT_LENGTH - 1, len(nlr_sequence) - 1))
        boundaries.append((s + 1, end + 1))  # convert to 1-indexed
    return boundaries


def classify_nlr_residue(residue_1idx: int,
                         boundaries: list[tuple[int, int]]
                         ) -> tuple[bool, int | None, int | None]:
    """Check if an NLR residue falls on a hypervariable position within a repeat.

    Args:
        residue_1idx: 1-indexed residue position in the NLR sequence.
        boundaries: output of find_repeat_boundaries().

    Returns:
        (is_hypervariable, repeat_number_or_None, intra_repeat_position_or_None).
        repeat_number is 1-indexed. If the residue is outside all detected
        repeats, returns (False, None, None).
    """
    for i, (start, end) in enumerate(boundaries):
        if start <= residue_1idx <= end:
            intra = residue_1idx - start + 1
            return (intra in NLR_REPEAT_HYPERVARIABLE, i + 1, intra)
    return (False, None, None)
