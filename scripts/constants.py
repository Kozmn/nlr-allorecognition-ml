"""
constants.py — Shared biological constants and thresholds for the pipeline.

All scripts in this pipeline import constants from here to ensure consistency.
Changing a value here propagates to every script that uses it.

Why this module exists:
  Before refactoring, the same constants (REPEAT_LENGTH, HETC_HYPERVARIABLE,
  etc.) were duplicated across 4+ scripts. A silent edit in one file without
  updating the others would produce inconsistent results — a serious risk in
  a classification pipeline where features depend on these exact values.

Author: Kacper Koźmin
"""

import re

# ── WD40 repeat geometry ─────────────────────────────────────────────────────
# Each WD40 repeat is ~42 amino acids long and contains the conserved TLEGH
# motif at positions 5-9. The NLR proteins (HET-E, HET-D) have two
# beta-propellers, each consisting of 7 WD40 repeats.
#
# Source: Paoletti et al. 2007 (diversifying selection analysis)
#         Ament-Velásquez et al. 2025 (functional allele structure)

REPEAT_LENGTH = 42                                       # AA per WD40 repeat
REPEATS_PER_PROPELLER = 7                                # repeats per propeller
PROPELLER_LENGTH = REPEAT_LENGTH * REPEATS_PER_PROPELLER  # 294 AA

# Regex for the conserved motif used to detect repeat boundaries
TLEGH_RE = re.compile(r"TLEGH")

# ── Propeller geometry thresholds ─────────────────────────────────────────────
# Used by check_propellers.py to verify that both halves of the NLR chain
# fold into compact globular domains separated in space.
#
# Rg (radius of gyration) of a WD40 propeller is typically 15-25 Å.
# RG_MIN/RG_MAX define the acceptable range.
# COM_SEPARATION_MIN: the two propeller centres of mass must be at least
# this far apart (otherwise they may be collapsed into a single domain).

RG_MIN = 12.0               # Å
RG_MAX = 38.0               # Å
COM_SEPARATION_MIN = 30.0   # Å

# ── Contact analysis ─────────────────────────────────────────────────────────
# AF3 outputs a contact_probs matrix where each value is the predicted
# probability that two residues are in contact (<8Å Cβ distance).
# We count a pair as "in contact" if the probability exceeds this threshold.
#
# 0.2 is intentionally permissive — we want to capture weak but real
# interactions rather than miss them. The ML model will learn which
# contacts matter.

CONTACT_PROB_THRESHOLD = 0.2

# ── Positions under diversifying selection ────────────────────────────────────
# These are the residue positions where natural selection has driven
# allelic diversification. If AF3 predicts contacts at these positions,
# it suggests the model captures the specificity-determining interface.
#
# HET-C (208 AA effector protein):
#   Positions 118, 133, 153 — hypervariable across alleles C1-C11.
#   Source: Bastiaans et al. 2014 (Mol. Biol. Evol.)
#
# NLR WD40 repeats (within each ~42 AA repeat):
#   Positions 10, 11, 12, 14, 30, 32, 39 — under diversifying selection.
#   Source: Paoletti et al. 2007; Ament-Velásquez et al. 2025

HETC_HYPERVARIABLE = {118, 133, 153}
NLR_REPEAT_HYPERVARIABLE = {10, 11, 12, 14, 30, 32, 39}

# ── Dataset exclusions ───────────────────────────────────────────────────────
# E1_Fj897789: GenBank reference allele. All 11 AF3 jobs for this NLR
# variant failed to produce output (2 propellers never formed).
# Excluded from all analyses so it doesn't create missing-data artifacts.

EXCLUDED_NLR_PREFIXES = ("e1_fj897789",)
