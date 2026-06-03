# NLR–HET-C interaction prediction pipeline

Master's thesis project on interaction specificity between *Podospora anserina*
NLR immune receptors (HET-E, HET-D) and their ligand (HET-C).

This document describes the AlphaFold 3 contact-validation track: how the
structural predictions are generated, quality-controlled, and tested for
contact enrichment on the hypervariable positions. The sequence-only
compatibility classifier (Model 1) is a separate, independent analysis and is
documented in [`README.md`](README.md); it does **not** use any AF3 output as
a feature.

---

## Stages

The AF3 track runs in three stages, each producing artifacts the next stage
consumes:

1. **Sequence preparation** — build 36 NLR protein sequences and 11 HET-C
   alleles; emit 275 AlphaFold 3 input JSONs.
2. **Structure prediction on WCSS** — run AF3 3.0.1 on the Bem2 cluster
   (Apptainer/Singularity); manually QC CIFs, rerun failures, bundle accepted
   predictions.
3. **Biological validation** — verify that AF3 contacts are enriched on
   residues under diversifying selection, using a surface-filtered null
   (permutation pool restricted to `rel_SASA ≥ 0.20`); per-NLR acceptance
   statistics.

The sequence-only compatibility classifier (Model 1) is documented separately
in [`README.md`](README.md). It shares only the input sequences with this
track and is otherwise independent.

---

## Scripts

### Stage 1: sequence preparation

| Script | Role |
|---|---|
| `parse_nlr_fasta.py` | Parse aligned WD40 nucleotide FASTA (Ament-Velásquez 2025), strip gaps, translate to amino acids, assign phenotype labels (E1–e4, D1–d3). Emits 36 NLR protein sequences + metadata. |
| `download_hetc_sequences.py` | Fetch 11 HET-C alleles from NCBI (C1–C4 protein records from Saupe 1995; C5–C11 translated CDS from Bastiaans 2014). |
| `generate_af3_inputs.py` | Build AF3 input JSONs for all 275 NLR × HET-C pairs (36 × 11 minus 121 D3/d3 pairs already covered by d3 phenotype labels in metadata). Each JSON has two `proteinChain` entries and a `_pipeline_meta` block with the ground-truth label. |

### Stage 2: AF3 on WCSS

The WCSS pipeline replaces the former AlphaFold Server workflow. All
predictions now run locally on the cluster using the official AF3 3.0.1
container, giving full access to CIF files and per-token confidence arrays —
information the public AF Server does not expose.

| Script | Where it runs | Role |
|---|---|---|
| `generate_af3_wcss.py` | local | Convert AF3-Server-style JSONs from `data/af3_inputs/labeled/` to AF3 3.0.1-local format (adds `modelSeeds`, strips `_pipeline_meta`) and emit a batch submission script. |
| `data/af3_wcss/run_af3_labeled.sh`, `rerun_all_bad.sh`, `rerun_3.sh` | WCSS | Submit AF3 jobs via `sub-alphafold-3.0.1`. Each subsequent rerun targets jobs that failed propeller QC in the previous batch. |
| `check_propellers.py` | WCSS | Standalone CIF QC. Computes centre of mass and radius of gyration for the two WD40 propellers (residues 1–294 and 295–588), flags wrapping artefacts via `abs(d(HET-C, P1) − d(HET-C, P2))`, and copies the best-scoring CIF per job to `wcss/selected/<batch>/` for manual review. |
| `collect_selected.py` | WCSS | Standalone bundler. Reads `accepted_jobs.txt`, locates each job's CIF + two JSONs across all AF3 output directories (`af3_outputs`, `af3_outputs_rerun_{1,2,3}`), and packs them into a flat `thesis/ready/<job>/` folder with an MD5 manifest. |

Filename convention for the `wcss/selected/<batch>/` handoff:

    <job>.cif                            ← auto-pick (default best seed)
    <job>__seed-N_sample-M.cif           ← manual override of a specific seed

`collect_selected.py` reads the seed encoding from the filename when present;
otherwise it re-runs the centrality metric and picks the best seed itself.

### Stage 3: validation and statistics

| Script | Role |
|---|---|
| `selection_stats.py` | Per-NLR acceptance table across batches: how many AF3 runs were needed per NLR variant, batch-by-batch progression, and a "fun facts" section summarising wasted GPU time on still-unresolved jobs. |
| `validate_af3_contacts.py` | For each bundle, extracts inter-chain contacts with `contact_prob ≥ 0.2` from the confidences JSON and tests their enrichment on diversifying-selection sites (NLR: positions 10, 11, 12, 14, 30, 32, 39 within each WD40 repeat; HET-C: 118, 133, 153). **Legacy baseline**: uniform over whole sequence. Kept for continuity; newer analyses use the SASA-filtered variant below. Uses Laplace-smoothed permutation p-values `(k + 1) / (N + 1)`. |
| `compute_sasa.py` | Precomputes per-residue SASA (Bio.PDB.SASA ShrakeRupley) and surface masks at thresholds {0.15, 0.20, 0.25} for every accepted bundle. Caches one JSON per job in `data/validation/sasa/`. Also writes `_summary.tsv` showing how the baseline changes per job. |
| `validate_af3_contacts_sasa.py` | **Primary enrichment test.** Uses surface-filtered null: permutation pool = residues with `rel_SASA ≥ threshold`, intersected with repeat boundaries for the NLR side. Produces `contact_validation_sasa.tsv` + `summary_sasa.txt` with both the new surface-baseline enrichment and the classic number for comparison. |

### Shared modules (not run directly)

| Module | Contents |
|---|---|
| `constants.py` | WD40 geometry, propeller thresholds, contact cutoff, hypervariable position sets, dataset exclusions. Single source of truth — every script imports from here. |
| `cif_utils.py` | CIF parsing helpers: `find_repeat_boundaries`, `classify_nlr_residue`, centre-of-mass and radius-of-gyration utilities. |
| `sasa_utils.py` | Shrake-Rupley wrapper on Bio.PDB, Tien 2013 max-ASA reference, surface-mask derivation, JSON cache I/O. |

Both `check_propellers.py` and `collect_selected.py` deliberately have **no
local imports** — they are copied to WCSS as single files and must stand alone.

---

## Directory layout

    thesis/
      scripts/                  # all pipeline scripts
      data/
        sequences/
          het_c/                # 11 HET-C FASTAs + full C1–C11 matrix TSV
          het_ed_wd40/          # raw WD40 FASTA + 36 NLR AA FASTAs + metadata
        af3_inputs/
          labeled/              # 275 AF3 input JSONs (all with ground-truth label)
        af3_outputs/            # 233 accepted bundles (flat layout)
          <job>/
            *_model.cif
            *_confidences.json
            *_summary_confidences.json
        af3_wcss/               # labeled/ (AF3 3.0.1 JSONs) + submission .sh scripts
        validation/reports/
          bundle_manifest.tsv   # MD5 + source dir for each accepted CIF
          selection_stats.tsv   # per-NLR acceptance counts
          contact_validation.tsv        # legacy (uniform-over-sequence null)
          contact_validation_sasa.tsv   # primary (surface-filtered null)
          summary.txt
          summary_sasa.txt
        validation/sasa/
          <job>.json            # per-residue SASA + precomputed masks @0.15/0.20/0.25
          _summary.tsv          # surface counts + baseline-change summary
        reference/FixingHetDE/  # Ament-Velásquez 2025 companion repo (vendored)
        embeddings/             # placeholder — filled in stage 4
      PIPELINE.md               # this file

    wcss/                       # mirror of WCSS-side workspace (manual QC)
      selected/{original,rerun_1,rerun_2,rerun_3}/    # accepted CIFs per batch

---

## Interaction matrix

1 = incompatible (triggers cell death), 0 = compatible.

Sources: C1–C4 from Ament-Velásquez et al. 2025 Fig. 1b;
C5–C11 from Bastiaans et al. 2014 Fig. 1.

```
         C1  C2  C3  C4  C5  C6  C7  C8  C9  C10 C11
E1        0   1   0   0   0   0   1   1   0   0   0
E2        1   0   1   1   1   1   0   1   1   1   1
E3        1   0   0   1   1   0   0   1   0   1   1
e4        0   0   0   0   0   0   0   0   0   0   0
D1        0   1   0   1   0   0   1   1   1   1   1
D2        0   0   0   1   0   0   1   0   1   1   1
d3        0   0   0   0   0   0   0   0   0   0   0
```

Every C-allele has a known specificity, so there is no separate discovery
set — all 275 pairs carry a label. After excluding the 11 jobs for
`E1_Fj897789` (which never produced two propellers), 264 pairs remain active.
Of those, 233 were accepted across four AF3 batches (88%), giving
roughly a 1 : 3 positive-to-negative ratio for the classifier.

---

## HET-C allele provenance

### C1–C4

Saupe et al. 1995, *Curr Genet* 27:466. Original four alleles characterised by
incompatibility testing. Interaction table for C1–C4 is taken directly from
Ament-Velásquez 2025 Fig. 1b.

| Allele | Protein accession | Reactive with |
|---|---|---|
| C1 | AAA33626.1 | E2, E3 |
| C2 | AAA20542.1 | E1, D1 |
| C3 | AAA33628.1 | E2 |
| C4 | AAA33629.1 | E2, E3, D1, D2 |

### C5–C11

Bastiaans et al. 2014, *Mol Biol Evol* 31(4):962–974.
Sequences deposited as nucleotide records with CDS annotations; translated to
amino acids via NCBI `fasta_cds_aa`.

| Allele | Nucleotide | Interaction basis |
|---|---|---|
| C5  | KF951052 | ≡ C1 class |
| C6  | KF951053 | ≡ C3 class |
| C7  | KF951054 | E1, D1, D2 |
| C8  | KF951055 | E1, E2, E3, D1 (C1 + {E1, D1}) |
| C9  | KF951056 | E2, D1, D2 |
| C10 | KF951057 | ≡ C4 class |
| C11 | KF951058 | ≡ C4 class |

**C8 = C1 + one residue change.** A single substitution at HET-C position 118
(Cys → Arg) expands recognition from {E2, E3} to {E1, E2, E3, D1}. This
near-isogenic contrast is a key benchmark: AF3 should predict a visibly
different interface for C1 vs C8 against E1 / D1.

---

## Key metrics and thresholds

All defined once in `constants.py`.

| Constant | Value | Used by |
|---|---|---|
| `REPEAT_LENGTH` | 42 AA | repeat boundary detection |
| `REPEATS_PER_PROPELLER` | 7 | propeller geometry |
| `PROPELLER_LENGTH` | 294 AA | propeller slicing |
| `RG_MIN`, `RG_MAX` | 12, 38 Å | propeller radius of gyration |
| `COM_SEPARATION_MIN` | 30 Å | two propellers must be spatially separated |
| `CONTACT_PROB_THRESHOLD` | 0.2 | contact definition |
| `HETC_HYPERVARIABLE` | {118, 133, 153} | HET-C sites under diversifying selection |
| `NLR_REPEAT_HYPERVARIABLE` | {10, 11, 12, 14, 30, 32, 39} | WD40-repeat-local sites under selection |
| `EXCLUDED_NLR_PREFIXES` | `("e1_fj897789",)` | excluded from all analyses |

---

## The compatibility classifier (Model 1)

The AF3 data-preparation pipeline described above is frozen — any further
changes are bug-fixes only. The sequence-only compatibility classifier is a
separate, independent analysis: it embeds the NLR and HET-C sequences with
ESM-C 600M, forms pair features, and trains an XGBoost classifier validated by
leave-one-NLR-out cross-validation. It does not consume any AF3 output. See
[`README.md`](README.md) for its architecture, scripts, and results.

---

## References

- Ament-Velásquez et al. 2025 — *Reconstructing NOD-like receptor alleles
  with high internal conservation in Podospora anserina using long-read
  sequencing.* Microbial Genomics 11:001442.
- Bastiaans et al. 2014 — *Natural Variation of Heterokaryon Incompatibility
  Gene het-c in Podospora anserina Reveals Diversifying Selection.*
  Mol Biol Evol 31(4):962–974.
- Paoletti et al. 2007 — *Diversifying selection in the WD40 domain of NLR
  proteins in Podospora anserina.*
- Saupe et al. 1995 — *Inviability of het-c/het-c heterokaryons involves a
  multicopy suppressor gene and induced cell death in Podospora anserina.*
  Curr Genet 27:466.
