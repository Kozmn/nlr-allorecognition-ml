# NLR × HET-C compatibility prediction

Master's thesis code base on allorecognition between WD40-repeat NLR proteins
and their HET-C partners in the filamentous fungus *Podospora anserina*.

The repository contains **two independent analyses** sharing the same input
sequences:

1. **AF3 contact validation** — does AlphaFold 3 place inter-chain contacts
   on the biologically known hypervariable positions?
2. **Model 1 — sequence-only compatibility classifier** — given two sequences
   (one NLR, one HET-C), can we predict whether they trigger incompatibility?
   Built from ESM-C 600M embeddings + XGBoost, validated by leave-one-NLR-out
   cross-validation on 25 NLRs × 11 HET-C alleles (275 pairs).

> **Important:** the two analyses are completely independent. Model 1 does
> **not** use any AF3 output as a feature. It runs purely on ESM-C sequence
> embeddings. The AF3 work is parallel evidence about where the interface
> lies, not an input to the classifier.

This is the code base for a master's thesis at the Wrocław University of Science and Technology.

---

## Data and credit

The 25 NLR sequences (HET-D / HET-E variants) and the curated NLR–HET-C
interaction matrix used in this project come from the work of
**S. Lorena Ament-Velásquez and collaborators** (Stockholm University /
Uppsala University), distributed via the
[`SLAment/FixingHetDE`](https://github.com/SLAment/FixingHetDE/) repository.

If you use this data, please cite the corresponding publications listed in
[`CITATIONS.md`](CITATIONS.md), starting with Ament-Velásquez *et al.*.
The HET-C alleles C1–C11 are taken from Saupe 1995 and Bastiaans 2014.

The **code in this repository** is original (master's thesis, Kacper Koźmin)
and is released under the MIT licence — see [`LICENSE`](LICENSE).

---

## Background

In *Podospora anserina*, vegetative (heterokaryon) incompatibility is governed
by the *het-c* locus encoding a glycolipid-transfer protein, and a paired NLR
(nucleotide-binding leucine-rich repeat) gene encoding HET-D or HET-E. Each NLR
recognises a specific subset of HET-C alleles and triggers a programmed cell
death response in incompatible cells. The recognition specificity is encoded
in seven hypervariable positions of every WD40 repeat in the NLR propeller and
in three hypervariable positions of HET-C.

This project asks two **separate** questions:

1. **Where does AlphaFold 3 place inter-chain contacts in NLR×HET-C complexes,
   and does it concentrate them on the biologically known hypervariable
   positions?** (See `validate_af3_contacts.py`.)
2. **Can a classifier predict whether a given NLR×HET-C pair will trigger
   incompatibility from sequence alone?** (See `train_lono.py` and the model
   comparison scripts.)

Both questions use the same NLR and HET-C sequences as input, but the two
pipelines are otherwise independent.

---

## Pipeline

The two branches share only the input sequences. Nothing flows between them:

```
                                          ┌──→ AlphaFold 3 (233 bundles)
                                          │      ↓
                                          │      contact validation
                                          │      (permutation test, SASA, FWHM)
sequences (25 NLR × 11 HET-C = 275 pairs) ─┤
                                          │
                                          └──→ ESM-C 600M embeddings
                                                 ↓
                                                 pair features (4608-D)
                                                 ↓
                                                 Model 1 (XGBoost + LONO)
```

### 1. Data preparation
- 25 NLR sequences (HET-D / HET-E variants) covering 7 reactivity phenotypes
  (E1, E2, E3, e4, D1, D2, d3); curated by Ament-Velásquez *et al.*
- 11 HET-C alleles (C1–C11) from Saupe 1995 and Bastiaans 2014
- 275 NLR×HET-C pairs labelled with the literature-derived interaction matrix

### 2. AlphaFold 3 contact validation (independent track)
- 264 active pairs submitted to AF3 in four runs on the WCSS HPC cluster;
  233 bundles accepted after manual QC of WD40 propeller folding (88 % yield)
- Permutation test (10 000 permutations, Laplace correction) of the fraction
  of inter-chain contacts landing on hypervariable positions
- **Result (raw baseline):** NLR enrichment **4.96×**, HET-C **4.66×**;
  *p* < 0.05 in 96 % of pairs (NLR), 45 % (HET-C)
- **Result (SASA-filtered baseline, threshold rel-SASA ≥ 0.25):** NLR
  enrichment **5.84×**, HET-C **2.27×** — surface-restricted null is more
  conservative for HET-C but stronger for NLR

### 3. Model 1 — pair compatibility classifier (independent track)
- ESM-C 600M sequence embeddings (mean and max pool, 1152 D each) combined
  into pair features `[a; b; |a − b|; a ⊙ b]` of dimension 4608
- **No AF3-derived features.** Inputs are entirely sequence-based.
- XGBoost binary classifier (compatible vs incompatible)
- Leave-one-NLR-out (LONO) cross-validation: every NLR sequence is held out
  once, all 11 of its pairs predicted from the remaining 24 sequences
- **Best model — XGBoost ensemble (mean + max pool), LONO:** MCC **0.665**,
  AUC **0.850**
- **After hyper-parameter tuning (nested 5-fold inner CV, 108-config grid):**
  MCC **~0.683** (preliminary, full run ongoing on WCSS)

---

## Results summary

### Model 1 — classifier comparison (LONO)

| Model | Validation | MCC | AUC |
| ----- | ---------- | --- | --- |
| **XGBoost (tuned, mean+max ensemble)** | LONO | **0.683** | 0.847 |
| XGBoost (default, ensemble) | LONO | 0.665 | 0.850 |
| XGBoost on hypervariable-only embeddings | LONO | 0.624 | 0.846 |
| SVM with RBF kernel | LONO | 0.570 | 0.808 |
| Logistic regression | LONO | 0.498 | 0.821 |
| Supervised contrastive (SupCon) classifier | LONO | 0.285 | 0.714 |
| XGBoost (LOO — matrix-completion diagnostic) | LOO | 0.902 | 0.994 |

Per-phenotype MCC is high for phenotypes with several training NLRs (E1: 0.96,
E3: 0.91) and undefined for singletons (D1, E2 — n = 1 NLR each), where LONO
removes the only example from training.

### AF3 contact validation (parallel evidence)

| | observed | surface-baseline | enrichment |
| --- | --- | --- | --- |
| NLR (hypervariable positions) | 70.7 % | 12.1 % | **5.84×** |
| HET-C (positions 118 / 133 / 153) | 1.9 % | 0.8 % | **2.27×** |

These numbers come from the SASA-filtered permutation test at rel-SASA ≥ 0.25.

---

## Repository layout

```
.
├── README.md
├── LICENSE
├── CITATIONS.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── scripts/        # all analysis pipelines
│   ├── parse_nlr_fasta.py
│   ├── download_hetc_sequences.py
│   ├── generate_af3_inputs.py
│   ├── compute_embeddings.py
│   ├── compute_embeddings_per_residue.py
│   ├── build_pair_features.py
│   ├── build_pair_features_hv.py
│   ├── validate_af3_contacts.py
│   ├── validate_af3_contacts_sasa.py
│   ├── refine_enrichment.py
│   ├── compute_sasa.py
│   ├── sasa_utils.py
│   ├── train_lono.py             # XGBoost + LONO
│   ├── train_loo.py              # XGBoost + LOO
│   ├── train_lono_logistic.py    # logistic regression baseline
│   ├── train_lono_svm.py         # SVM RBF baseline
│   ├── xgb_grid_nested.py        # nested-CV hyperparameter tuning
│   ├── contrastive_supcon.py     # SupCon + UMAP visualization
│   ├── ensemble_eval.py
│   ├── plot_models_comparison.py   # comparison table + figure for all models
│   ├── visualize_embeddings_pca.py
│   ├── visualize_supcon_pca.py
│   ├── aggregate_heatmaps.py       # builds the aggregated contact cache
│   ├── make_contact_heatmaps.py    # publication-quality contact heatmaps
│   ├── selection_stats.py
│   ├── collect_selected.py
│   ├── check_propellers.py
│   ├── cif_utils.py
│   ├── constants.py
│   ├── model_evaluation.py         # Model 1 evaluation report + figures
│   └── generate_af3_wcss.py
├── wcss/           # SLURM submission scripts for the HPC cluster
└── data/
    ├── sequences/  # NLR FASTA, HET-C FASTA, interaction matrix
    ├── models/     # CSV outputs (predictions, fold summaries, metrics)
    │   └── eval/   # comparison tables and figures
    └── validation/
        └── reports/  # AF3 contact validation results, heatmaps
```

Large files (raw AF3 outputs, full embedding matrices, per-pair feature
matrices, SASA caches) are excluded by `.gitignore` and can be regenerated
from the scripts.

---

## Reproducing the analysis

> Disclaimer: AlphaFold 3 structure prediction requires either a Google
> AlphaFold Server account or the official AF3 model weights (CC BY-NC-SA 4.0
> licence) on a GPU-enabled machine. The 233 prebuilt bundles used in this
> project are not redistributed — only the downstream analyses are runnable
> from this repository.

### Setup

```bash
git clone <repo-url>
cd <repo-name>
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run Model 1 — classifier (independent of AF3)

```bash
# 1. Build pair features from precomputed ESM-C embeddings
python scripts/build_pair_features.py

# 2. Train XGBoost with leave-one-NLR-out CV
python scripts/train_lono.py

# 3. Compare against logistic regression and SVM baselines
python scripts/train_lono_logistic.py
python scripts/train_lono_svm.py

# 4. Hyperparameter tuning (nested 5-fold inner CV, ~1–2 h locally)
python scripts/xgb_grid_nested.py

# 5. Aggregate all models into one comparison table and figure
python scripts/plot_models_comparison.py
```

### Run the AF3 contact validation (independent of Model 1)

```bash
# Permutation test, raw baseline
python scripts/validate_af3_contacts.py

# SASA-filtered surface-restricted baseline
python scripts/compute_sasa.py
python scripts/validate_af3_contacts_sasa.py --threshold 0.25
```


---

## Tech stack

| Component | Tools |
| --------- | ----- |
| Structure prediction (parallel evidence track) | AlphaFold 3 |
| Sequence embeddings (Model 1 input) | ESM-C 600M (EvolutionaryScale) |
| Classification | XGBoost, scikit-learn (SVM, logistic regression) |
| Contrastive learning | PyTorch, SupCon |
| Visualization | matplotlib, UMAP, PCA |
| Structural analysis | Biopython (Bio.PDB.SASA.ShrakeRupley) |
| Cluster | WCSS HPC, Slurm |

---

## Citation and acknowledgements

- External tools (AlphaFold 3, ESM-C, XGBoost, SupCon, UMAP, etc.) and primary
  biological references are listed in [`CITATIONS.md`](CITATIONS.md). Please
  cite them if you build on this work.
- The NLR sequence data and the NLR–HET-C interaction matrix come from
  Ament-Velásquez *et al.* via
  [`SLAment/FixingHetDE`](https://github.com/SLAment/FixingHetDE/). The
  experimental phenotype assignments are theirs; this project's code does not
  redistribute their data.
---


## Contact

Kacper Koźmin — `kacper.kozmin@gmail.com`
