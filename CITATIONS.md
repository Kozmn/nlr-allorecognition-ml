# Citations and Attributions

This project builds on the following tools and datasets. If you use this code
or its outputs in academic work, please cite the corresponding publications.

## Structure prediction

**AlphaFold 3** (Abramson et al., *Nature*, 2024) — used to predict NLR×HET-C
complex structures (233 accepted bundles).

> Abramson, J., Adler, J., Dunger, J. *et al.* Accurate structure prediction of
> biomolecular interactions with AlphaFold 3. *Nature* **630**, 493–500 (2024).
> https://doi.org/10.1038/s41586-024-07487-w

License: CC BY-NC-SA 4.0 (model weights, see https://github.com/google-deepmind/alphafold3).
Predictions in this project were generated on the WCSS HPC cluster under
non-commercial academic use.

## Protein language model

**ESM-C 600M** (EvolutionaryScale, 2024) — used to compute per-sequence and
per-residue embeddings for NLR and HET-C proteins.

> Hayes, T., Rao, R., Akin, H. *et al.* Simulating 500 million years of evolution
> with a language model. *Science* **387**, 850–858 (2025).
> https://doi.org/10.1126/science.ads0018

License: ESM Cambrian (Apache 2.0), see https://github.com/evolutionaryscale/esm.

## Classifiers and feature analysis

**XGBoost** (Chen & Guestrin, *KDD*, 2016) — gradient-boosted decision trees,
primary classifier for NLR×HET-C compatibility prediction.

> Chen, T. & Guestrin, C. XGBoost: A Scalable Tree Boosting System.
> *Proceedings of the 22nd ACM SIGKDD*, 785–794 (2016).
> https://doi.org/10.1145/2939672.2939785

**scikit-learn** (Pedregosa et al., *JMLR*, 2011) — SVM, logistic regression,
metrics, utilities.

> Pedregosa, F. *et al.* Scikit-learn: Machine Learning in Python.
> *Journal of Machine Learning Research* **12**, 2825–2830 (2011).

## Contrastive learning

**SupCon** (Khosla et al., *NeurIPS*, 2020) — supervised contrastive loss used
to project pair embeddings into a class-separable space.

> Khosla, P., Teterwak, P., Wang, C. *et al.* Supervised Contrastive Learning.
> *Advances in Neural Information Processing Systems* **33**, 18661–18673 (2020).

## Visualization

**UMAP** (McInnes, Healy & Melville, 2018) — non-linear dimensionality reduction
for embedding visualization.

> McInnes, L., Healy, J. & Melville, J. UMAP: Uniform Manifold Approximation
> and Projection for Dimension Reduction. *arXiv:1802.03426* (2018).

## Surface-accessible area (SASA)

**Shrake–Rupley algorithm** as implemented in Biopython
(`Bio.PDB.SASA.ShrakeRupley`).

> Shrake, A. & Rupley, J. A. Environment and exposure to solvent of protein
> atoms. Lysozyme and insulin. *Journal of Molecular Biology* **79**, 351–371 (1973).

Maximum-accessible-surface-area reference values:

> Tien, M. Z., Meyer, A. G., Sydykova, D. K. *et al.* Maximum allowed solvent
> accessibilities of residues in proteins. *PLOS ONE* **8**, e80635 (2013).

## Biological references

The biology of *Podospora anserina* het genes and HET-C / NLR (HET-D / HET-E)
heterocompatibility is described in:

- Saupe, S. J. *et al.* The genetic and molecular dissection of an incompatibility
  reaction in *Podospora anserina*. *Genetics* **141**, 1305–1314 (1995).
- Bastiaans, E. *et al.* Allelic diversity and selection at *het-c* of *Podospora
  anserina*. (2014).
- Paoletti, M. *et al.* Selective acquisition of novel mating type and vegetative
  incompatibility genes via interspecies gene transfer in the globally invading
  eukaryote *Ophiostoma novo-ulmi*. *Molecular Ecology* (2007).

## Source of the NLR sequence data and interaction matrix

The 25 NLR (HET-D / HET-E) sequence variants and their NLR–HET-C reactivity
phenotypes used in this project were curated and made publicly available by
**S. Lorena Ament-Velásquez and collaborators** through:

> [`https://github.com/SLAment/FixingHetDE/`](https://github.com/SLAment/FixingHetDE/)

If you use this dataset, please credit Ament-Velásquez *et al.* and cite the
relevant publications referenced in that repository. The phenotype assignments
and reference sequences are theirs; this project's code does not redistribute
their data and only uses them as input.

## Project license

The code in this repository is released under the MIT License (see `LICENSE`).
This applies only to the code authored as part of this project; external tools
and weights remain subject to their own licenses listed above.
