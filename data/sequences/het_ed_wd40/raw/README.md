# Source

`wd40_raw.fa` is a copy of the aligned WD40 domain from the FixingHetDE repository
(Ament-Velásquez et al. 2025):

    data/reference/FixingHetDE/NWDgenes/data/2025.04.14_hnwd_master_onlyWDdomain_NWDs_noGuides_noemptycols_4paper.fa

This file contains aligned nucleotide sequences of the WD40 domain for all NWD genes
(het-d, het-e, het-r, hnwd1, hnwd3, nwd1-6, nwdp-2) from multiple P. anserina strains.
The alignment includes gaps (---) and the cryptic repeats at the C-terminus.

Processed by: `scripts/parse_nlr_fasta.py`
Output: `../het_ed_sequences.fasta` (only het-d and het-e, translated to amino acids)
