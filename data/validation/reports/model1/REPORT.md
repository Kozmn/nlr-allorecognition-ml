# Model 1 — ewaluacja (ESM-C 600M + XGBoost + LONO)

**Zadanie:** klasyfikacja binarna — czy para (NLR WD40, HET-C) wywołuje reakcję niekompatybilności (apoptozę).

**Walidacja:** Leave-One-NLR-Out (25 foldów). Każda z 275 par została predicted dokładnie raz (out-of-fold).

---

## Wyniki overall (275 par)

| Metryka | mean-pool | max-pool |
|---|---:|---:|
| N | 275 | 275 |
| Pozytywy | 68 | 68 |
| Negatywy | 207 | 207 |
| Accuracy | 0.869 | 0.873 |
| MCC | 0.633 | 0.643 |
| F1 (weighted) | 0.865 | 0.869 |
| F1 (macro) | 0.815 | 0.819 |
| F1 (pozytyw) | 0.714 | 0.720 |
| Precision | 0.776 | 0.789 |
| Recall | 0.662 | 0.662 |
| AUC | 0.826 | 0.860 |

---

## Metryki per fenotyp

| Fenotyp | n | pos | MCC (mean) | MCC (max) | F1 (mean) | F1 (max) |
|---|---:|---:|---:|---:|---:|---:|
| D1 | 11 | 7 | 0.000 | 0.000 | 0.000 | 0.000 |
| D2 | 22 | 10 | 0.000 | 0.000 | 0.000 | 0.000 |
| E1 | 66 | 18 | 0.896 | 0.886 | 0.923 | 0.909 |
| E2 | 11 | 9 | 0.289 | 0.516 | 0.500 | 0.800 |
| E3 | 44 | 24 | 0.869 | 0.955 | 0.941 | 0.980 |
| d3 | 77 | 0 | n/d | n/d | 0.000 | 0.000 |
| e4 | 44 | 0 | n/d | n/d | 0.000 | 0.000 |

---

## Rozkład MCC po foldach

- **mean-pool** (n=14 foldów z MCC):
  - Mediana: 1.000
  - Średnia: 0.669 ± 0.430
  - Min:     0.000
  - Max:     1.000
- **max-pool** (n=14 foldów z MCC):
  - Mediana: 1.000
  - Średnia: 0.667 ± 0.457
  - Min:     0.000
  - Max:     1.000

---

## Główny wniosek

- Lepszy pooling: **max** (MCC = 0.643 vs 0.633 dla mean).
- MCC = 0.64 — **silny sygnał klasyfikacyjny**. ESM-C poprawnie koduje cechy istotne dla kompatybilności.
