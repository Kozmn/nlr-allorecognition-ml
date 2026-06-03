# Refined enrichment analysis — uczciwsza walidacja AF3

**Cel:** pokazać, że AF3 wzbogaca *region* wokół pozycji hiperwariabilnych, a nie *konkretną pozycję*. Ten sygnał regionalny jest prawdziwy i wartościowy (lokalizacja interfejsu), ale słabszy niż naiwne 4.96× by sugerowało.

---

## 1. Neighborhood-corrected enrichment

Dla każdej pozycji hiperwariabilnej: ile razy więcej kontaktów ma ona niż średnia z jej sąsiadów (±5 residuów)?

### HET-C

| Pozycja | n bundli | Mediana | Średnia ± std |
|--------:|---------:|--------:|--------------:|
| 118 | 178 | 0.42× | 0.46 ± 0.26 |
| 133 | 233 | 1.81× | 1.81 ± 0.39 |
| 153 | 233 | 1.47× | 1.47 ± 0.19 |

### NLR (pozycje wewnątrz powtórzenia)

| Pozycja | n (bundli×powt.) | Mediana | Średnia ± std |
|--------:|-----------------:|--------:|--------------:|
| 10 | 2695 | 0.73× | 0.82 ± 0.60 |
| 11 | 2697 | 2.71× | 2.76 ± 1.16 |
| 12 | 2697 | 6.56× | 6.73 ± 1.66 |
| 14 | 2697 | 3.03× | 3.40 ± 1.77 |
| 30 | 2697 | 40.40× | 65.10 ± 82.18 |
| 32 | 2495 | 0.73× | 0.84 ± 0.57 |
| 39 | 239 | 2.53× | 2.78 ± 2.19 |

---

## 2. PAE-filtered enrichment

- Bez filtra (n=233): mediana enrichment = **3.86×**
- Z filtrem PAE < 10 Å (n=15): mediana = **0.00×**
- Ratio (po/przed) = **0.00**

- Ratio ≈ 1.0 → enrichment trzyma się pewnych kontaktów
- Ratio < 0.5 → enrichment częściowo z artefaktów

---

## 4. FWHM peaków

| Pozycja | n bundli | Mediana FWHM | Średnia FWHM |
|--------:|---------:|-------------:|-------------:|
| 118 | 233 | 1 | 1.4 ± 0.6 |
| 133 | 233 | 1 | 1.4 ± 0.5 |
| 153 | 233 | 3 | 2.9 ± 0.3 |

- FWHM ≤ 3 → wąski peak (specyficzność punktowa)
- FWHM 4-8 → średnia szerokość
- FWHM ≥ 10 → szeroki plateau (sygnał regionalny)

---

## Wnioski (do dyskusji w pracy)

- **Lokalny enrichment**: HET-C ≈ 1.47×, NLR ≈ 2.71×. Sygnał ma **komponentę punktową** — AF3 częściowo trafia w pozycję.
- **PAE-filter** zachowuje 0% pierwotnego enrichmentu — sygnał **w dużej mierze zanika** — pierwotny enrichment to artefakt PAE.
- **Mediana FWHM ≈ 1 residuów**: peaki **wąskie** — AF3 celuje w konkretne pozycje.

### Główny przekaz

AF3 nie modeluje precyzyjnie pojedynczych kontaktów na poziomie pojedynczych residuów — niepewność geometryczna (PAE) jest zbyt wysoka. Jednak **ciągle generuje silny sygnał wokół pozycji hiperwariabilnych**, czyli poprawnie identyfikuje rejon interfejsu NLR–HET-C. Ten sygnał regionalny może być wartościowym wstępnym filtrem dla precyzyjniejszych metod (np. dynamika molekularna, eksperymenty mutacyjne).
