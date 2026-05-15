# Validation and Baseline Report

This repo is a research-grade deconvolution case study, not a clinical model.
The useful portfolio claim is that it frames the biological question correctly,
uses a tissue-matched reference, records its model contract, and states where
the evidence stops.

## Validation ladder

| Check | Evidence | Interpretation |
|---|---:|---|
| Synthetic pseudo-bulk held-out split | Pearson r = 0.954, RMSE = 0.031 | The network can recover known mixtures generated from the Ziegler reference. |
| 5-fold synthetic pseudo-bulk CV | Pearson r = 0.954 +/- 0.001, RMSE = 0.032 | Low fold variance; the pseudo-bulk training setup is stable. |
| NNLS baseline on same validation data | Pearson r = 0.609 | The learned ensemble beats a simple linear deconvolution baseline on this task. |
| External cohort GSE163151 | Direction concordance 8/14, effect-size r = 0.057 | Partial biological replication only; cohort and gene-space mismatch remain large. |
| CI smoke tests | Pseudo-bulk generation, simplex output, summaries, plots, metadata | The public repo can be installed and exercised without the large raw datasets. |

## Baseline positioning

The closest conceptual baselines are CIBERSORTx, MuSiC, BayesPrism, Scaden, and
simple NNLS. This repo is strongest against weak baselines because it uses the
right tissue reference and records an NNLS comparison. It is not yet a full
benchmark against the leading deconvolution packages.

The original Lieberman et al. COVID airway analysis used CIBERSORTx with LM22, a
blood immune reference. That can estimate immune infiltration but cannot resolve
nasopharyngeal epithelial remodelling. This repo's stronger claim is therefore
not "new best deconvolution method"; it is "better biological reference for this
tissue and question".

## Statistical controls

`deconvolve.py` writes `mean_proportions_by_condition.csv` with:

- mean COVID-positive and negative proportions,
- positive-minus-negative difference,
- Mann-Whitney U p-value,
- Benjamini-Hochberg q-value across tested cell types,
- `significant` flag at q < 0.05.

This makes the multiple-testing correction explicit instead of relying on raw
p-values in downstream interpretation.

## Current gaps

- No real paired bulk/scRNA-seq ground truth for GSE152075.
- No committed full benchmark against MuSiC, BayesPrism, Scaden, or CIBERSORTx.
- Single scRNA-seq reference cohort; multi-reference robustness is not proven.
- External validation partially replicates direction, but effect-size correlation
  is weak.
- Large raw datasets and trained weights are not committed, so full reruns still
  require local data setup.

## Portfolio read

This is a credible applied computational-biology repo because it links bulk
RNA-seq, single-cell reference data, neural deconvolution, baseline comparison,
external validation, figures, and CI. The next highest-signal upgrade would be a
small committed benchmark table comparing this model against NNLS plus at least
one established package on the same pseudo-bulk split.
