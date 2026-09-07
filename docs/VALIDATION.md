# Validation and Baseline Report

This research case study estimates airway cell-type proportions from bulk
RNA-seq. It is not a clinical model. The checks below distinguish performance
on simulated mixtures from agreement in an external cohort.

## Validation checks

| Check | Evidence | Interpretation |
|---|---:|---|
| Synthetic pseudo-bulk held-out split | Pearson r = 0.954, RMSE = 0.031 | The network can recover known mixtures generated from the Ziegler reference. |
| 5-fold synthetic pseudo-bulk CV | Pearson r = 0.954 +/- 0.001, RMSE = 0.032 | Low fold variance; the pseudo-bulk training setup is stable. |
| NNLS baseline on same validation data | Pearson r = 0.609 | The learned ensemble beats a simple linear deconvolution baseline on this task. |
| External cohort GSE163151 | Direction concordance 8/14, effect-size r = 0.057 | Partial biological replication only; cohort and gene-space mismatch remain large. |
| CI smoke tests | Pseudo-bulk generation, simplex output, summaries, plots, metadata | The public repo can be installed and exercised without the large raw datasets. |

## Baseline positioning

The closest conceptual baselines are CIBERSORTx, MuSiC, BayesPrism, Scaden, and
simple NNLS. Only the NNLS comparison is recorded here. The project has not established
superiority over the other methods on matched inputs.

The tissue-matched reference includes epithelial and immune cells, which fits
the scope of the airway question. Whether it improves accuracy over alternative
references requires a controlled comparison.

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

## Interpretation and next steps

The simulated-mixture results show that the model can recover proportions under
its training assumptions. Weak external effect-size agreement limits stronger
biological claims. Next steps are a matched comparison with an established
method and validation against independently measured cell proportions.
