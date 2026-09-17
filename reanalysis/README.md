# TS-Fault Common-Reference Audit

This directory contains the standalone audit behind the published-aggregate section of the paper:

- [`tsfault_common_reference.py`](tsfault_common_reference.py)

## Source and transcription

The script transcribes three columns for all 21 models from Table V of
*TS-Fault: Benchmarking Time Series Forecasters Against Structural Faults*
(arXiv:2606.18539): Clean MSE, Faulted MSE, and the reported mean robustness
ratio `r`. The rows are kept inline so a reader can compare every number with
the source table without opening an intermediate binary file.

The Naive model is the trivial reference, with clean MSE `1.238` and faulted
MSE `129.5`.

## Calculation

The script constructs two rankings from the same printed mean aggregates:

1. **Own-clean-error ratio:** rank `Faulted MSE / Clean MSE`, computed from the two printed mean-error columns.
   The source’s reported `r` instead averages within-dataset ratios across six datasets and is retained solely for provenance.
2. **Common-reference skill:** rank `129.5 / Faulted MSE`, which scores every
   model against the same faulted Naive reference.

Thus the audit comparison changes the scoring reference while holding the
Table V mean aggregates fixed. It is deliberately not an exact reconstruction
of TS-Fault's published robustness rank, because that rank uses the median
relative degradation across configurations rather than the table's mean
ratio.

## Run and expected checks

```bash
python reanalysis/tsfault_common_reference.py
```

Expected final results:

- own-clean-error ratio: Spearman `rho = -0.414`, `p = 0.062`;
- common-reference skill: Spearman `rho = +0.052`, `p = 0.823`;
- Moirai, TimesFM, and Chronos fall below the common Naive reference.

`python tests/test_numbers.py` executes this audit and checks all three
outcomes. `python scripts/reproduce_all.py --tables-only` exposes the complete
21-model table in the one-command reproduction path.
