# September 2026 reference-sensitivity revision

This addition contains four-reference evaluation on the original MBR and TEP core panels. Persistence, history mean, endpoint drift and nominal-training AR(1) share each cell's target population and validity mask. Model weights are fixed. The full sensitivity record is in `REFERENCE_PROTOCOL.md`.

## Reproduce from frozen losses

With Python 3.10+ and NumPy:

```sh
python revision_2026_09/rescore.py
```

The script verifies SHA-256 hashes, recomputes the 2,000-draw paired cluster bootstrap, and requires exact agreement with the released JSON. The default also reproduces the separate complete clean-admission tasks (504 MBR windows and 1,440 TEP windows). It does not need a process simulator or model retraining to reproduce these scores. MBR groups windows by eight parent trajectories; TEP groups by 80 independent episode IDs. Four of 68 TEP cells are censored; all 52 MBR cells are evaluated.

`summary/` contains raw model/reference mean losses, skill curves, nominal admission, crossing brackets and conditional intervals, bootstrap no-crossing fractions, grid aggregates and the modern-model point estimates. The modern models use the original coarser 26-cell grid, without the second valve-sticking operator. Their archived aggregate curves cannot supply new episode-level confidence intervals.

## LaTeX figures

All revised curve plots are drawn by PGFPlots/TikZ. The `.dat` tables are numeric inputs and `.tex` files are editable standalone figure sources.

```sh
python revision_2026_09/build_pgf_figures.py
python revision_2026_09/build_gain_figure.py
cd revision_2026_09/latex_figures
tectonic -X compile reference_sensitivity.tex
tectonic -X compile skill_retention.tex
tectonic -X compile tep_retention_v2.tex
tectonic -X compile gain_sweep.tex
```

## Simulator correction

`tep/tep_sim.py` now uses spawned processes and explicit pool chunk size one. `maxtasksperchild=1` alone retired a process after a whole chunk and could allow stateful Fortran calls to affect one another. The physical regression check is:

```sh
python tests/test_replay_isolation.py
```

It requires the compiled Fortran extension in `tep/build` or on `PYTHONPATH`. The new reference losses were generated from independent replay and frozen checkpoints. The reference replay matches the corrected-retention TEP masks; the 56/64 channel-weighting and 8/64 window-aggregation counts remain unchanged. A separate legacy seed-summary merge overwrote nine masks and mixed window estimators; its precedence and masking are corrected in `tep/seed_spread.py`. Primary seed tables are regenerated from fresh losses, giving 22/64 core-order disagreements for five seeds under mean window RMSE (24/64 under pooled RMS). The prior archived files are retained for traceability; corrected reference comparisons use this directory's losses.

Full MBR physical re-simulation still requires the separately available plant environment and data described in the repository's existing data notes. Frozen losses support score reproduction, not a claim that the complete MBR simulator is redistributed here.

## Control-gain supporting study

`audit/gain_sweep.json` is the independently replayed, separately retrained four-gain study. `summary/gain_statistics.json` records each intersected severity grid and all plotted minima. `audit/gain_probe_audit.json` distinguishes shutdown over a full 24-hour episode from failure before forecast sample 340. Training and validation retain finite post-shutdown segments under the original design; evaluation uses forecast-span survival. These comparisons therefore include changes in training coverage and surviving populations. The gain figure is a point-estimate study at training seed 42.
