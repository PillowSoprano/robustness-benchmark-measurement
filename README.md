<h1 align="center">Measurement Choices Decide Robustness Benchmarks</h1>
> September 2026 revision: the updated four-reference and clean-admission analyses, corrected seed statistics and editable LaTeX figures are in [`revision_2026_09/`](revision_2026_09/README.md). Use `python revision_2026_09/rescore.py` for the revised results. The original release and its reproduction entry point remain available below.


<p align="center">
  Experiment code for the paper's two systems, seven measurement choices, and
  cross-system induction table.
</p>

<p align="center">
  <a href="#what-is-here">What Is Here</a>
  ·
  <a href="#install">Install</a>
  ·
  <a href="#run-the-tep-experiments">TEP</a>
  ·
  <a href="#run-the-cstr-experiment">CSTR</a>
  ·
  <a href="#run-the-mbr-experiments">MBR</a>
  ·
  <a href="#audit-the-ts-fault-re-ranking">TS-Fault audit</a>
  ·
  <a href="#figures">Figures</a>
  ·
  <a href="#licences">Licences</a>
</p>

---

## What Is Here

The experiment code, plus the released per-window errors it produced. With
those committed, the released figures, tables, and benchmark re-ranking
regenerate in under a minute with no simulator or plant data:

```bash
python scripts/reproduce_all.py
```

`REPRODUCIBILITY.md` defines the released reproduction tiers and their
dependencies. The manuscript is distributed separately.

| Directory | Contents |
| --- | --- |
| `tep/` | Tennessee Eastman: patched Fortran simulator, task, panel, seven-operator evaluation, and the claim-testing probes |
| `mbr/` | Membrane bioreactor adapter and analysis implementation |
| `cstr/` | CSTR admission study and implementation-validation record |
| `reanalysis/` | Re-ranks a released benchmark's 21 models against a common reference, from its printed aggregates |
| `frozen_predictions/` | Eleven released artifacts (about 175 KiB), all checksummed |
| `scripts/` | One-command validation and regeneration of released outputs |
| `tests/` | Manifest integrity, and the paper's numbers against the artifacts |
| `data/` | Input provenance and adapter contracts |
| `figstyle.py` | Shared figure style, colour semantics, and save routine |

The three adapters are independent. The released TEP and CSTR workflows run
from this repository alone.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python tests/test_sync.py && python tests/test_numbers.py
```

`conda env create -f environment.yml` does the same and brings a Fortran
compiler, which only tier 2 needs.

Run the two tests before anything else. `test_sync.py` checks that every
committed artifact is checksummed and that no build product or absolute home
path made it into the repository. `test_numbers.py` compares the manuscript's
stated numbers against the committed artifacts and detects manuscript–artifact
drift before release.

Python 3.10. Rebuilding the TEP simulator additionally needs a Fortran
compiler.

## Run the TEP Experiments

### 1. Build the simulator

The Fortran sources in `tep/build/` are the original TEP code with four
additions: a settable random seed, continuous severity multipliers on the
disturbance amplitudes, setpoint offsets for regime interventions, and a
scalar multiplier on every controller gain. Each is marked `PATCH` in the
source, and each defaults to reproducing the original behaviour exactly.

```bash
cd tep/build
python -m numpy.f2py -c temain_mod-smart.pyf temain_mod.f teprob.f
```

Check that the seed took effect. Two runs at one seed must be bit-identical,
and two seeds must differ:

```bash
cd .. && python tep_sim.py
```

Every simulation runs in a fresh worker process. The plant keeps state in
COMMON blocks across calls, and resetting the exposed blocks is not enough to
make repeat calls reproducible.

### 2. Task and panel

```bash
python probe_horizon.py       # picks the forecast horizon
python train_tep_panel.py     # persistence, ridge, MLP, GRU at five seeds
```

The seed list is `RBM_SEEDS`, default `42,123,7,2024,31337`. Five rather
than three because this project's own results are about seed sensitivity, and
three draws show that variability exists without bounding it: going from
three seeds to five moved the widest domain-minimum spread from 0.155 to
0.773 and the count of cells whose member ordering disagrees from 8 of 64 to
27 of 64.

`train_tep_panel.py` applies the skill gate. A panel whose linear and
nonlinear members do not clear a bootstrap lower bound above 1 fails, and the
task instance is invalid rather than the models being ranked anyway.

### 3. Interventions

Run in this order; each step consumes the previous step's tables.

```bash
python tep_retention_eval.py       # five-domain retention curves
python tep_severity_extension.py   # severity extension by the declared rule
python shutdown_scan.py            # detect frozen episodes, rebuild the tables
python tep_regime_actuation2.py    # regime domain, second actuation operator
python tep_tau_abs_check.py        # per-channel absolute admissibility
python tep_seven_checks_v2.py      # gate semantics and both aggregation axes
python gain_probe.py               # viability limit of the gain sweep
python gain_sweep.py               # control-authority sweep, panel retrained per gain
python plot_gain_sweep.py          # verdict table and figure
python tep_modern_panel.py         # DecompLinear and PatchAttn through the same protocol
python probe_history.py            # does a longer history window pay on this task
```

Three of these test a claim instead of producing a ranking, and each prints
its verdict.

`gain_sweep.py` scales every controller gain with the plant, task, and panel
architecture fixed, retrains the panel at each setting, and asks whether the
disturbance domains yield crossings as regulation weakens. They move the
other way, which is the result.

`tep_modern_panel.py` adds a decomposition linear model and a patch-attention
model to the panel and asks whether the retention shapes belong to the task
or to the original four architectures. All five behave alike.

`probe_history.py` asks whether a history window long enough for a zero-shot
forecaster pays on this task. It declares its abandon criterion before
running and reports the verdict; on this task the answer is no.

`shutdown_scan.py` matters more than its name suggests. When the plant trips
its interlock the simulator zeroes every state derivative, so a shut-down
episode produces a frozen trajectory that is still finite and still gets
scored. Skipping this step silently mixes dead plants into the results.

Runtime on eight cores: about 7 minutes per full plant-side sweep, a few
seconds for the scoring.

## Run the CSTR Experiment

The CSTR needs no external data and no compiler. It is retained as a
documented admission study: implementation validation revised two protocol
assumptions before any predictor comparison. `cstr/CARD_DEVIATIONS.md`
records the evidence and corrections.

```bash
python cstr/horizon_excitation_sweep.py   # about fifteen minutes
```

Six forecast horizons and four excitation levels. No configuration produces a
panel that clears the skill gate by a margin supporting a robustness
comparison, and the three trained members are near-ties at every one of them.

## Audit the TS-Fault Re-ranking

The manuscript reanalysis of all 21 TS-Fault models is a standalone, directly
inspectable script:

[`reanalysis/tsfault_common_reference.py`](reanalysis/tsfault_common_reference.py)

Its only inputs are the Clean MSE, Faulted MSE, and robustness-ratio columns
transcribed from Table V of TS-Fault (arXiv:2606.18539). The own-error ratio is recomputed as faulted mean MSE divided by clean mean MSE; the source’s printed r instead averages within-dataset ratios. Both reconstructed
rankings use those same mean aggregates. The comparison changes the scoring
reference from each model's own clean error to the faulted error of the Naive
baseline; it does not claim to reproduce TS-Fault's published robustness rank,
which uses median relative degradation across configurations.

```bash
python reanalysis/tsfault_common_reference.py
```

The final lines report Spearman correlations of `-0.414` for the
own-clean-error ratio and `+0.052` for common-reference skill. The same audit
runs as Table 6 under `python scripts/reproduce_all.py --tables-only`.
[`reanalysis/README.md`](reanalysis/README.md) records the source columns,
calculation, expected output, and scope boundary.

## Figures

```bash
python tep/plot_tep_retention_v2.py
python tep/plot_gain_sweep.py
```

Each writes PDF, SVG, and PNG side by side. Two conventions in
`figstyle.py` are worth keeping if you edit them.

Figures are drawn at final print size, `COL` and `TEXT`, and included with no
scaling. A figure authored wide and scaled down shrinks every glyph by the
same factor, which is how a nominal 9 pt tick label ends up under 4 pt.

Colour carries meaning. Persistence is grey in every figure because it is the
uninformed baseline, and each learned model keeps one colour and one marker
throughout. The palette is Paul Tol's bright scheme, and the distinct markers
keep the figures readable in greyscale.

## Licences

`tep/build/temain_mod.f` and `tep/build/teprob.f` are modified from the
Tennessee Eastman code released by the Large Scale Systems Research
Laboratory, University of Illinois; `tep/build/LICENSE-tep-fortran` applies
and its citation requirements carry over. The f2py interface derives from
`tep2py` by Maurício Melo Câmara under the MIT licence,
`tep/build/LICENSE-tep2py`.

Our additions to those files are four: the seed argument, the `DSEV(20)`
severity multipliers carried in `COMMON/DSEVC/`, the `SPOFF(20)` setpoint
offsets, and the `GAINMUL` controller-gain multiplier. Each is marked `PATCH`
in the source and each defaults to reproducing the original behaviour.

Everything else in this repository is MIT, `LICENSE`.
