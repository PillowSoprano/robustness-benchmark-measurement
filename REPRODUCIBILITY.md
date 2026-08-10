# Released reproducibility scope

Two released tiers separate artifact-level reproduction from simulator-level
reproduction.

| Tier | Needs | Time | Reproduces |
|---|---|---|---|
| 1 | Python and `requirements.txt` | under a minute | all artifact-backed released figures, tables, and checks |
| 2 | tier 1 plus a Fortran compiler | about an hour | all TEP results from the simulator up |

## Tier 1

```bash
python scripts/reproduce_all.py
```

Verifies `frozen_predictions/MANIFEST.json` checksums, then regenerates the
figures into `reproduced/`. A figure rebuilt from a silently edited artifact
would be worse than no figure, so a checksum mismatch stops the run.

`frozen_predictions/` holds eleven checksummed artifacts (about 175 KiB)
covering the Tennessee Eastman system, the CSTR admission sweep, and the
published-benchmark re-ranking. Everything downstream of those numbers—
curves, crossings, rankings, and tables—is recomputed rather than trusted.

The CSTR is the exception that needs nothing committed at all. Its equations
and parameters are in `cstr/cstr_system.py`, so
`cstr/horizon_excitation_sweep.py` regenerates its own artifact from scratch
in about fifteen minutes with no external data.

## Tier 2

Build the simulator, then run the TEP pipeline in the order in `README.md`.
The Fortran carries four patches over the original release: a settable seed,
severity multipliers, setpoint offsets, and a controller-gain multiplier.
Each defaults to reproducing the original behaviour, and each is marked
`PATCH` in the source.

Two determinism facts a reimplementation needs. Every simulation runs in a
fresh worker process, because the plant keeps state in `COMMON` blocks that
resetting the exposed blocks does not clear. And a shut-down plant yields a
frozen but finite trajectory, detectable only on the measurement channels,
since the controllers keep integrating against frozen measurements.

## What is committed and what is generated

| | |
|---|---|
| committed | source, configs, `frozen_predictions/`, licences |
| generated | `reproduced/`, TEP episodes, trained checkpoints, intervention arms |

Raw trajectories and trained checkpoints are not committed.
`frozen_predictions/` contains the compact released results needed for the
artifact-level reproduction.

## Two tests, run them first

```bash
python tests/test_sync.py      # manifest coverage, checksums, nothing stray
python tests/test_numbers.py   # the paper's numbers against the artifacts
```

`test_numbers.py` carries the paper's claimed values as literals and reads the
committed artifacts for the other side of each comparison. It is the check
that a regenerated artifact still supports the sentence it was cited for.

## Simulator licence

The TEP Fortran is distributed under the University of Illinois licence in
`tep/build/`; its citation requirements carry over to users. The f2py
interface is MIT, from `tep2py`.

## Checks

```bash
python tests/test_sync.py          # manifest coverage, checksums, nothing stray
python scripts/reproduce_all.py    # tier 1, end to end
```

`tests/test_sync.py` reads the Git index, verifies complete manifest coverage
and checksums, and rejects tracked build products or absolute home paths.
`tests/test_numbers.py` independently checks the manuscript's load-bearing
values against the released artifacts.
