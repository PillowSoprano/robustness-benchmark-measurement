# Input provenance

This file records the input contract for each adapter and identifies which
compact artifacts are part of the release.

## Tennessee Eastman

The plant is the original Fortran, patched four ways and vendored under
`tep/build/` with its licences. Episodes are simulator output and are
regenerated rather than shipped:

```bash
python tep/tep_task.py --build-episodes
```

A seeded run reproduces a trajectory bit for bit, so a regenerated episode
set is the same episode set. `frozen_predictions/tep/` holds the per-episode
errors these runs produced, which is what the figures and tables are built
from.

## CSTR

Self-contained. `cstr/cstr_system.py` carries the equations and the
parameters, so `cstr/horizon_excitation_sweep.py` runs with no external
data. Its result is committed at
`frozen_predictions/cstr/horizon_excitation_sweep.json`.

## Membrane bioreactor adapter

The scripts under `mbr/` define the data split, transformations, intervention
interfaces, and evaluation procedures used by that adapter. Raw plant inputs
are not part of this source release.
