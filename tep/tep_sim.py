"""Seeded, process-isolated TEP simulation.

The Fortran plant keeps state in COMMON blocks across calls. Resetting the
exposed blocks (RANDSD, the CTRLn integrator states, FLAG6) fixes the first
repeat call but not later ones, so some further block still carries state.
Fresh processes with the same seed are bit-identical. Both maxtasksperchild=1
and chunksize=1 are required: a Pool task is a chunk, not necessarily one
simulation. Spawn also prevents inheriting Fortran state from the parent.

simulate(idata, seed, dsev=None, spoff=None, gainmul=1.0)
    -> (n_samples, 52) array of XMEAS(1..41) then XMV(1..11).

idata   (n_samples, 20) IDV switches per 3-min sample
dsev    (20,) severity multipliers on the IDV disturbance amplitudes
spoff   (20,) offsets added to SETPT(1..20) at t=0, for regime interventions
gainmul scalar on every controller gain; 1.0 is the original tuning and
        smaller values weaken the regulatory layer without retuning it
"""

import os
import sys
from multiprocessing import get_context, set_start_method

import numpy as np

BUILD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build")


def _worker(job):
    idata, seed, dsev, spoff, gmul = (*job, None, None, 1.0)[:5]
    sys.path.insert(0, BUILD)
    import temain_mod
    idata = np.asarray(idata, dtype=int)
    if dsev is None:
        dsev = np.ones(20)
    if spoff is None:
        spoff = np.zeros(20)
    if gmul is None:
        gmul = 1.0
    x = temain_mod.temain(60 * 3 * idata.shape[0], idata.shape[0],
                          idata, float(seed), np.asarray(dsev, dtype=float),
                          np.asarray(spoff, dtype=float), float(gmul), 1)
    return np.asarray(x)


def simulate(idata, seed, dsev=None, spoff=None, gainmul=1.0):
    with get_context("spawn").Pool(processes=1, maxtasksperchild=1) as p:
        return p.map(_worker, [(idata, seed, dsev, spoff, gainmul)], chunksize=1)[0]


def simulate_batch(jobs, workers=8):
    with get_context("spawn").Pool(processes=workers, maxtasksperchild=1) as p:
        return p.map(_worker, jobs, chunksize=1)


if __name__ == "__main__":
    try:
        set_start_method("spawn")
    except RuntimeError:
        pass
    idata = np.zeros((40, 20), dtype=int)
    a = simulate(idata, 12345)
    b = simulate(idata, 12345)
    c = simulate(idata, 99999)
    print("same seed identical:", np.array_equal(a, b))
    print("diff seed differs:", not np.array_equal(a, c))
    import time
    t0 = time.time()
    outs = simulate_batch([(idata, 1000 + i) for i in range(16)], workers=8)
    dt = time.time() - t0
    print(f"batch of 16 x 2h runs: {dt:.1f}s "
          f"({dt/16:.2f}s amortized), all finite:",
          all(np.isfinite(o).all() for o in outs))
