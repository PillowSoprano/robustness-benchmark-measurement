"""TEP prediction task: episodes, windows, normalization, splits.

Information contract: 20 samples (1 h) of XMEAS(1..22) and XMV(1..11) as
history, XMEAS(1..22) over the next HORIZON samples as target. No future
exogenous inputs, because the plant is closed loop and no future command
trajectory is known. This differs from the MBR task and is a property of the
system, not a defect.

IDV(8) feed-composition variation runs at dsev 1.0 throughout every episode:
a forecaster that only works with a perfectly still feed is not useful.

Splits are disjoint seed ranges declared before training. Targets are
increments from the last observed value normalized by the training std, so
persistence is exactly the zero prediction.
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from tep_sim import simulate_batch

N_MEAS = 22            # XMEAS(1..22), continuously sampled
N_MV = 11              # XMV(1..11)
HIST = 20
EP_SAMPLES = 480       # 24 h
BURN = 80              # 4 h
SEEDS = {"train": range(1000, 1200), "val": range(2000, 2040),
         "calib": range(3000, 3040), "eval": range(4000, 4080)}
NOMINAL_DSEV_IDV8 = 1.0

def nominal_idata(n=EP_SAMPLES):
    idata = np.zeros((n, 20), dtype=int)
    idata[:, 7] = 1          # IDV(8) on for the whole episode
    return idata

def nominal_dsev():
    d = np.ones(20)
    d[7] = NOMINAL_DSEV_IDV8
    return d

def gen_episodes(seeds, workers=8):
    jobs = [(nominal_idata(), s, nominal_dsev()) for s in seeds]
    outs = simulate_batch(jobs, workers=workers)
    keep = []
    for s, x in zip(seeds, outs):
        if np.isfinite(x).all():
            keep.append((int(s), x[:, :N_MEAS + N_MV].astype(np.float32)))
    return keep

def build_windows(eps, horizon, stride, shift, scale):
    """eps: list of (seed, (T, 33) array). Returns X, Y, last."""
    X, Y = [], []
    need = HIST + horizon
    for _, x in eps:
        z = (x - shift) / scale
        t = BURN
        while t + need <= z.shape[0]:
            h = z[t:t + HIST]                       # (HIST, 33)
            fut = z[t + HIST:t + HIST + horizon, :N_MEAS]
            X.append(h.reshape(-1))
            Y.append((fut - h[-1, :N_MEAS]).reshape(-1))
            t += stride
    return np.asarray(X, np.float32), np.asarray(Y, np.float32)

def fit_normalization(eps):
    allx = np.concatenate([x for _, x in eps])
    shift = allx.mean(0)
    scale = allx.std(0) + 1e-8
    return shift.astype(np.float32), scale.astype(np.float32)
