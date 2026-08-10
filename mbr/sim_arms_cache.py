"""Simulate and cache all plant-side arms once, so scoring across seeds and
grids needs no further simulation.

Arms (same windows and semantics as rebuild_fingerprint.py):
  nominal, process +{10,25,50,75,100}%, actuation s in {0.05,0.1,0.2,0.35,0.5}

Output: fingerprint/arms_cache.npz with arrays (n_win, HORIZON, 66).

Usage:
    PYTORCH_ENABLE_MPS_FALLBACK=1 python sim_arms_cache.py
"""

import contextlib
import io
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
WW_ROOT = os.environ.get("MBR_PROJECT_ROOT")
if not WW_ROOT:
    raise SystemExit("set MBR_PROJECT_ROOT to the MBR project checkout; see README")
OUT = HERE / "fingerprint"

HIST, HORIZON = 20, 80
TEST_SEQS = list(range(30, 38))
STRIDE = 150
PROC = [10, 25, 50, 75, 100]
ACT = [0.05, 0.1, 0.2, 0.35, 0.5]

def main():
    os.chdir(WW_ROOT)
    sys.path.insert(0, WW_ROOT)
    OUT.mkdir(parents=True, exist_ok=True)
    import args_new as new_args
    from experiments.process_disturbances import load_model_and_data
    from utils.disturbance_handler import get_default_disturbance_handler

    args = dict(new_args.args)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _m, _td, env, _nm = load_model_and_data("mamba", "MBR", args)
    del _m, _td, _nm
    f = "save_model/mamba/MBR"
    su, cu = (np.loadtxt(f"{f}/{k}.txt", dtype=np.float32)
              for k in ("shift_u", "scale_u"))
    d = torch.load("dataset/MBR/0/draw.pt", weights_only=False)
    dh = get_default_disturbance_handler()
    dist = dh.sample_window(window_length=d[0][0].shape[0], start_idx=0,
                            normalized=False).astype(np.float32)[:, :10]

    wins = []
    for si in TEST_SEQS:
        st = d[si][0].numpy().astype(np.float32)
        ac = d[si][1].numpy().astype(np.float32)
        t = 0
        while t + HIST + HORIZON <= st.shape[0]:
            wins.append({"t": t, "fa": ac[t + HIST:t + HIST + HORIZON, :4].copy(),
                         "x0": st[t + HIST - 1].copy()})
            t += STRIDE
    print(f"{len(wins)} windows", flush=True)

    def sim(x0, acts, dmat):
        env.reset()
        env.p = dmat.copy()
        env.state = x0.copy()
        tr = np.zeros((HORIZON, x0.shape[0]), np.float32)
        for k in range(HORIZON):
            try:
                nx, _, _, _ = env.step(acts[k], k)
            except TypeError:
                nx, _, _, _ = env.step(acts[k])
            nx = np.clip(np.nan_to_num(np.asarray(nx, dtype=float), nan=0.0,
                                       posinf=1e6, neginf=-1e6), -1e6, 1e6)
            env.state = nx
            tr[k] = nx
        return tr

    def dslice(w):
        s = w["t"] + HIST - 1
        return dist[s:s + HORIZON + 1].copy()

    arrays = {}
    t0 = time.time()
    arrays["nominal"] = np.stack([sim(w["x0"], w["fa"], dslice(w)) for w in wins])
    print(f"nominal ({time.time()-t0:.0f}s)", flush=True)
    for pct in PROC:
        t0 = time.time()
        arrays[f"proc_{pct}"] = np.stack([
            sim(w["x0"], w["fa"], dslice(w) * (1 + pct / 100.0)) for w in wins])
        print(f"proc {pct} ({time.time()-t0:.0f}s)", flush=True)
    for s in ACT:
        t0 = time.time()
        outp = []
        for w in wins:
            an = (w["fa"] - su[:4]) / cu[:4]
            ad = an * (1.0 - s) * cu[:4] + su[:4]
            outp.append(sim(w["x0"], ad, dslice(w)))
        arrays[f"act_{s}"] = np.stack(outp)
        print(f"act {s} ({time.time()-t0:.0f}s)", flush=True)

    np.savez_compressed(OUT / "arms_cache.npz",
                        win_t=np.array([w["t"] for w in wins]), **arrays)
    print("wrote arms_cache.npz", flush=True)

if __name__ == "__main__":
    main()
