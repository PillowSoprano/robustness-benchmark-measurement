"""Three checks on the TEP regime versus plant-side contrast, core panel.

A. regime_fine
   Reactor temperature setpoint offsets of 0.5, 1.0, 1.5 and 2.0 C
   (s = 0.05 ... 0.20), applied at t = 0 as in the regime domain.
B. onset_t0
   The four plant-side operators (IDV 1, 6, 13, 14) switched on at t = 0
   instead of 15 min before the forecast origin, on their original grids, so
   history and target lie 15 to 17 h after onset, like the regime arms.
C. grid_extension
   IDV(13) and IDV(14) continued past their archived endpoints by the
   doubling rule, with the original onset: IDV(13) at 12 and 24, IDV(14) at
   16, 32 and 50 (dead band 100%, the valve no longer moves).

Scoring follows the primary analysis: one window per episode, history
[280, 300), forecast [300, 340), normalised channels, skill = mean window
RMSE of persistence / mean window RMSE of the model over surviving episodes.
Episodes frozen before sample 340 are dropped; a cell with under 50%
surviving episodes is censored. Checkpoints are the saved core panel; ridge
has no training seed.

Usage:
    python tep_r2_checks.py
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tep_task as T
from tep_sim import simulate_batch
from train_tep_panel import MLP, Seq2Seq, HORIZON
from shutdown_scan import freeze_onset

ART = HERE / "artifacts"
OUT = HERE / "retention"
T_ORIGIN, T_ONSET = 300, 295
SPAN_END = T_ORIGIN + HORIZON
EVAL_SEEDS = list(range(4000, 4080))
SEEDS = [int(s) for s in
         os.environ.get("RBM_SEEDS", "42,123,7,2024,31337").split(",")]

PLANT = {"process_idv1": (1, [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]),
         "actuation_idv6": (6, [0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]),
         "dynamics_idv13": (13, [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]),
         "actuation_idv14": (14, [0.5, 1.0, 2.0, 3.0, 5.0, 8.0])}
REGIME_FINE = [0.05, 0.1, 0.15, 0.2]
EXTEND = {"dynamics_idv13": (13, [12.0, 24.0]),
          "actuation_idv14": (14, [16.0, 32.0, 50.0])}


def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    with open(OUT / "r2_checks.log", "a") as f:
        f.write(line + "\n")


def main():
    norm = np.load(ART / "normalization.npz")
    shift, scale = norm["shift"], norm["scale"]
    ep = np.load(ART / "episodes.npz")
    nominal = {int(s): ep[f"eval_{s}"] for s in EVAL_SEEDS}

    jobs, keys = [], []
    for s in REGIME_FINE:
        spoff = np.zeros(20)
        spoff[17] = s * 10.0
        for seed in EVAL_SEEDS:
            jobs.append((T.nominal_idata(), seed, T.nominal_dsev(), spoff))
            keys.append(("regime_fine", "regime_setpt18", s, seed))
    for dom, (idv, levels) in PLANT.items():
        for s in levels:
            idata = T.nominal_idata()
            idata[0:, idv - 1] = 1
            dsev = T.nominal_dsev()
            dsev[idv - 1] = s
            for seed in EVAL_SEEDS:
                jobs.append((idata, seed, dsev))
                keys.append(("onset_t0", dom, s, seed))
    for dom, (idv, levels) in EXTEND.items():
        for s in levels:
            idata = T.nominal_idata()
            idata[T_ONSET:, idv - 1] = 1
            dsev = T.nominal_dsev()
            dsev[idv - 1] = s
            for seed in EVAL_SEEDS:
                jobs.append((idata, seed, dsev))
                keys.append(("grid_extension", dom, s, seed))
    log(f"simulating {len(jobs)} arms")
    t0 = time.time()
    outs = simulate_batch(jobs, workers=8)
    arms = {}
    for k, x in zip(keys, outs):
        xx = np.asarray(x)[:, :T.N_MEAS + T.N_MV].astype(np.float32)
        if not np.isfinite(xx).all():
            continue
        f0 = freeze_onset(xx)
        if f0 is not None and f0 < SPAN_END:
            continue
        arms[k] = xx
    log(f"  done in {time.time()-t0:.0f}s")

    din = T.HIST * (T.N_MEAS + T.N_MV)
    dout = HORIZON * T.N_MEAS
    W = np.load(ART / "ridge_W.npy")
    nets = {}
    for sd in SEEDS:
        m = MLP(din, dout)
        m.load_state_dict(torch.load(ART / f"seed_{sd}" / "mlp.pt"))
        m.eval()
        g = Seq2Seq()
        g.load_state_dict(torch.load(ART / f"seed_{sd}" / "gru.pt"))
        g.eval()
        nets[f"mlp_s{sd}"] = m
        nets[f"gru_s{sd}"] = g
    MODELS = ["ridge"] + list(nets)

    def windows(trajs):
        F, L, G = [], [], []
        for x in trajs:
            z = (x - shift) / scale
            h = z[T_ORIGIN - T.HIST:T_ORIGIN]
            F.append(h.reshape(-1))
            L.append(h[-1, :T.N_MEAS])
            G.append(z[T_ORIGIN:T_ORIGIN + HORIZON, :T.N_MEAS])
        return (np.asarray(F, np.float32), np.asarray(L, np.float32),
                np.asarray(G, np.float32))

    def rmse(F, L, G, model):
        if model == "persistence":
            pred = np.repeat(L[:, None, :], HORIZON, axis=1)
        elif model == "ridge":
            inc = np.concatenate([F, np.ones((len(F), 1), np.float32)], 1) @ W
            pred = inc.reshape(-1, HORIZON, T.N_MEAS) + L[:, None, :]
        else:
            with torch.no_grad():
                inc = nets[model](torch.from_numpy(F)).numpy()
            pred = inc.reshape(-1, HORIZON, T.N_MEAS) + L[:, None, :]
        return np.sqrt(np.mean((pred - G) ** 2, axis=(1, 2)))

    def score(trajs):
        F, L, G = windows(trajs)
        rp = rmse(F, L, G, "persistence").mean()
        return {m: float(rp / rmse(F, L, G, m).mean()) for m in MODELS}

    result = {"method": {"seeds": SEEDS, "eval_episodes": len(EVAL_SEEDS),
                         "window": [T_ORIGIN - T.HIST, T_ORIGIN + HORIZON],
                         "skill": "mean window RMSE ratio, persistence/model",
                         "censor_below_survival": 0.5},
              "nominal": score([nominal[s] for s in EVAL_SEEDS]),
              "cells": []}
    cells = sorted({k[:3] for k in keys}, key=lambda k: (k[0], k[1], k[2]))
    for exp, dom, s in cells:
        keep = [seed for seed in EVAL_SEEDS if (exp, dom, s, seed) in arms]
        row = {"experiment": exp, "domain": dom, "severity": s,
               "surviving": len(keep)}
        if len(keep) < 0.5 * len(EVAL_SEEDS):
            row["censored"] = True
        else:
            row["skill"] = score([arms[(exp, dom, s, seed)] for seed in keep])
            row["min_skill"] = min(row["skill"].values())
        result["cells"].append(row)
        log(f"  {exp:15s} {dom:16s} s={s:<5} survive {len(keep):2d}/80 "
            + ("censored" if row.get("censored") else
               f"min {row['min_skill']:.3f} ridge {row['skill']['ridge']:.3f}"))

    (OUT / "r2_checks.json").write_text(json.dumps(result, indent=2))
    log("wrote r2_checks.json")


if __name__ == "__main__":
    main()
