"""Control-authority sweep: does weakening the regulatory layer make the TEP
disturbance domains yield the utility crossings they withhold at nominal gain?

The claim under test is that TEP's disturbance domains produce no crossing
because the controllers absorb those interventions before they reach the
measured trajectory. Scaling every controller gain moves that one variable
with chemistry, dimensionality, task, and panel held fixed, which two
different plants cannot do.

Design:
  same episode seeds at every gain, so the arms are paired across the sweep;
  the panel is retrained at each gain, otherwise weakened control would be
  confounded with a shifted training distribution;
  MLP and GRU use seed 42 alone because this is a paired mechanism diagnostic,
  not a seed-robust ranking; seed sensitivity is reported separately;
  severity grids are reduced to the levels that decide whether the minimum
  crosses 1, keeping the maxima from the full evaluation.

Gain 1.0 is included as a control: it must reproduce the published minima.
Viability limits the sweep to [0.7, 1.0]; below 0.6 most episodes trip the
shutdown interlock inside the scoring span (gain_probe.py).
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tep_task as T
from tep_sim import simulate_batch
from train_tep_panel import MLP, Seq2Seq, HORIZON, STRIDE_TR, fit
from shutdown_scan import freeze_onset

OUT = HERE / "gain_sweep"
GAINS = [1.0, 0.9, 0.8, 0.7]
T_ORIGIN, T_ONSET, SPAN_END = 300, 295, 340
EVAL_SEEDS = list(range(4000, 4080))
TRAIN_SEEDS = list(range(1000, 1200))
VAL_SEEDS = list(range(2000, 2040))
SEED = 42

OBS_GRID = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]
PLANT = {"process_idv1": (1, [0.5, 1.0, 2.0]),
         "actuation_idv6": (6, [0.35, 0.75, 1.0]),
         "dynamics_idv13": (13, [1.0, 3.0, 6.0])}


def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "run.log", "a") as f:
        f.write(line + "\n")


def episodes(seeds, gain):
    jobs = [(T.nominal_idata(), s, T.nominal_dsev(), None, gain) for s in seeds]
    out = []
    for s, x in zip(seeds, simulate_batch(jobs, workers=8)):
        x = np.asarray(x)
        if np.isfinite(x).all():
            out.append((int(s), x[:, :T.N_MEAS + T.N_MV].astype(np.float32)))
    return out


def valid(x):
    f0 = freeze_onset(x)
    return np.isfinite(x).all() and (f0 is None or f0 >= SPAN_END)


def main():
    results = {}
    for gain in GAINS:
        log(f"=== gain {gain} ===")
        t0 = time.time()
        tr = episodes(TRAIN_SEEDS, gain)
        va = episodes(VAL_SEEDS, gain)
        ev = [(s, x) for s, x in episodes(EVAL_SEEDS, gain) if valid(x)]
        log(f"  episodes train {len(tr)} val {len(va)} eval {len(ev)}/80 "
            f"({time.time()-t0:.0f}s)")

        shift, scale = T.fit_normalization(tr)
        Xtr, Ytr = T.build_windows(tr, HORIZON, STRIDE_TR, shift, scale)
        Xva, Yva = T.build_windows(va, HORIZON, STRIDE_TR, shift, scale)
        din, dout = Xtr.shape[1], Ytr.shape[1]

        lam = 10.0
        A = np.concatenate([Xtr, np.ones((len(Xtr), 1), np.float32)], 1)
        W = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1], dtype=np.float32),
                            A.T @ Ytr)
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        mlp = fit(MLP(din, dout), Xtr, Ytr, Xva, Yva, 60, 256, 1e-3, "mlp")
        gru = fit(Seq2Seq(), Xtr, Ytr, Xva, Yva, 60, 256, 1e-3, "gru")
        mlp.eval(); gru.eval()
        log(f"  panel trained ({time.time()-t0:.0f}s)")

        keep = [s for s, _ in ev]
        nom = {s: x for s, x in ev}

        arms, jobs, keys = {}, [], []
        for dom, (idv, levels) in PLANT.items():
            for s in levels:
                idata = T.nominal_idata()
                idata[T_ONSET:, idv - 1] = 1
                dsev = T.nominal_dsev()
                dsev[idv - 1] = s
                for sd in keep:
                    jobs.append((idata, sd, dsev, None, gain))
                    keys.append((dom, s, sd))
        for k, x in zip(keys, simulate_batch(jobs, workers=8)):
            x = np.asarray(x)[:, :T.N_MEAS + T.N_MV].astype(np.float32)
            if valid(x):
                arms[k] = x
        log(f"  intervention arms {len(arms)}/{len(jobs)} "
            f"({time.time()-t0:.0f}s)")

        meas = scale[:T.N_MEAS]

        def score(traj, seeds, op=None, sev=0.0):
            F, L, G = [], [], []
            for sd in seeds:
                z = (traj[sd] - shift) / scale
                h = z[T_ORIGIN - T.HIST:T_ORIGIN].copy()
                if op == "bias" and sev > 0:
                    h[:, :T.N_MEAS] += sev
                elif op == "noise" and sev > 0:
                    rng = np.random.RandomState(97 + sd)
                    h[:, :T.N_MEAS] += rng.randn(T.HIST, T.N_MEAS) * sev
                F.append(h.reshape(-1)); L.append(h[-1, :T.N_MEAS])
                G.append(z[T_ORIGIN:T_ORIGIN + HORIZON, :T.N_MEAS])
            F = np.asarray(F, np.float32); L = np.asarray(L, np.float32)
            G = np.asarray(G, np.float32)
            out = {}
            for name in ("persistence", "ridge", "mlp", "gru"):
                if name == "persistence":
                    pred = np.repeat(L[:, None, :], HORIZON, axis=1)
                elif name == "ridge":
                    inc = np.concatenate(
                        [F, np.ones((len(F), 1), np.float32)], 1) @ W
                    pred = inc.reshape(-1, HORIZON, T.N_MEAS) + L[:, None, :]
                else:
                    with torch.no_grad():
                        inc = (mlp if name == "mlp" else gru)(
                            torch.from_numpy(F)).numpy()
                    pred = inc.reshape(-1, HORIZON, T.N_MEAS) + L[:, None, :]
                out[name] = np.sqrt(np.mean((pred - G) ** 2, axis=(1, 2)))
            return out

        g = {}
        for op in ("bias", "noise"):
            for sev in OBS_GRID:
                r = score(nom, keep, op, sev)
                g[(f"obs_{op}", sev)] = {m: r["persistence"].mean() / r[m].mean()
                                         for m in ("ridge", "mlp", "gru")}
        for dom, (_, levels) in PLANT.items():
            r0 = score(nom, keep)
            g[(dom, 0.0)] = {m: r0["persistence"].mean() / r0[m].mean()
                             for m in ("ridge", "mlp", "gru")}
            for s in levels:
                sub = [sd for sd in keep if (dom, s, sd) in arms]
                if len(sub) < len(keep) * 0.5:
                    log(f"  {dom} s={s}: {len(sub)}/{len(keep)} valid, censored")
                    continue
                r = score({sd: arms[(dom, s, sd)] for sd in sub}, sub)
                g[(dom, s)] = {m: r["persistence"].mean() / r[m].mean()
                               for m in ("ridge", "mlp", "gru")}

        doms = sorted({d for d, _ in g})
        res = {"n_eval": len(keep), "n_arms": len(arms), "curves": {}}
        for d in doms:
            levels = sorted(s for dd, s in g if dd == d)
            res["curves"][d] = {"levels": levels,
                                **{m: [g[(d, s)][m] for s in levels]
                                   for m in ("ridge", "mlp", "gru")}}
            mins = {m: min(g[(d, s)][m] for s in levels)
                    for m in ("ridge", "mlp", "gru")}
            log(f"  {d:16s} min skill " +
                " ".join(f"{m}={v:.2f}" for m, v in mins.items()) +
                ("   CROSSES" if min(mins.values()) < 1 else ""))
        results[str(gain)] = res
        log(f"  gain {gain} done ({time.time()-t0:.0f}s)")

    with open(OUT / "gain_sweep.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    log("wrote gain_sweep.json")


if __name__ == "__main__":
    main()
