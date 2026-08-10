"""Shutdown detection and table correction for the TEP plant-side arms.

teprob.f zeroes every state derivative when the shutdown flag ISD trips, so a
shut-down plant yields a frozen but FINITE trajectory that an isfinite check
cannot catch.

Freeze detection uses the 22 XMEAS channels only. Measurement noise stops
once ISD trips so XMEAS goes exactly constant, while XMV keeps moving because
the PI controllers integrate a constant error against the frozen
measurements. An episode is invalid if the freeze starts before sample
T_ORIGIN + HORIZON.
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

ART = HERE / "artifacts"
OUT = HERE / "retention"

T_ORIGIN, T_ONSET = 300, 295
SPAN_END = T_ORIGIN + HORIZON          # 340
EVAL_SEEDS = list(range(4000, 4080))
SEEDS = [int(s) for s in
         os.environ.get("RBM_SEEDS", "42,123,7,2024,31337").split(",")]
N_BOOT = 1000

OBS_GRID = [0.0, 0.02, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.5, 0.7, 1.0,
            1.3, 1.6, 2.0]
PLANT = {
    "process_idv1": {"idv": 1,
                     "levels": [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]},
    "actuation_idv6": {"idv": 6,
                       "levels": [0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]},
    "dynamics_idv13": {"idv": 13,
                       "levels": [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]},
}

def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    with open(OUT / "shutdown_scan.log", "a") as f:
        f.write(line + "\n")

def freeze_onset(x):
    """First sample index where XMEAS is exactly frozen, or None.

    XMEAS columns only: post-shutdown, controllers keep integrating a
    constant error, so XMV drifts while the plant is frozen.
    """
    xm = x[:, :T.N_MEAS]
    same = np.all(xm[1:] == xm[:-1], axis=1)
    for t in range(len(same) - 1):
        if same[t] and same[t + 1]:
            return t
    return None

def main():
    norm = np.load(ART / "normalization.npz")
    shift, scale = norm["shift"], norm["scale"]
    ep = np.load(ART / "episodes.npz")
    nominal = {int(s): ep[f"eval_{s}"] for s in EVAL_SEEDS}

    n_froz = sum(freeze_onset(v) is not None for v in nominal.values())
    log(f"nominal eval episodes frozen: {n_froz}/80")

    jobs, keys = [], []
    for dom, spec in PLANT.items():
        for s in spec["levels"]:
            for seed in EVAL_SEEDS:
                idata = T.nominal_idata()
                idata[T_ONSET:, spec["idv"] - 1] = 1
                dsev = T.nominal_dsev()
                dsev[spec["idv"] - 1] = s
                jobs.append((idata, seed, dsev))
                keys.append((dom, s, seed))
    log(f"simulating {len(jobs)} plant-side arms (full grids)")
    t0 = time.time()
    outs = simulate_batch(jobs, workers=8)
    log(f"  done in {time.time()-t0:.0f}s")

    arms, freeze = {}, {}
    for k, x in zip(keys, outs):
        dom, s, seed = k
        xx = x[:, :T.N_MEAS + T.N_MV].astype(np.float32)
        if not np.isfinite(xx).all():
            freeze[k] = -1                      # non-finite: hard failure
            continue
        f0 = freeze_onset(xx)
        if f0 is not None and f0 < SPAN_END:
            freeze[k] = f0                      # shut down inside scoring span
        else:
            arms[k] = xx
    sd_table = {}
    for dom, spec in PLANT.items():
        for s in spec["levels"]:
            n_bad = sum(1 for seed in EVAL_SEEDS if (dom, s, seed) in freeze)
            sd_table[f"{dom}@{s}"] = n_bad
            if n_bad:
                log(f"  {dom} s={s}: {n_bad}/80 episodes shut down "
                    f"inside scoring span")
    with open(OUT / "shutdown_table.json", "w") as f:
        json.dump({"span_end_sample": SPAN_END,
                   "counts": sd_table,
                   "freeze_onsets": {f"{d}|{s}|{seed}": int(v)
                                     for (d, s, seed), v in freeze.items()}},
                  f, indent=2)
    log("wrote shutdown_table.json")

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
    MODELS = ["persistence", "ridge"] + list(nets)

    def windows_for(traj_by_seed, seeds, obs_op=None, s_obs=0.0):
        F, L, G = [], [], []
        for seed in seeds:
            x = traj_by_seed[seed]
            z = (x - shift) / scale
            h = z[T_ORIGIN - T.HIST:T_ORIGIN].copy()
            if obs_op == "bias" and s_obs > 0:
                h[:, :T.N_MEAS] += s_obs
            elif obs_op == "noise" and s_obs > 0:
                rng = np.random.RandomState(97 + seed)
                h[:, :T.N_MEAS] += rng.randn(T.HIST, T.N_MEAS) * s_obs
            F.append(h.reshape(-1))
            L.append(h[-1, :T.N_MEAS])
            G.append(z[T_ORIGIN:T_ORIGIN + HORIZON, :T.N_MEAS])
        return (np.asarray(F, np.float32), np.asarray(L, np.float32),
                np.asarray(G, np.float32))

    meas_scale = scale[:T.N_MEAS]

    def rmse_w(F, L, G, model):
        if model == "persistence":
            pred = np.repeat(L[:, None, :], HORIZON, axis=1)
        elif model == "ridge":
            inc = np.concatenate([F, np.ones((len(F), 1), np.float32)], 1) @ W
            pred = inc.reshape(-1, HORIZON, T.N_MEAS) + L[:, None, :]
        else:
            with torch.no_grad():
                inc = nets[model](torch.from_numpy(F)).numpy()
            pred = inc.reshape(-1, HORIZON, T.N_MEAS) + L[:, None, :]
        e = pred - G
        return (np.sqrt(np.mean(e ** 2, axis=(1, 2))),
                np.sqrt(np.mean((e * meas_scale) ** 2, axis=(1, 2))))

    tables = {"norm": {m: {} for m in MODELS},
              "phys": {m: {} for m in MODELS}}

    for op in ("bias", "noise"):
        for s in OBS_GRID:
            F, L, G = windows_for(nominal, EVAL_SEEDS, op, s)
            for m in MODELS:
                rn, rp = rmse_w(F, L, G, m)
                tables["norm"][m][(f"obs_{op}", s)] = rn
                tables["phys"][m][(f"obs_{op}", s)] = rp
    log("scored observation domains")

    for dom, spec in PLANT.items():
        for m in MODELS:
            tables["norm"][m][(dom, 0.0)] = tables["norm"][m][("obs_bias", 0.0)]
            tables["phys"][m][(dom, 0.0)] = tables["phys"][m][("obs_bias", 0.0)]
        for s in spec["levels"]:
            keep = [seed for seed in EVAL_SEEDS if (dom, s, seed) in arms]
            if len(keep) < len(EVAL_SEEDS) * 0.5:
                log(f"  {dom} s={s}: {len(keep)}/80 valid -> level censored "
                    f"(rule criterion 3), not scored")
                continue
            traj = {seed: arms[(dom, s, seed)] for seed in keep}
            F, L, G = windows_for(traj, keep)
            for m in MODELS:
                rn, rp = rmse_w(F, L, G, m)
                fn = np.full(len(EVAL_SEEDS), np.nan)
                fp = np.full(len(EVAL_SEEDS), np.nan)
                for i, seed in enumerate(keep):
                    fn[EVAL_SEEDS.index(seed)] = rn[i]
                    fp[EVAL_SEEDS.index(seed)] = rp[i]
                tables["norm"][m][(dom, s)] = fn
                tables["phys"][m][(dom, s)] = fp
    log("scored plant-side domains (shutdown-masked)")

    np.savez_compressed(
        OUT / "tables_v2.npz",
        **{f"{sp}|{m}|{d}|{s}": v for sp, tt in tables.items()
           for m, t in tt.items() for (d, s), v in t.items()})
    log("wrote tables_v2.npz")

    domains = {}
    for (d, s) in tables["norm"]["persistence"]:
        domains.setdefault(d, set()).add(s)
    domains = {d: sorted(v) for d, v in domains.items()}

    rng = np.random.default_rng(0)
    bidx = rng.integers(0, len(EVAL_SEEDS), size=(N_BOOT, len(EVAL_SEEDS)))

    def curve(mname, dom, idx=None):
        out = []
        for s in domains[dom]:
            a = tables["norm"]["persistence"][(dom, s)]
            b = tables["norm"][mname][(dom, s)]
            if idx is not None:
                a, b = a[idx], b[idx]
            ok = np.isfinite(a) & np.isfinite(b)
            out.append(np.mean(a[ok]) / np.mean(b[ok]))
        return np.array(out)

    def crossing(levels, c):
        for i in range(1, len(levels)):
            if c[i] < 1.0 <= c[i - 1]:
                return levels[i - 1] + (c[i - 1] - 1.0) / (c[i - 1] - c[i]) * \
                    (levels[i] - levels[i - 1])
        return None

    results = {"n_episodes": len(EVAL_SEEDS),
               "shutdown_counts": sd_table,
               "domains": {d: list(v) for d, v in domains.items()},
               "models": {}}
    for m in MODELS:
        if m == "persistence":
            continue
        mres = {}
        for dom, levels in domains.items():
            pt = curve(m, dom)
            boots = np.stack([curve(m, dom, bidx[b]) for b in range(N_BOOT)])
            ux = crossing(levels, pt)
            uxb = [crossing(levels, boots[b]) for b in range(N_BOOT)]
            uxv = [u for u in uxb if u is not None]
            mres[dom] = {
                "levels": list(levels), "skill": pt.tolist(),
                "ci_lo": np.percentile(boots, 2.5, axis=0).tolist(),
                "ci_hi": np.percentile(boots, 97.5, axis=0).tolist(),
                "utility_crossing": ux,
                "utility_crossing_ci": ([float(np.percentile(uxv, 2.5)),
                                         float(np.percentile(uxv, 97.5))]
                                        if len(uxv) > N_BOOT * 0.5 else None),
                "crossing_censored_frac": 1 - len(uxv) / N_BOOT,
            }
            log(f"  {m:10s} {dom:15s} min={pt.min():.2f} ux={ux}")
        results["models"][m] = mres

    with open(OUT / "results_v2.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    log("wrote results_v2.json")

if __name__ == "__main__":
    main()
