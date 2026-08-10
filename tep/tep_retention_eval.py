"""TEP skill-retention evaluation across the frozen intervention map.

Intervention map (mechanism-defined, frozen here):
  obs_bias   sustained additive bias on all measured history channels,
             severity s in per-channel training-std units; plant untouched
  obs_noise  gaussian noise on measured history, sd = s std units
  process    IDV(1) A/C feed-ratio step at severity dsev = s
  actuation  IDV(6) A-feed loss at severity dsev = s (stream-1 authority)
  dynamics   IDV(13) reaction-kinetics drift at severity dsev = s

Timing: plant-side interventions switch on 5 samples (15 min) before the
forecast origin and persist (diagnostic-prefix semantics). The predictor and
the persistence reference both see the realized (disturbed) history of the
intervention arm; ground truth is that arm's forecast span. The paired
nominal arm shares the episode seed, so the IDV(8) excitation realization is
identical across arms and severities.

One window per evaluation episode: history [280, 300), forecast [300, 340).
80 independent episodes (seeds 4000-4079).

skill(model; s) = mean-of-window RMSE(persistence; s) / RMSE(model; s).

Usage:
    python tep_retention_eval.py
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
EVAL_SEEDS = list(range(4000, 4080))
SEEDS = [int(s) for s in
         os.environ.get("RBM_SEEDS", "42,123,7,2024,31337").split(",")]
N_BOOT = 1000

OBS_GRID = [0.0, 0.02, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.5, 0.7, 1.0]
DOMAINS = {
    "process_idv1": {"idv": 1, "levels": [0.1, 0.25, 0.5, 0.75, 1.0, 1.5]},
    "actuation_idv6": {"idv": 6, "levels": [0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]},
    "dynamics_idv13": {"idv": 13, "levels": [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]},
}


def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    with open(OUT / "run.log", "a") as f:
        f.write(line + "\n")


def make_jobs():
    jobs, keys = [], []
    for dom, spec in DOMAINS.items():
        for s in spec["levels"]:
            for seed in EVAL_SEEDS:
                idata = T.nominal_idata()
                idata[T_ONSET:, spec["idv"] - 1] = 1
                dsev = T.nominal_dsev()
                dsev[spec["idv"] - 1] = s
                jobs.append((idata, seed, dsev))
                keys.append((dom, s, seed))
    return jobs, keys


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    norm = np.load(ART / "normalization.npz")
    shift, scale = norm["shift"], norm["scale"]
    ep = np.load(ART / "episodes.npz")
    nominal = {int(s): ep[f"eval_{s}"] for s in EVAL_SEEDS}

    log(f"simulating intervention arms")
    jobs, keys = make_jobs()
    t0 = time.time()
    outs = simulate_batch(jobs, workers=8)
    arms = {}
    dropped = 0
    for k, x in zip(keys, outs):
        if np.isfinite(x).all():
            arms[k] = x[:, :T.N_MEAS + T.N_MV].astype(np.float32)
        else:
            dropped += 1
    log(f"  {len(jobs)} sims in {time.time()-t0:.0f}s, non-finite dropped: {dropped}")

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

    def windows_for(traj_by_seed, obs_op=None, s_obs=0.0):
        """Features, last-values and gt for each eval episode."""
        F, L, G = [], [], []
        for seed in EVAL_SEEDS:
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
        """Per-window RMSE, normalized units and physical units."""
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
        r_norm = np.sqrt(np.mean(e ** 2, axis=(1, 2)))
        r_phys = np.sqrt(np.mean((e * meas_scale) ** 2, axis=(1, 2)))
        return r_norm, r_phys

    MODELS = ["persistence", "ridge"] + list(nets)
    tables = {m: {} for m in MODELS}
    tables_phys = {m: {} for m in MODELS}

    # observation domains (gt = nominal arm)
    for op in ("bias", "noise"):
        for s in OBS_GRID:
            F, L, G = windows_for(nominal, op, s)
            for m in MODELS:
                rn, rp = rmse_w(F, L, G, m)
                tables[m][(f"obs_{op}", s)] = rn
                tables_phys[m][(f"obs_{op}", s)] = rp
    # plant-side domains
    for dom, spec in DOMAINS.items():
        for m in MODELS:
            tables[m][(dom, 0.0)] = tables[m][("obs_bias", 0.0)]
            tables_phys[m][(dom, 0.0)] = tables_phys[m][("obs_bias", 0.0)]
        for s in spec["levels"]:
            traj = {seed: arms[(dom, s, seed)] for seed in EVAL_SEEDS
                    if (dom, s, seed) in arms}
            keep = list(traj)
            F, L, G = windows_for({k: traj[k] for k in keep})
            for m in MODELS:
                rn, rp = rmse_w(F, L, G, m)
                fn = np.full(len(EVAL_SEEDS), np.nan)
                fp = np.full(len(EVAL_SEEDS), np.nan)
                for i, seed in enumerate(keep):
                    fn[EVAL_SEEDS.index(seed)] = rn[i]
                    fp[EVAL_SEEDS.index(seed)] = rp[i]
                tables[m][(dom, s)] = fn
                tables_phys[m][(dom, s)] = fp

    np.savez_compressed(
        OUT / "tables.npz",
        **{f"norm|{m}|{d}|{s}": v for m, t in tables.items()
           for (d, s), v in t.items()},
        **{f"phys|{m}|{d}|{s}": v for m, t in tables_phys.items()
           for (d, s), v in t.items()})
    log("wrote tables.npz")

    all_domains = {"obs_bias": OBS_GRID, "obs_noise": OBS_GRID,
                   **{d: [0.0] + spec["levels"] for d, spec in DOMAINS.items()}}

    rng = np.random.default_rng(0)
    bidx = rng.integers(0, len(EVAL_SEEDS), size=(N_BOOT, len(EVAL_SEEDS)))

    def curve(mname, dom, idx=None):
        out = []
        for s in all_domains[dom]:
            a = tables["persistence"][(dom, s)]
            b = tables[mname][(dom, s)]
            ok = np.isfinite(a) & np.isfinite(b)
            if idx is not None:
                a, b = a[idx], b[idx]
                ok = np.isfinite(a) & np.isfinite(b)
            out.append(np.nanmean(a[ok]) / np.nanmean(b[ok]))
        return np.array(out)

    def crossing(levels, c):
        for i in range(1, len(levels)):
            if c[i] < 1.0 <= c[i - 1]:
                return levels[i - 1] + (c[i - 1] - 1.0) / (c[i - 1] - c[i]) * \
                    (levels[i] - levels[i - 1])
        return None

    results = {"n_episodes": len(EVAL_SEEDS), "domains": {
        k: list(v) for k, v in all_domains.items()}, "models": {}}
    for m in MODELS:
        if m == "persistence":
            continue
        mres = {}
        for dom, levels in all_domains.items():
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
            log(f"  {m:10s} {dom:15s} skill(0)={pt[0]:.2f} "
                f"min={pt.min():.2f} ux={ux}")
        results["models"][m] = mres

    with open(OUT / "raw_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    log("wrote raw_results.json")


if __name__ == "__main__":
    main()
