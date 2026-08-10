"""Regime domain and second actuation operator on the TEP panel.

Two intervention families complete the protocol's required operator coverage:
the regime domain uses the SETPT path, and a second actuation operator enables
the within-domain operator-choice check on TEP.

regime_setpt18
  Reactor temperature setpoint offset, applied at t = 0 for the WHOLE
  episode: the plant operates around a shifted operating point, which is the
  taxonomy's regime semantics (sustained support shift of the evaluation
  distribution relative to training), an onset event is deliberately not
  used. Severity map: offset = s * 10 C. Nominal setpoint 120.4 C, operating
  limit 150 C, so s = 2.96 reaches the limit. Grid stops at 2.5.
  History AND ground truth both come from the shifted-regime trajectory.

actuation_idv14
  Reactor cooling-water valve sticking (IDV(14)) with the dead-band width as
  continuous severity: VST(10) = 2.0 * dsev(14) percent (patched in
  teprob.f). dsev = 1 reproduces the original TEP fault; dsev < 1 shrinks the
  dead band, dsev > 1 widens it. Onset 15 min before forecast origin, like
  the other plant-side domains.

Shutdown masking: XMEAS-frozen-before-sample-340 episodes are dropped; a
level with under 50% surviving episodes is censored (extension rule
criterion 3).

Usage:
    python tep_regime_actuation2.py
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
N_BOOT = 1000

REGIME_LEVELS = [0.25, 0.5, 1.0, 1.5, 2.0, 2.5]     # offset = s * 10 C
ACT2_LEVELS = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0]        # dead band = 2 * s percent


def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    with open(OUT / "regime_act2.log", "a") as f:
        f.write(line + "\n")


def main():
    norm = np.load(ART / "normalization.npz")
    shift, scale = norm["shift"], norm["scale"]
    ep = np.load(ART / "episodes.npz")
    nominal = {int(s): ep[f"eval_{s}"] for s in EVAL_SEEDS}

    jobs, keys = [], []
    for s in REGIME_LEVELS:
        spoff = np.zeros(20)
        spoff[17] = s * 10.0                          # SETPT(18), 0-based 17
        for seed in EVAL_SEEDS:
            jobs.append((T.nominal_idata(), seed, T.nominal_dsev(), spoff))
            keys.append(("regime_setpt18", s, seed))
    for s in ACT2_LEVELS:
        idata = T.nominal_idata()
        idata[T_ONSET:, 13] = 1                       # IDV(14), 0-based 13
        dsev = T.nominal_dsev()
        dsev[13] = s
        for seed in EVAL_SEEDS:
            jobs.append((idata, seed, dsev))
            keys.append(("actuation_idv14", s, seed))
    log(f"simulating {len(jobs)} arms (regime + actuation2)")
    t0 = time.time()
    outs = simulate_batch(jobs, workers=8)
    arms, shutdown = {}, {}
    for k, x in zip(keys, outs):
        xx = np.asarray(x)[:, :T.N_MEAS + T.N_MV].astype(np.float32)
        f0 = None if not np.isfinite(xx).all() else freeze_onset(xx)
        if (not np.isfinite(xx).all()) or (f0 is not None and f0 < SPAN_END):
            shutdown[k] = -1 if f0 is None else int(f0)
        else:
            arms[k] = xx
    sd_counts = {}
    for dom, levels in (("regime_setpt18", REGIME_LEVELS),
                        ("actuation_idv14", ACT2_LEVELS)):
        for s in levels:
            n = sum(1 for seed in EVAL_SEEDS if (dom, s, seed) in shutdown)
            sd_counts[f"{dom}@{s}"] = n
            if n:
                log(f"  {dom} s={s}: {n}/80 shut down inside scoring span")
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
    MODELS = ["persistence", "ridge"] + list(nets)

    def windows_for(traj_by_seed, seeds):
        F, L, G = [], [], []
        for seed in seeds:
            z = (traj_by_seed[seed] - shift) / scale
            h = z[T_ORIGIN - T.HIST:T_ORIGIN]
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
    F0, L0, G0 = windows_for(nominal, EVAL_SEEDS)
    for m in MODELS:
        rn, rp = rmse_w(F0, L0, G0, m)
        for dom in ("regime_setpt18", "actuation_idv14"):
            tables["norm"][m][(dom, 0.0)] = rn
            tables["phys"][m][(dom, 0.0)] = rp

    for dom, levels in (("regime_setpt18", REGIME_LEVELS),
                        ("actuation_idv14", ACT2_LEVELS)):
        for s in levels:
            keep = [seed for seed in EVAL_SEEDS if (dom, s, seed) in arms]
            if len(keep) < len(EVAL_SEEDS) * 0.5:
                log(f"  {dom} s={s}: {len(keep)}/80 valid -> censored")
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
    log("scored both domains")

    np.savez_compressed(
        OUT / "tables_regime_act2.npz",
        **{f"{sp}|{m}|{d}|{s}": v for sp, tt in tables.items()
           for m, t in tt.items() for (d, s), v in t.items()})

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

    results = {"shutdown_counts": sd_counts,
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
            log(f"  {m:10s} {dom:16s} skill(0)={pt[0]:.2f} "
                f"min={pt.min():.2f} ux={ux}")
        results["models"][m] = mres

    with open(OUT / "regime_act2_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    log("wrote regime_act2_results.json")


if __name__ == "__main__":
    main()
