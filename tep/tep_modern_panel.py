"""Put two modern architectures through the same TEP protocol as the panel.

The panel in the paper stops at persistence, ridge, MLP, and GRU. The
question this answers is whether the retention shapes it reports are
properties of the task or of those four architectures: does a decomposition
linear model and a patch transformer, trained on the same episodes with the
same information contract and the same seeds, retain skill the same way?

The regime domain is the test that matters. Every existing member crosses
below persistence there while the disturbance domains leave everyone above 1.
If that is about what an intervention invalidates, architectures with
different inductive biases should still collapse in the regime domain.

Reuses the stored episodes and normalization, so no nominal re-simulation.
Intervention arms are re-simulated because only per-window errors were kept.

Usage:
    python tep_modern_panel.py
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
from train_tep_panel import MLP, Seq2Seq, HORIZON, STRIDE_TR, STRIDE_EV, fit
from tep_modern_arch import DecompLinear, PatchTransformer
from shutdown_scan import freeze_onset

ART = HERE / "artifacts"
OUT = HERE / "modern"
SEEDS = [int(s) for s in
         os.environ.get("RBM_SEEDS", "42,123,7,2024,31337").split(",")]
N_BOOT = 1000
T_ORIGIN, T_ONSET, SPAN_END = 300, 295, 340
EVAL_SEEDS = list(range(4000, 4080))

DOMAINS = {
    "process_idv1": ("idv", 1, [0.5, 1.0, 2.0]),
    "actuation_idv6": ("idv", 6, [0.35, 0.75, 1.0]),
    "dynamics_idv13": ("idv", 13, [1.0, 3.0, 6.0]),
    "regime_setpt18": ("setpt", 18, [0.25, 0.5, 1.0]),
}
OBS_GRID = [0.0, 0.25, 0.5, 1.0, 2.0]
NEW = {"dlinear": DecompLinear, "patchtf": PatchTransformer}


def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "run.log", "a") as f:
        f.write(line + "\n")


def main():
    norm = np.load(ART / "normalization.npz")
    shift, scale = norm["shift"], norm["scale"]
    ep = np.load(ART / "episodes.npz")
    split = {k: [(int(s), ep[f"{k}_{s}"]) for s in v
                 if f"{k}_{s}" in ep.files] for k, v in T.SEEDS.items()}
    log("episodes " + ", ".join(f"{k} {len(v)}" for k, v in split.items()))

    Xtr, Ytr = T.build_windows(split["train"], HORIZON, STRIDE_TR, shift, scale)
    Xva, Yva = T.build_windows(split["val"], HORIZON, STRIDE_TR, shift, scale)
    Xte, Yte = T.build_windows(split["eval"], HORIZON, STRIDE_EV, shift, scale)
    log(f"windows train {len(Xtr)} val {len(Xva)} eval {len(Xte)}")

    def rmse_w(pred):
        return np.sqrt(np.mean((pred - Yte).reshape(len(Yte), -1) ** 2, axis=1))

    r_pers = rmse_w(np.zeros_like(Yte))
    rng = np.random.default_rng(0)
    bidx = rng.integers(0, len(Yte), size=(N_BOOT, len(Yte)))

    def skill_ci(r):
        pt = r_pers.mean() / r.mean()
        bs = [r_pers[i].mean() / r[i].mean() for i in bidx]
        return pt, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

    # ---- train the new members and gate them like the rest of the panel
    nets, gate = {}, {}
    for name, cls in NEW.items():
        for sd in SEEDS:
            torch.manual_seed(sd)
            np.random.seed(sd)
            m = cls(T.HIST, HORIZON)
            lr = 1e-3 if name == "dlinear" else 5e-4
            m = fit(m, Xtr, Ytr, Xva, Yva, 60, 256, lr, f"{name}_s{sd}")
            m.eval()
            with torch.no_grad():
                pr = np.concatenate([m(torch.from_numpy(Xte[i:i + 512])).numpy()
                                     for i in range(0, len(Xte), 512)])
            pt, lo, hi = skill_ci(rmse_w(pr))
            gate[f"{name}_s{sd}"] = {"skill": pt, "ci": [lo, hi]}
            nets[f"{name}_s{sd}"] = m
            log(f"  {name}_s{sd} skill {pt:.2f} CI [{lo:.2f},{hi:.2f}]"
                + ("" if lo > 1 else "   FAILS THE GATE"))

    # ---- the incumbent panel, for a matched comparison
    din, dout = Xtr.shape[1], Ytr.shape[1]
    W = np.load(ART / "ridge_W.npy")
    for sd in SEEDS:
        a = MLP(din, dout)
        a.load_state_dict(torch.load(ART / f"seed_{sd}" / "mlp.pt"))
        a.eval()
        b = Seq2Seq()
        b.load_state_dict(torch.load(ART / f"seed_{sd}" / "gru.pt"))
        b.eval()
        nets[f"mlp_s{sd}"] = a
        nets[f"gru_s{sd}"] = b
    FAM = ["ridge", "mlp", "gru", "dlinear", "patchtf"]

    # ---- intervention arms
    nominal = {s: x for s, x in split["eval"]}
    keep0 = list(nominal)
    jobs, keys = [], []
    for dom, (kind, idx, levels) in DOMAINS.items():
        for s in levels:
            idata, dsev, spoff = T.nominal_idata(), T.nominal_dsev(), None
            if kind == "idv":
                idata[T_ONSET:, idx - 1] = 1
                dsev[idx - 1] = s
            else:
                spoff = np.zeros(20)
                spoff[idx - 1] = s * 10.0
            for sd in keep0:
                jobs.append((idata.copy(), sd, dsev.copy(), spoff))
                keys.append((dom, s, sd))
    log(f"simulating {len(jobs)} intervention arms")
    t0 = time.time()
    arms = {}
    for k, x in zip(keys, simulate_batch(jobs, workers=8)):
        x = np.asarray(x)[:, :T.N_MEAS + T.N_MV].astype(np.float32)
        f0 = freeze_onset(x) if np.isfinite(x).all() else None
        if np.isfinite(x).all() and (f0 is None or f0 >= SPAN_END):
            arms[k] = x
    log(f"  {len(arms)}/{len(jobs)} valid ({time.time()-t0:.0f}s)")

    def score(traj, seeds, op=None, sev=0.0):
        F, L, G = [], [], []
        for sd in seeds:
            z = (traj[sd] - shift) / scale
            h = z[T_ORIGIN - T.HIST:T_ORIGIN].copy()
            if op == "bias" and sev > 0:
                h[:, :T.N_MEAS] += sev
            elif op == "noise" and sev > 0:
                r = np.random.RandomState(97 + sd)
                h[:, :T.N_MEAS] += r.randn(T.HIST, T.N_MEAS) * sev
            F.append(h.reshape(-1)); L.append(h[-1, :T.N_MEAS])
            G.append(z[T_ORIGIN:T_ORIGIN + HORIZON, :T.N_MEAS])
        F = np.asarray(F, np.float32); L = np.asarray(L, np.float32)
        G = np.asarray(G, np.float32)
        out = {"persistence": np.sqrt(np.mean(
            (np.repeat(L[:, None, :], HORIZON, axis=1) - G) ** 2, axis=(1, 2)))}
        inc_r = np.concatenate([F, np.ones((len(F), 1), np.float32)], 1) @ W
        out["ridge"] = np.sqrt(np.mean(
            (inc_r.reshape(-1, HORIZON, T.N_MEAS) + L[:, None, :] - G) ** 2,
            axis=(1, 2)))
        for fam in ("mlp", "gru", "dlinear", "patchtf"):
            per = []
            for sd in SEEDS:
                with torch.no_grad():
                    inc = nets[f"{fam}_s{sd}"](torch.from_numpy(F)).numpy()
                pred = inc.reshape(-1, HORIZON, T.N_MEAS) + L[:, None, :]
                e = np.sqrt(np.mean((pred - G) ** 2, axis=(1, 2)))
                # keep the seed resolved as well as averaged: averaging the
                # errors first throws away the spread the protocol requires
                out[f"{fam}_s{sd}"] = e
                per.append(e)
            out[fam] = np.mean(per, axis=0)
        return out

    MEMBERS = FAM + [f"{f}_s{sd}" for f in ("mlp", "gru", "dlinear", "patchtf")
                     for sd in SEEDS]

    def ratios(r):
        return {m: float(r["persistence"].mean() / r[m].mean())
                for m in MEMBERS if m in r}

    curves = {}
    for op in ("bias", "noise"):
        for sev in OBS_GRID:
            curves[(f"obs_{op}", sev)] = ratios(score(nominal, keep0, op, sev))
    r0 = score(nominal, keep0)
    for dom, (_, _, levels) in DOMAINS.items():
        curves[(dom, 0.0)] = ratios(r0)
        for s in levels:
            sub = [sd for sd in keep0 if (dom, s, sd) in arms]
            if len(sub) < len(keep0) * 0.5:
                log(f"  {dom} s={s}: {len(sub)}/{len(keep0)} valid, censored")
                continue
            curves[(dom, s)] = ratios(
                score({sd: arms[(dom, s, sd)] for sd in sub}, sub))

    doms = sorted({d for d, _ in curves})
    log("")
    log(f"{'domain':17s} " + " ".join(f"{m:>9s}" for m in FAM))
    res = {"gate": gate, "curves": {}}
    for d in doms:
        levels = sorted(s for dd, s in curves if dd == d)
        res["curves"][d] = {"levels": levels,
                            **{m: [curves[(d, s)][m] for s in levels]
                               for m in MEMBERS
                               if all(m in curves[(d, s)] for s in levels)}}
        mins = {m: min(curves[(d, s)][m] for s in levels) for m in FAM}
        log(f"{d:17s} " + " ".join(f"{mins[m]:9.2f}" for m in FAM)
            + ("   all cross" if max(mins.values()) < 1 else
               "   some cross" if min(mins.values()) < 1 else ""))
    with open(OUT / "modern_panel.json", "w") as f:
        json.dump(res, f, indent=2, default=float)
    log("wrote modern_panel.json")


if __name__ == "__main__":
    main()
