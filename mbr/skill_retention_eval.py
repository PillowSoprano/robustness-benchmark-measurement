"""Skill-retention curves with bootstrap intervals and multi-seed spread.

skill(s) = mean-of-window RMSE of persistence at severity s over the same
quantity for the model. The reference degrades with severity too: its
forecast uses the corrupted history in the observation domain and is scored
against the perturbed plant elsewhere, so the ratio measures retained
advantage rather than raw error.

Reports the curve, the utility crossing, and the GRU-to-MLP crossover, each
with a 95% bootstrap interval over windows and the spread over training
seeds.
"""

import contextlib
import io
import json
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
ART = HERE / "artifacts"
FP = HERE / "fingerprint"
OUT = HERE / "skill_retention"

SEED = 42
SEEDS = [42, 123, 7]
HIST, HORIZON = 20, 80
TEST_SEQS = list(range(30, 38))
STRIDE = 150
OBS_GRID = [0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.07, 0.09,
            0.12, 0.16, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.80, 1.0]
PROC = [0, 10, 25, 50, 75, 100]
ACT = [0.0, 0.05, 0.1, 0.2, 0.35, 0.5]
N_BOOT = 1000

def main():
    os.chdir(WW_ROOT)
    sys.path.insert(0, WW_ROOT)
    sys.path.insert(0, str(HERE))
    OUT.mkdir(parents=True, exist_ok=True)

    import args_new as new_args
    from experiments.process_disturbances import load_model_and_data
    from utils.disturbance_handler import get_default_disturbance_handler
    from train_panel import MLP, Seq2Seq

    args = dict(new_args.args)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        load_model_and_data("mamba", "MBR", args)
    track = np.asarray(args["pred_tracking"], dtype=int)
    K = len(track)
    f = "save_model/mamba/MBR"
    sx, cx, su, cu = (np.loadtxt(f"{f}/{k}.txt", dtype=np.float32)
                      for k in ("shift_x", "scale_x", "shift_u", "scale_u"))
    scale_tr = cx[track]
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
            wins.append({"si": si, "t": t, "hs": st[t:t + HIST].copy(),
                         "ha": ac[t:t + HIST, :4].copy(),
                         "fa": ac[t + HIST:t + HIST + HORIZON, :4].copy()})
            t += STRIDE
    n_win = len(wins)

    cache = np.load(FP / "arms_cache.npz")
    assert cache["nominal"].shape[0] == n_win, "window mismatch vs arms cache"
    gt = {("nominal", 0): cache["nominal"][:, :, track]}
    for p in PROC[1:]:
        gt[("proc", p)] = cache[f"proc_{p}"][:, :, track]
    for s in ACT[1:]:
        gt[("act", s)] = cache[f"act_{s}"][:, :, track]

    def features(w, obs_op=None, s=0.0):
        hs = ((w["hs"] - sx) / cx)[:, track]
        if obs_op == "bias" and s > 0:
            hs = hs + s
        elif obs_op == "noise" and s > 0:
            rng = np.random.RandomState(SEED + 7919 * (w["si"] * 10000 + w["t"]))
            hs = hs + rng.randn(*hs.shape) * s
        ha = (w["ha"] - su[:4]) / cu[:4]
        fa = (w["fa"] - su[:4]) / cu[:4]
        ts = w["t"]
        hd = (dist[ts:ts + HIST] - su[4:14]) / (cu[4:14] + 1e-8)
        fd = (dist[ts + HIST:ts + HIST + HORIZON] - su[4:14]) / (cu[4:14] + 1e-8)
        return (np.concatenate([hs.reshape(-1), ha.reshape(-1), hd.reshape(-1),
                                fa.reshape(-1), fd.reshape(-1)]).astype(np.float32),
                hs[-1])

    din = HIST * (K + 14) + HORIZON * 14
    dout = HORIZON * K
    W = np.load(ART / "ridge_W.npy")
    nets = {}
    for sd in SEEDS:
        base = ART if sd == 42 else ART / f"seed_{sd}"
        m = MLP(din, dout)
        m.load_state_dict(torch.load(base / "mlp.pt"))
        m.eval()
        g = Seq2Seq(K, 14)
        g.load_state_dict(torch.load(base / "gru.pt"))
        g.eval()
        nets[("mlp", sd)] = m
        nets[("gru", sd)] = g

    def predict(model_key, obs_op=None, s=0.0):
        F, L = zip(*(features(w, obs_op, s) for w in wins))
        F = np.asarray(F, np.float32)
        L = np.asarray(L, np.float32)
        if model_key == "persistence":
            inc = np.zeros((n_win, dout), np.float32)
        elif model_key == "ridge":
            inc = np.concatenate([F, np.ones((len(F), 1), np.float32)], 1) @ W
        else:
            net = nets[model_key]
            with torch.no_grad():
                inc = np.concatenate([net(torch.from_numpy(F[i:i + 128])).numpy()
                                      for i in range(0, len(F), 128)])
        return (inc.reshape(-1, HORIZON, K) + np.asarray(L)[:, None, :]) \
            * scale_tr + sx[track]

    def rmse_w(pred, gtarr):
        return np.sqrt(np.mean((pred - gtarr) ** 2, axis=(1, 2)))

    # ---- per-window RMSE tables: r[model][domain][severity] -> (n_win,)
    log = lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
    model_keys = (["persistence", "ridge"]
                  + [("mlp", sd) for sd in SEEDS] + [("gru", sd) for sd in SEEDS])
    tables = {}
    for mk in model_keys:
        name = mk if isinstance(mk, str) else f"{mk[0]}_s{mk[1]}"
        log(f"scoring {name}")
        tab = {}
        clean = predict(mk)
        for op in ("bias", "noise"):
            for s in OBS_GRID:
                pred = clean if s == 0 else predict(mk, op, s)
                tab[(f"obs_{op}", s)] = rmse_w(pred, gt[("nominal", 0)])
        for p in PROC:
            tab[("process", p)] = rmse_w(clean, gt[("nominal", 0)] if p == 0
                                         else gt[("proc", p)])
        for a in ACT:
            tab[("actuation", a)] = rmse_w(clean, gt[("nominal", 0)] if a == 0
                                           else gt[("act", a)])
        tables[name] = tab

    domains = {"obs_bias": OBS_GRID, "obs_noise": OBS_GRID,
               "process": PROC, "actuation": ACT}

    rng = np.random.default_rng(0)
    boot_idx = rng.integers(0, n_win, size=(N_BOOT, n_win))

    def skill_curve(model_name, dom, idx=None):
        levels = domains[dom]
        pers = tables["persistence"]
        mod = tables[model_name]
        out = []
        for lv in levels:
            a = pers[(dom, lv)]
            b = mod[(dom, lv)]
            if idx is None:
                out.append(a.mean() / b.mean())
            else:
                out.append(a[idx].mean() / b[idx].mean())
        return np.array(out)

    def crossing(levels, curve, target=1.0):
        for i in range(1, len(levels)):
            if curve[i] < target <= curve[i - 1]:
                s0, s1 = levels[i - 1], levels[i]
                c0, c1 = curve[i - 1], curve[i]
                return s0 + (c0 - target) / (c0 - c1) * (s1 - s0)
        return None

    results = {"n_windows": n_win, "seeds": SEEDS, "obs_grid": OBS_GRID,
               "proc": PROC, "act": ACT, "models": {}}
    for name in tables:
        if name == "persistence":
            continue
        mres = {}
        for dom, levels in domains.items():
            point = skill_curve(name, dom)
            boots = np.stack([skill_curve(name, dom, boot_idx[b])
                              for b in range(N_BOOT)])
            lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
            ux = crossing(levels, point)
            ux_b = [crossing(levels, boots[b]) for b in range(N_BOOT)]
            ux_valid = [u for u in ux_b if u is not None]
            mres[dom] = {
                "levels": list(levels),
                "skill": point.tolist(),
                "ci_lo": lo.tolist(), "ci_hi": hi.tolist(),
                "utility_crossing": ux,
                "utility_crossing_ci": (
                    [float(np.percentile(ux_valid, 2.5)),
                     float(np.percentile(ux_valid, 97.5))]
                    if len(ux_valid) > N_BOOT * 0.5 else None),
                "crossing_censored_frac": 1 - len(ux_valid) / N_BOOT,
            }
        results["models"][name] = mres

    xo = {}
    for sd in SEEDS:
        g = tables[f"gru_s{sd}"]
        m = tables[f"mlp_s{sd}"]
        levels = ACT
        curve = [m[("actuation", a)].mean() / g[("actuation", a)].mean()
                 for a in levels]
        xo[str(sd)] = crossing(levels, np.array(curve))
    results["gru_mlp_actuation_crossover_per_seed"] = xo

    with open(OUT / "raw_results.json", "w") as fh:
        json.dump(results, fh, indent=2, default=float)
    log("wrote raw_results.json")

if __name__ == "__main__":
    main()
