"""Four-domain fingerprint on the skillful panel.

Windowed re-anchored evaluation: each window is scored against the plant
state at its own forecast origin, which confines the open-loop drift that
inverts model ordering over long re-simulations.

Failure radius uses the relative gate, degradation above tau on at least a
fraction alpha of windows. Absolute error under the same interventions is
reported alongside, since the two disagree.
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

WW_ROOT = os.environ.get("MBR_PROJECT_ROOT")
if not WW_ROOT:
    raise SystemExit("set MBR_PROJECT_ROOT to the MBR project checkout; see README")
HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"
OUT = HERE / "fingerprint"

SEED = 42
HIST, HORIZON = 20, 80
TEST_SEQS = list(range(30, 38))
STRIDE = 150

OBS_GRID = [0.0, 0.01, 0.02, 0.035, 0.05, 0.08, 0.12, 0.18, 0.25, 0.4, 0.6, 1.0]
MAX_BIAS = MAX_NOISE = 1.0
PROC_LEVELS = [0, 10, 25, 50, 75, 100]
ACT_LEVELS = [0.0, 0.05, 0.1, 0.2, 0.35, 0.5]
TAU, ALPHA = 0.10, 0.10

LOG = None

def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")

def main():
    global LOG
    os.chdir(WW_ROOT)
    sys.path.insert(0, WW_ROOT)
    OUT.mkdir(parents=True, exist_ok=True)
    LOG = OUT / "run.log"
    sys.path.insert(0, str(HERE))

    import args_new as new_args
    from experiments.process_disturbances import load_model_and_data
    from utils.disturbance_handler import get_default_disturbance_handler
    from train_panel import MLP, Seq2Seq

    args = dict(new_args.args)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _m, _td, env, _nm = load_model_and_data("mamba", "MBR", args)
    del _m, _td, _nm
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
            wins.append({"si": si, "t": t,
                         "hs": st[t:t + HIST].copy(),
                         "ha": ac[t:t + HIST, :4].copy(),
                         "fa": ac[t + HIST:t + HIST + HORIZON, :4].copy(),
                         "x0": st[t + HIST - 1].copy()})
            t += STRIDE
    log(f"{len(wins)} frozen-test windows")

    # ---- plant simulation of the forecast horizon
    def sim(x0, acts, dmat):
        env.reset()
        env.p = dmat.copy()
        env.state = x0.copy()
        tr = np.zeros((HORIZON, x0.shape[0]))
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

    log("simulating nominal arm")
    t0 = time.time()
    nominal = [sim(w["x0"], w["fa"], dslice(w)) for w in wins]
    log(f"  nominal done ({time.time()-t0:.0f}s)")

    arms = {}
    for pct in PROC_LEVELS[1:]:
        t0 = time.time()
        for i, w in enumerate(wins):
            dm = dslice(w)
            dm *= (1 + pct / 100.0)
            arms[("proc", pct, i)] = sim(w["x0"], w["fa"], dm)
        log(f"  process +{pct}% done ({time.time()-t0:.0f}s)")
    for s in ACT_LEVELS[1:]:
        t0 = time.time()
        for i, w in enumerate(wins):
            an = (w["fa"] - su[:4]) / cu[:4]
            ad = an * (1.0 - s) * cu[:4] + su[:4]
            arms[("act", s, i)] = sim(w["x0"], ad, dslice(w))
        log(f"  actuation s={s} done ({time.time()-t0:.0f}s)")

    # ---- feature builder matching train_panel
    def features(w, obs_op=None, s=0.0):
        hs = ((w["hs"] - sx) / cx)[:, track]
        if obs_op == "bias" and s > 0:
            hs = hs + s * MAX_BIAS
        elif obs_op == "noise" and s > 0:
            rng = np.random.RandomState(SEED + 7919 * (w["si"] * 10000 + w["t"]))
            hs = hs + rng.randn(*hs.shape) * (s * MAX_NOISE)
        ha = (w["ha"] - su[:4]) / cu[:4]
        fa = (w["fa"] - su[:4]) / cu[:4]
        ts = w["t"]
        hd = (dist[ts:ts + HIST] - su[4:14]) / (cu[4:14] + 1e-8)
        fd = (dist[ts + HIST:ts + HIST + HORIZON] - su[4:14]) / (cu[4:14] + 1e-8)
        feat = np.concatenate([hs.reshape(-1), ha.reshape(-1), hd.reshape(-1),
                               fa.reshape(-1), fd.reshape(-1)])
        return feat.astype(np.float32), hs[-1]

    din = HIST * (K + 4 + 10) + HORIZON * 14
    dout = HORIZON * K
    W = np.load(ART / "ridge_W.npy")
    mlp = MLP(din, dout)
    mlp.load_state_dict(torch.load(ART / "mlp.pt"))
    mlp.eval()
    gru = Seq2Seq(K, 14)
    gru.load_state_dict(torch.load(ART / "gru.pt"))
    gru.eval()

    def predict(name, obs_op=None, s=0.0):
        """Absolute physical predictions, shape (n_win, HORIZON, K)."""
        F, L = [], []
        for w in wins:
            fe, last = features(w, obs_op, s)
            F.append(fe)
            L.append(last)
        F = np.asarray(F, np.float32)
        L = np.asarray(L, np.float32)
        if name == "persistence":
            inc = np.zeros((len(wins), dout), np.float32)
        elif name == "ridge":
            inc = np.concatenate([F, np.ones((len(F), 1), np.float32)], 1) @ W
        else:
            net = mlp if name == "mlp" else gru
            with torch.no_grad():
                inc = np.concatenate([net(torch.from_numpy(F[i:i + 128])).numpy()
                                      for i in range(0, len(F), 128)])
        norm_pred = inc.reshape(-1, HORIZON, K) + L[:, None, :]
        return norm_pred * scale_tr + sx[track]

    def rmse_per_window(pred, arm_list):
        gt = np.stack([a[:, track] for a in arm_list])
        return np.sqrt(np.mean((pred - gt) ** 2, axis=(1, 2)))

    MODELS = ["persistence", "ridge", "mlp", "gru"]
    results = {"n_windows": len(wins), "test_seqs": TEST_SEQS,
               "horizon": HORIZON, "tau": TAU, "alpha": ALPHA, "models": {}}

    for name in MODELS:
        log(f"=== {name} ===")
        clean = predict(name)
        r_nom = rmse_per_window(clean, nominal)
        mres = {"nominal_mean_rmse": float(r_nom.mean()),
                "nominal_pooled": float(np.sqrt(np.mean(r_nom ** 2))),
                "domains": {}}

        def curve(levels, get_pred, get_arm, key):
            out, ps = {}, []
            for lv in levels:
                pred = get_pred(lv)
                arm = get_arm(lv)
                r = rmse_per_window(pred, arm)
                dg = r / r_nom - 1.0
                p = float(np.mean(dg > TAU))
                ps.append(p)
                out[str(lv)] = {"mean_rmse": float(r.mean()),
                                "pooled_rmse": float(np.sqrt(np.mean(r ** 2))),
                                "mean_degr": float(dg.mean()),
                                "p_fail": p}
            radius, prev = None, levels[0]
            for lv, p in zip(levels, ps):
                if lv == levels[0]:
                    continue
                if p >= ALPHA:
                    radius = 0.5 * (prev + lv)
                    break
                prev = lv
            mres["domains"][key] = {"curve": out, "radius": radius}
            log(f"  {key}: radius="
                f"{'censored' if radius is None else round(radius, 4)} "
                f"max_degr={max(v['mean_degr'] for v in out.values()):.3f}")

        curve(OBS_GRID, lambda s: predict(name, "bias", s),
              lambda s: nominal, "obs_bias")
        curve(OBS_GRID, lambda s: predict(name, "noise", s),
              lambda s: nominal, "obs_noise")
        curve(PROC_LEVELS, lambda p: clean,
              lambda p: nominal if p == 0 else
              [arms[("proc", p, i)] for i in range(len(wins))], "process")
        curve(ACT_LEVELS, lambda s: clean,
              lambda s: nominal if s == 0 else
              [arms[("act", s, i)] for i in range(len(wins))], "actuation")
        results["models"][name] = mres

    with open(OUT / "raw_results.json", "w") as fh:
        json.dump(results, fh, indent=2)
    log("wrote raw_results.json")

if __name__ == "__main__":
    main()
