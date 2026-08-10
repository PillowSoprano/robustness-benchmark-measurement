"""Per-channel absolute admissibility on the MBR panel.

Decision channels are the soluble effluent states of the membrane
compartment, with BSM1 effluent limits as tolerances: S_S 40, S_NO 18,
S_NH 4, S_ND 18 g/m3. The membrane retains particulates, so particulate
states carry no effluent tolerance.

Reports nominal admissibility, absolute trips, and the conjunction-gate
radius wherever it differs from the relative-only radius.
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
TAU, ALPHA = 0.10, 0.10

DECISION = {48: ("S_S_t5", 40.0), 55: ("S_NO_t5", 18.0),
            56: ("S_NH_t5", 4.0), 57: ("S_ND_t5", 18.0)}

def main():
    os.chdir(WW_ROOT)
    sys.path.insert(0, WW_ROOT)
    sys.path.insert(0, str(HERE))

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

    log = lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
    model_keys = (["persistence", "ridge"]
                  + [("mlp", sd) for sd in SEEDS] + [("gru", sd) for sd in SEEDS])
    dch = sorted(DECISION)

    # per-window aggregate RMSE (for the relative gate) and per-channel
    # decision RMSE (for the absolute clause)
    tab_w, tab_c = {}, {}
    for mk in model_keys:
        name = mk if isinstance(mk, str) else f"{mk[0]}_s{mk[1]}"
        log(f"scoring {name}")
        clean = predict(mk)
        cells = ([(f"obs_{op}", s, (None if s == 0 else op), s)
                  for op in ("bias", "noise") for s in OBS_GRID]
                 + [("process", p, None, 0) for p in PROC]
                 + [("actuation", a, None, 0) for a in ACT])
        for dom, lv, op, s_obs in cells:
            pred = clean if (op is None and s_obs == 0) else predict(mk, op, s_obs)
            if dom == "process":
                g = gt[("nominal", 0)] if lv == 0 else gt[("proc", lv)]
            elif dom == "actuation":
                g = gt[("nominal", 0)] if lv == 0 else gt[("act", lv)]
            else:
                g = gt[("nominal", 0)]
            e = pred - g
            # normalized aggregate per window, matching the fingerprint gate
            en = e / scale_tr
            tab_w[(name, dom, lv)] = np.sqrt(np.mean(en ** 2, axis=(1, 2)))
            tab_c[(name, dom, lv)] = {
                c: float(np.sqrt(np.mean(e[:, :, c] ** 2))) for c in dch}

    domains = {"obs_bias": OBS_GRID, "obs_noise": OBS_GRID,
               "process": PROC, "actuation": ACT}
    names = [mk if isinstance(mk, str) else f"{mk[0]}_s{mk[1]}"
             for mk in model_keys]

    out = {"decision_channels": {str(c): {"name": DECISION[c][0],
                                          "tol": DECISION[c][1]}
                                 for c in dch},
           "nominal_admissibility": {}, "absolute_trips": [],
           "radii": {}}

    for name in names:
        trips = {DECISION[c][0]: tab_c[(name, "obs_bias", 0.0)][c]
                 for c in dch
                 if tab_c[(name, "obs_bias", 0.0)][c] > DECISION[c][1]}
        out["nominal_admissibility"][name] = (
            "inadmissible" if trips else "admissible")
        if trips:
            log(f"  NOMINAL TRIP {name}: {trips}")

    for (name, dom, lv), ch in tab_c.items():
        for c in dch:
            if ch[c] > DECISION[c][1]:
                out["absolute_trips"].append(
                    {"model": name, "domain": dom, "severity": lv,
                     "channel": DECISION[c][0], "rmse": ch[c],
                     "tol": DECISION[c][1]})

    for name in names:
        for dom, levels in domains.items():
            r0 = tab_w[(name, dom, levels[0])]
            rel_radius = conj_radius = None
            for i, lv in enumerate(levels[1:], 1):
                rs = tab_w[(name, dom, lv)]
                p_rel = np.mean(rs / r0 - 1 > TAU)
                abs_trip = any(tab_c[(name, dom, lv)][c] > DECISION[c][1]
                               for c in dch)
                if p_rel >= ALPHA and rel_radius is None:
                    rel_radius = 0.5 * (levels[i - 1] + lv)
                if p_rel >= ALPHA and abs_trip and conj_radius is None:
                    conj_radius = 0.5 * (levels[i - 1] + lv)
            out["radii"][f"{name}|{dom}"] = {
                "relative_only": rel_radius, "conjunction": conj_radius}

    # worst-case margin per channel: max over all cells of RMSE_j / tol_j
    out["worst_margin"] = {}
    for c in dch:
        worst = max(tab_c[k][c] for k in tab_c)
        wk = max(tab_c, key=lambda k: tab_c[k][c])
        out["worst_margin"][DECISION[c][0]] = {
            "max_rmse": worst, "tol": DECISION[c][1],
            "ratio": worst / DECISION[c][1],
            "at": f"{wk[0]}|{wk[1]}|{wk[2]}"}
        log(f"  margin {DECISION[c][0]}: max RMSE {worst:.3f} vs tol "
            f"{DECISION[c][1]} (ratio {worst/DECISION[c][1]:.3f}) at {wk}")

    n_diff = sum(1 for v in out["radii"].values()
                 if v["relative_only"] != v["conjunction"])
    log(f"absolute trips: {len(out['absolute_trips'])} cells; "
        f"radii differing relative vs conjunction: {n_diff}/{len(out['radii'])}")

    with open(OUT / "tau_abs_check.json", "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    log("wrote tau_abs_check.json")

if __name__ == "__main__":
    main()
