"""Per-channel absolute admissibility on the TEP panel.

Decision channels are the XMEAS variables carrying declared operating limits,
with the tolerance set to the distance from the base case to the nearest
limit: reactor pressure 190 kPa, reactor level 25%, reactor temperature
29.6 C, separator level 20%, stripper level 20%.

Shutdown limits are deliberately unused: the plant's own protection layer
trips there, so they say nothing about whether a forecast is useful.

Reports absolute trips and the conjunction-gate radius wherever it differs
from the relative-only radius.
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
TAU, ALPHA = 0.10, 0.10

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

DECISION = {6: ("reactor_pressure", 190.0), 7: ("reactor_level", 25.0),
            8: ("reactor_temp", 29.6), 11: ("separator_level", 20.0),
            14: ("stripper_level", 20.0)}

def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    with open(OUT / "tau_abs_check.log", "a") as f:
        f.write(line + "\n")

def main():
    norm = np.load(ART / "normalization.npz")
    shift, scale = norm["shift"], norm["scale"]
    ep = np.load(ART / "episodes.npz")
    nominal = {int(s): ep[f"eval_{s}"] for s in EVAL_SEEDS}
    sd = json.load(open(OUT / "shutdown_table.json"))
    bad = set(sd["freeze_onsets"])            # "dom|s|seed" strings

    jobs, keys = [], []
    for dom, spec in PLANT.items():
        for s in spec["levels"]:
            for seed in EVAL_SEEDS:
                if f"{dom}|{s}|{seed}" in bad:
                    continue
                idata = T.nominal_idata()
                idata[T_ONSET:, spec["idv"] - 1] = 1
                dsev = T.nominal_dsev()
                dsev[spec["idv"] - 1] = s
                jobs.append((idata, seed, dsev))
                keys.append((dom, s, seed))
    log(f"simulating {len(jobs)} valid plant-side arms")
    t0 = time.time()
    outs = simulate_batch(jobs, workers=8)
    arms = {k: x[:, :T.N_MEAS + T.N_MV].astype(np.float32)
            for k, x in zip(keys, outs) if np.isfinite(x).all()}
    log(f"  done in {time.time()-t0:.0f}s")

    din = T.HIST * (T.N_MEAS + T.N_MV)
    dout = HORIZON * T.N_MEAS
    W = np.load(ART / "ridge_W.npy")
    nets = {}
    for sdd in SEEDS:
        m = MLP(din, dout)
        m.load_state_dict(torch.load(ART / f"seed_{sdd}" / "mlp.pt"))
        m.eval()
        g = Seq2Seq()
        g.load_state_dict(torch.load(ART / f"seed_{sdd}" / "gru.pt"))
        g.eval()
        nets[f"mlp_s{sdd}"] = m
        nets[f"gru_s{sdd}"] = g
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
    dch = sorted(DECISION)

    def score(F, L, G, model):
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
        rw = np.sqrt(np.mean(e ** 2, axis=(1, 2)))          # per-window norm
        ep_phys = e * meas_scale                             # physical units
        rc = {c: float(np.sqrt(np.mean(ep_phys[:, :, c] ** 2))) for c in dch}
        return rw, rc

    tab_w, tab_c = {}, {}
    for op in ("bias", "noise"):
        for s in OBS_GRID:
            F, L, G = windows_for(nominal, EVAL_SEEDS, op, s)
            for m in MODELS:
                rw, rc = score(F, L, G, m)
                tab_w[(m, f"obs_{op}", s)] = rw
                tab_c[(m, f"obs_{op}", s)] = rc
    log("scored observation domains")

    for dom, spec in PLANT.items():
        for m in MODELS:
            tab_w[(m, dom, 0.0)] = tab_w[(m, "obs_bias", 0.0)]
            tab_c[(m, dom, 0.0)] = tab_c[(m, "obs_bias", 0.0)]
        for s in spec["levels"]:
            keep = [seed for seed in EVAL_SEEDS if (dom, s, seed) in arms]
            if len(keep) < len(EVAL_SEEDS) * 0.5:
                continue
            traj = {seed: arms[(dom, s, seed)] for seed in keep}
            F, L, G = windows_for(traj, keep)
            for m in MODELS:
                rw, rc = score(F, L, G, m)
                fw = np.full(len(EVAL_SEEDS), np.nan)
                for i, seed in enumerate(keep):
                    fw[EVAL_SEEDS.index(seed)] = rw[i]
                tab_w[(m, dom, s)] = fw
                tab_c[(m, dom, s)] = rc
    log("scored plant-side domains")

    domains = {"obs_bias": OBS_GRID, "obs_noise": OBS_GRID,
               **{d: [0.0] + spec["levels"] for d, spec in PLANT.items()}}

    out = {"decision_channels": {str(c): {"name": DECISION[c][0],
                                          "tol": DECISION[c][1]}
                                 for c in dch},
           "nominal_admissibility": {}, "absolute_trips": [],
           "radii": {}, "worst_margin": {}}

    for m in MODELS:
        trips = {DECISION[c][0]: tab_c[(m, "obs_bias", 0.0)][c] for c in dch
                 if tab_c[(m, "obs_bias", 0.0)][c] > DECISION[c][1]}
        out["nominal_admissibility"][m] = (
            "inadmissible" if trips else "admissible")
        if trips:
            log(f"  NOMINAL TRIP {m}: {trips}")

    for (m, dom, lv), ch in tab_c.items():
        for c in dch:
            if ch[c] > DECISION[c][1]:
                out["absolute_trips"].append(
                    {"model": m, "domain": dom, "severity": lv,
                     "channel": DECISION[c][0], "rmse": ch[c],
                     "tol": DECISION[c][1]})

    for m in MODELS:
        for dom, levels in domains.items():
            lv_have = [lv for lv in levels if (m, dom, lv) in tab_w]
            r0 = tab_w[(m, dom, lv_have[0])]
            rel = conj = None
            for i, lv in enumerate(lv_have[1:], 1):
                rs = tab_w[(m, dom, lv)]
                ok = np.isfinite(rs) & np.isfinite(r0)
                p_rel = np.mean(rs[ok] / r0[ok] - 1 > TAU)
                abs_trip = any(tab_c[(m, dom, lv)][c] > DECISION[c][1]
                               for c in dch)
                if p_rel >= ALPHA and rel is None:
                    rel = 0.5 * (lv_have[i - 1] + lv)
                if p_rel >= ALPHA and abs_trip and conj is None:
                    conj = 0.5 * (lv_have[i - 1] + lv)
            out["radii"][f"{m}|{dom}"] = {"relative_only": rel,
                                          "conjunction": conj}

    for c in dch:
        worst = max(tab_c[k][c] for k in tab_c)
        wk = max(tab_c, key=lambda k: tab_c[k][c])
        out["worst_margin"][DECISION[c][0]] = {
            "max_rmse": worst, "tol": DECISION[c][1],
            "ratio": worst / DECISION[c][1], "at": f"{wk[0]}|{wk[1]}|{wk[2]}"}
        log(f"  margin {DECISION[c][0]}: max {worst:.2f} vs tol "
            f"{DECISION[c][1]} (ratio {worst/DECISION[c][1]:.3f}) at {wk}")

    n_diff = sum(1 for v in out["radii"].values()
                 if v["relative_only"] != v["conjunction"])
    log(f"absolute trips: {len(out['absolute_trips'])}; radii differing: "
        f"{n_diff}/{len(out['radii'])}")

    with open(OUT / "tau_abs_check.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    log("wrote tau_abs_check.json")

if __name__ == "__main__":
    main()
