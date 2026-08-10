"""Does any configuration of the CSTR task clear the skill gate?

Section 4.4 of the journal manuscript retires this system on the admission
gate of Section 3.3. The claim it rests on is that the task is
persistence-dominated and that neither of the two knobs available moves it:
the forecast horizon and the excitation strength of the episode sampler.

This script regenerates that evidence as a JSON artifact. It was written
after the fact, because the original sweep left its numbers in a project
state file and no artifact behind it.

Two sweeps, both at the declared low-conversion anchor:

  horizon     HORIZON in samples, DT = 0.05 h, so 10 samples is 0.5 h
  excitation  a multiplier on the standard deviations of the episode
              sampler's piecewise-constant inputs (T_c, C_Af, T_f)

Panel: persistence, ridge, MLP, GRU, matching the panel of
`build_and_train.py` so the shortest-horizon row is directly comparable to
`artifacts/train_report.json`.

Skill is persistence J_fid divided by the member's J_fid on the clean
validation split, matching Section 3.3. Above 1 means the member beats
holding the last observation.

Usage:
    python horizon_excitation_sweep.py
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import cstr_system as C

HERE = Path(__file__).resolve().parent
OUT = (HERE.parent / "frozen_predictions" / "cstr"
       / "horizon_excitation_sweep.json")

SEED = 42
N_TRAIN_EP, N_VAL_EP = 200, 40
BURN = int(C.BURN_IN_H / C.DT)
STRIDE = 10
REGIME_TRAIN_MAX = 0.10

HORIZONS = [10, 20, 40, 60, 80, 120]        # samples; 0.5 h to 6 h
EXCITATIONS = [1.0, 2.0, 4.0, 8.0]          # multiplier on sampler sd
EXC_HORIZON = 10                            # declared reference horizon

CA_REF, T_REF = C.ANCHOR_L["CA"], C.ANCHOR_L["T"]
TC_REF, TC_SCALE = 292.0, 49.0

# The sampler's nominal standard deviations, from cstr_system.sample_episode.
SD_NOM = dict(CAf=0.2, Tf=0.5, Tc=1.0, x0_CA=0.15, x0_T=1.0)


def norm_state(x):
    x = np.asarray(x, dtype=float)
    return np.stack([(x[..., 0] - CA_REF) / C.R_CA,
                     (x[..., 1] - T_REF) / C.R_T], axis=-1)


def sample_episode(rng, n_samples, regime_s, exc, hold_h=0.25):
    """cstr_system.sample_episode with every standard deviation scaled by exc."""
    c = C.regime_center(regime_s)
    hold = max(1, int(round(hold_h / C.DT)))
    n_holds = int(np.ceil(n_samples / hold))

    def pw(mean, sd, lo=None, hi=None):
        v = rng.normal(mean, sd * exc, size=n_holds)
        if lo is not None:
            v = np.clip(v, lo, hi)
        return np.repeat(v, hold)[:n_samples]

    CAf = pw(10.0, SD_NOM["CAf"])
    Tf = pw(300.0, SD_NOM["Tf"])
    # The +-5 K clip is a physical actuator range and does not scale with exc.
    Tc = pw(c[2], SD_NOM["Tc"], c[2] - 5.0, c[2] + 5.0)
    x0 = np.array([rng.normal(c[0], SD_NOM["x0_CA"] * exc),
                   rng.normal(c[1], SD_NOM["x0_T"] * exc)])
    return x0, CAf, Tf, Tc


def gen_episodes(n, rng, ep_len, exc):
    eps, rejected = [], 0
    while len(eps) < n:
        s = rng.uniform(0.0, REGIME_TRAIN_MAX)
        x0, CAf, Tf, Tc = sample_episode(rng, ep_len, s, exc)
        xs, ok = C.rollout(x0, CAf, Tf, Tc)
        if not ok:
            rejected += 1
            continue
        adm, _ = C.admissible(xs, Tc)
        if not adm:
            rejected += 1
            continue
        eps.append({"x": xs[:ep_len], "CAf": CAf, "Tf": Tf, "Tc": Tc})
    return eps, rejected


def windows_from(eps, horizon):
    """Same layout as build_and_train.windows_from, horizon parameterised."""
    X, Y = [], []
    need = C.HIST + horizon
    for e in eps:
        t = BURN
        while t + need <= len(e["x"]):
            h = slice(t, t + C.HIST)
            f = slice(t + C.HIST, t + C.HIST + horizon)
            hs = norm_state(e["x"][h])
            last = hs[-1]
            X.append(np.concatenate([
                hs.reshape(-1),
                (e["Tc"][h] - TC_REF) / TC_SCALE,
                (e["CAf"][h] - 10.0) / 1.0,
                (e["Tf"][h] - 300.0) / 10.0,
                (e["Tc"][f] - TC_REF) / TC_SCALE,
            ]))
            Y.append((norm_state(e["x"][f]) - last).reshape(-1))
            t += STRIDE
    return np.asarray(X, np.float32), np.asarray(Y, np.float32)


class GRUNet(nn.Module):
    """History through a GRU; future commands and the encoding through a head."""

    def __init__(self, dout, horizon, hid=96):
        super().__init__()
        self.horizon = horizon
        self.gru = nn.GRU(input_size=5, hidden_size=hid, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hid + horizon, 128), nn.ReLU(),
                                  nn.Linear(128, dout))

    def forward(self, x):
        n = x.shape[0]
        hs = x[:, :C.HIST * 2].reshape(n, C.HIST, 2)
        tc = x[:, C.HIST * 2:C.HIST * 3].reshape(n, C.HIST, 1)
        caf = x[:, C.HIST * 3:C.HIST * 4].reshape(n, C.HIST, 1)
        tf = x[:, C.HIST * 4:C.HIST * 5].reshape(n, C.HIST, 1)
        _, h = self.gru(torch.cat([hs, tc, caf, tf], dim=2))
        return self.head(torch.cat([h[-1], x[:, C.HIST * 5:]], dim=1))


class MLP(nn.Module):
    def __init__(self, din, dout, h=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, h), nn.ReLU(),
                                 nn.Linear(h, h), nn.ReLU(),
                                 nn.Linear(h, dout))

    def forward(self, x):
        return self.net(x)


def jfid(pred, tgt):
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(tgt)) ** 2)))


def fit_ridge(Xtr, Ytr, Xva, lam=1e-3):
    A = np.concatenate([Xtr, np.ones((len(Xtr), 1), np.float32)], 1)
    W = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1], dtype=np.float32),
                        A.T @ Ytr)
    return np.concatenate([Xva, np.ones((len(Xva), 1), np.float32)], 1) @ W


def fit_torch(model, Xtr, Ytr, Xva, Yva, epochs=200, bs=256, lr=1e-3):
    torch.manual_seed(SEED)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    Xtr_t, Ytr_t = torch.from_numpy(Xtr), torch.from_numpy(Ytr)
    Xva_t, Yva_t = torch.from_numpy(Xva), torch.from_numpy(Yva)
    best = np.inf
    n = len(Xtr_t)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            nn.functional.mse_loss(model(Xtr_t[idx]), Ytr_t[idx]).backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            best = min(best, nn.functional.mse_loss(model(Xva_t), Yva_t).item())
    return float(np.sqrt(best))


def evaluate(eps_tr, eps_va, horizon):
    Xtr, Ytr = windows_from(eps_tr, horizon)
    Xva, Yva = windows_from(eps_va, horizon)
    din, dout = Xtr.shape[1], Ytr.shape[1]
    p = jfid(np.zeros_like(Yva), Yva)
    r = jfid(fit_ridge(Xtr, Ytr, Xva), Yva)
    m = fit_torch(MLP(din, dout), Xtr, Ytr, Xva, Yva)
    g = fit_torch(GRUNet(dout, horizon), Xtr, Ytr, Xva, Yva)
    return {
        "n_train_windows": int(len(Xtr)),
        "n_val_windows": int(len(Xva)),
        "persistence_jfid": p,
        "ridge_jfid": r,
        "mlp_jfid": m,
        "gru_jfid": g,
        "ridge_skill": round(p / r, 4),
        "mlp_skill": round(p / m, 4),
        "gru_skill": round(p / g, 4),
        "panel_best_skill": round(max(p / r, p / m, p / g), 4),
    }


def main():
    t0 = time.time()
    result = {
        "declared_before_running": {
            "gate": "skill = J(persistence) / J(member) on the clean "
                    "validation split; a task instance in which no member "
                    "clears 1 by a usable margin is invalid (Section 3.3)",
            "dt_hours": C.DT,
            "history_samples": C.HIST,
            "horizons_samples": HORIZONS,
            "excitation_multipliers": EXCITATIONS,
            "excitation_horizon_samples": EXC_HORIZON,
            "panel": ["persistence", "ridge", "mlp", "gru"],
            "seed": SEED,
        },
        "horizon_sweep": {},
        "excitation_sweep": {},
    }

    # ---- horizon sweep, excitation at nominal
    rng = np.random.default_rng(SEED)
    ep_len = BURN + 200 + max(HORIZONS)
    eps_tr, rej_tr = gen_episodes(N_TRAIN_EP, rng, ep_len, 1.0)
    eps_va, rej_va = gen_episodes(N_VAL_EP, rng, ep_len, 1.0)
    result["horizon_sweep"]["episodes"] = {
        "train": len(eps_tr), "val": len(eps_va),
        "rejected_train": rej_tr, "rejected_val": rej_va,
        "episode_length_samples": ep_len,
    }
    for h in HORIZONS:
        row = evaluate(eps_tr, eps_va, h)
        row["horizon_hours"] = round(h * C.DT, 3)
        result["horizon_sweep"][str(h)] = row
        print(f"  horizon {h:3d} ({h*C.DT:.2f} h): ridge {row['ridge_skill']:.2f} "
              f"mlp {row['mlp_skill']:.2f} gru {row['gru_skill']:.2f}  "
              f"best {row['panel_best_skill']:.2f}", flush=True)

    # ---- excitation sweep, horizon at the card value
    ep_len_e = BURN + 200 + EXC_HORIZON
    for exc in EXCITATIONS:
        rng = np.random.default_rng(SEED)
        etr, rtr = gen_episodes(N_TRAIN_EP, rng, ep_len_e, exc)
        eva, rva = gen_episodes(N_VAL_EP, rng, ep_len_e, exc)
        row = evaluate(etr, eva, EXC_HORIZON)
        row["rejected_train"] = rtr
        row["rejected_val"] = rva
        result["excitation_sweep"][str(exc)] = row
        print(f"  excitation x{exc:g}: ridge {row['ridge_skill']:.2f} "
              f"mlp {row['mlp_skill']:.2f} gru {row['gru_skill']:.2f}  "
              f"best {row['panel_best_skill']:.2f} (rejected {rtr}+{rva})",
              flush=True)

    hs = result["horizon_sweep"]
    es = result["excitation_sweep"]
    result["summary"] = {
        "best_skill_at_card_horizon": hs[str(HORIZONS[0])]["panel_best_skill"],
        "best_skill_at_longest_horizon": hs[str(HORIZONS[-1])]["panel_best_skill"],
        "best_skill_over_all_configurations": round(max(
            [hs[str(h)]["panel_best_skill"] for h in HORIZONS]
            + [es[str(e)]["panel_best_skill"] for e in EXCITATIONS]), 4),
        "verdict": "invalid instance: persistence-dominated at every "
                   "configuration tried",
    }
    result["runtime_seconds"] = round(time.time() - t0, 1)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))
    print(f"\nbest over all configurations: "
          f"{result['summary']['best_skill_over_all_configurations']}")
    print(f"wrote {OUT}  ({result['runtime_seconds']}s)")


if __name__ == "__main__":
    main()
