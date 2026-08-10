"""CSTR data generation and predictor panel for the declared protocol.

Task (uninformed condition): from 20 samples of measured [C_A, T], commanded
T_c, and measured feed [C_Af, T_f], plus the future 10 commanded T_c, forecast
the next 10 samples of [C_A, T].

Targets are increments relative to the last observed state, normalized by the
protocol reference scales r_CA = 10, r_T = 80. The training loss is then
algebraically 2 * J_fid^2, so training and evaluation use one metric.

Panel: persistence, ridge-linear, MLP, GRU.
Splits by episode: train / clean-validation / calibration / frozen evaluation.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import cstr_system as C

HERE = Path(__file__).resolve().parent
OUT = HERE / "artifacts"   # run products, gitignored
SEED = 42

N_TRAIN_EP, N_VAL_EP, N_CAL_EP, N_EVAL_EP = 200, 40, 40, 80
BURN = int(C.BURN_IN_H / C.DT)          # 80 samples
EP_LEN = BURN + 200
STRIDE = 10
REGIME_TRAIN_MAX = 0.10                  # declared training-support limit

CA_REF, T_REF = C.ANCHOR_L["CA"], C.ANCHOR_L["T"]
TC_REF, TC_SCALE = 292.0, 49.0


def norm_state(x):
    x = np.asarray(x, dtype=float)
    return np.stack([(x[..., 0] - CA_REF) / C.R_CA,
                     (x[..., 1] - T_REF) / C.R_T], axis=-1)


def gen_episodes(n, rng, regime_lo=0.0, regime_hi=REGIME_TRAIN_MAX):
    eps = []
    while len(eps) < n:
        s = rng.uniform(regime_lo, regime_hi)
        x0, CAf, Tf, Tc = C.sample_episode(rng, EP_LEN, regime_s=s)
        xs, ok = C.rollout(x0, CAf, Tf, Tc)
        if not ok:
            continue
        adm, _ = C.admissible(xs, Tc)
        if not adm:
            continue
        eps.append({"x": xs[:EP_LEN], "CAf": CAf, "Tf": Tf, "Tc": Tc,
                    "regime_s": float(s)})
    return eps


def windows_from(eps, start=BURN, stride=STRIDE):
    """Feature/target pairs. Window layout: HIST history then HORIZON future."""
    X, Y, meta = [], [], []
    need = C.HIST + C.HORIZON
    for ei, e in enumerate(eps):
        t = start
        while t + need <= len(e["x"]):
            h = slice(t, t + C.HIST)
            f = slice(t + C.HIST, t + C.HIST + C.HORIZON)
            hs = norm_state(e["x"][h])
            last = hs[-1]
            feat = np.concatenate([
                hs.reshape(-1),
                (e["Tc"][h] - TC_REF) / TC_SCALE,
                (e["CAf"][h] - 10.0) / 1.0,
                (e["Tf"][h] - 300.0) / 10.0,
                (e["Tc"][f] - TC_REF) / TC_SCALE,
            ])
            tgt = (norm_state(e["x"][f]) - last).reshape(-1)
            X.append(feat)
            Y.append(tgt)
            meta.append({"ep": ei, "t": t})
            t += stride
    return np.asarray(X, np.float32), np.asarray(Y, np.float32), meta


class MLP(nn.Module):
    def __init__(self, din, dout, h=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, h), nn.ReLU(),
                                 nn.Linear(h, h), nn.ReLU(),
                                 nn.Linear(h, dout))

    def forward(self, x):
        return self.net(x)


class GRUNet(nn.Module):
    """History through a GRU; future commands and the encoding through a head."""

    def __init__(self, dout, hid=96):
        super().__init__()
        self.gru = nn.GRU(input_size=5, hidden_size=hid, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hid + C.HORIZON, 128), nn.ReLU(),
                                  nn.Linear(128, dout))

    def forward(self, x):
        n = x.shape[0]
        hs = x[:, :C.HIST * 2].reshape(n, C.HIST, 2)
        tc = x[:, C.HIST * 2:C.HIST * 3].reshape(n, C.HIST, 1)
        caf = x[:, C.HIST * 3:C.HIST * 4].reshape(n, C.HIST, 1)
        tf = x[:, C.HIST * 4:C.HIST * 5].reshape(n, C.HIST, 1)
        seq = torch.cat([hs, tc, caf, tf], dim=2)
        _, h = self.gru(seq)
        return self.head(torch.cat([h[-1], x[:, C.HIST * 5:]], dim=1))


def train_torch(model, Xtr, Ytr, Xva, Yva, epochs=200, bs=256, lr=1e-3, tag=""):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    Xtr_t, Ytr_t = torch.from_numpy(Xtr), torch.from_numpy(Ytr)
    Xva_t, Yva_t = torch.from_numpy(Xva), torch.from_numpy(Yva)
    best, best_state = np.inf, None
    n = len(Xtr_t)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(Xtr_t[idx]), Ytr_t[idx])
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            v = nn.functional.mse_loss(model(Xva_t), Yva_t).item()
        if v < best:
            best = v
            best_state = {k: p.detach().clone() for k, p in model.state_dict().items()}
        if (ep + 1) % 50 == 0:
            print(f"    {tag} epoch {ep+1}: val_mse={v:.6e} best={best:.6e}", flush=True)
    model.load_state_dict(best_state)
    return model, best


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    t0 = time.time()

    print("generating episodes")
    splits = {}
    for name, n in [("train", N_TRAIN_EP), ("val", N_VAL_EP),
                    ("calib", N_CAL_EP), ("eval", N_EVAL_EP)]:
        splits[name] = gen_episodes(n, rng)
        print(f"  {name}: {len(splits[name])} episodes")
    print(f"  ({time.time()-t0:.1f}s)")

    Xtr, Ytr, _ = windows_from(splits["train"])
    Xva, Yva, _ = windows_from(splits["val"])
    print(f"windows: train {Xtr.shape} val {Xva.shape}")

    def jfid_of(pred, tgt):
        """pred/tgt are normalized increments; J_fid is RMS over both dims."""
        return float(np.sqrt(np.mean((pred - tgt) ** 2)))

    report = {"n_windows_train": int(len(Xtr)), "n_windows_val": int(len(Xva))}

    # persistence
    p_val = jfid_of(np.zeros_like(Yva), Yva)
    report["persistence_val_jfid"] = p_val
    print(f"  persistence val J_fid = {p_val:.5f}")

    # ridge
    lam = 1e-3
    A = np.concatenate([Xtr, np.ones((len(Xtr), 1), np.float32)], 1)
    W = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1], dtype=np.float32),
                        A.T @ Ytr)
    Ava = np.concatenate([Xva, np.ones((len(Xva), 1), np.float32)], 1)
    r_val = jfid_of(Ava @ W, Yva)
    report["ridge_val_jfid"] = r_val
    np.save(OUT / "ridge_W.npy", W)
    print(f"  ridge       val J_fid = {r_val:.5f}")

    din, dout = Xtr.shape[1], Ytr.shape[1]
    mlp, mlp_v = train_torch(MLP(din, dout), Xtr, Ytr, Xva, Yva, tag="mlp")
    torch.save(mlp.state_dict(), OUT / "mlp.pt")
    report["mlp_val_jfid"] = float(np.sqrt(mlp_v))
    print(f"  mlp         val J_fid = {np.sqrt(mlp_v):.5f}")

    gru, gru_v = train_torch(GRUNet(dout), Xtr, Ytr, Xva, Yva, tag="gru")
    torch.save(gru.state_dict(), OUT / "gru.pt")
    report["gru_val_jfid"] = float(np.sqrt(gru_v))
    print(f"  gru         val J_fid = {np.sqrt(gru_v):.5f}")

    np.savez_compressed(OUT / "episodes.npz",
                        **{f"{k}_{i}_{f}": v[f]
                           for k, eps in splits.items()
                           for i, v in enumerate(eps)
                           for f in ("x", "CAf", "Tf", "Tc")},
                        index=json.dumps({k: len(v) for k, v in splits.items()}))
    with open(OUT / "train_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"done in {time.time()-t0:.1f}s -> {OUT}")


if __name__ == "__main__":
    main()
