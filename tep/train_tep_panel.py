"""TEP panel: persistence, ridge, MLP, GRU, with the protocol admission
skill gate (bootstrap CI lower bound above 1 for linear and nonlinear
members, across seeds).

Horizon fixed at 40 samples (2 h) from probe_horizon.py (ridge skill 1.78,
the maximum over the probed horizons).
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tep_task as T

OUT = HERE / "artifacts"
HORIZON = 40
STRIDE_TR, STRIDE_EV = 10, 20
SEEDS = [int(s) for s in
         os.environ.get("RBM_SEEDS", "42,123,7,2024,31337").split(",")]
N_BOOT = 1000


class MLP(nn.Module):
    def __init__(self, din, dout, h=384):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(din, h), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(h, h), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(h, dout))

    def forward(self, x):
        return self.net(x)


class Seq2Seq(nn.Module):
    """GRU encoder over history; decoder driven by a learned step embedding
    (no future exogenous inputs exist in this closed-loop task)."""

    def __init__(self, n_in=33, n_out=22, hid=192):
        super().__init__()
        self.n_in, self.n_out = n_in, n_out
        self.enc = nn.GRU(n_in, hid, batch_first=True)
        self.step_emb = nn.Parameter(torch.randn(HORIZON, 16) * 0.1)
        self.dec = nn.GRU(16, hid, batch_first=True)
        self.head = nn.Linear(hid, n_out)

    def forward(self, x):
        n = x.shape[0]
        seq = x.reshape(n, T.HIST, self.n_in)
        _, h = self.enc(seq)
        dec_in = self.step_emb.unsqueeze(0).expand(n, -1, -1)
        out, _ = self.dec(dec_in, h)
        return self.head(out).reshape(n, HORIZON * self.n_out)


def fit(model, Xtr, Ytr, Xva, Yva, epochs, bs, lr, tag):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    Xt, Yt = torch.from_numpy(Xtr), torch.from_numpy(Ytr)
    Xv, Yv = torch.from_numpy(Xva), torch.from_numpy(Yva)
    best, state = np.inf, None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            nn.functional.mse_loss(model(Xt[idx]), Yt[idx]).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sch.step()
        model.eval()
        with torch.no_grad():
            v = float(np.mean([nn.functional.mse_loss(
                model(Xv[i:i + 512]), Yv[i:i + 512]).item()
                for i in range(0, len(Xv), 512)]))
        if v < best:
            best, state = v, {k: p.detach().clone()
                              for k, p in model.state_dict().items()}
        if (ep + 1) % 20 == 0:
            print(f"    {tag} ep{ep+1}: val={v:.5e} best={best:.5e}", flush=True)
    model.load_state_dict(state)
    return model


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print("generating episodes", flush=True)
    eps = {k: T.gen_episodes(v) for k, v in T.SEEDS.items()}
    for k, v in eps.items():
        print(f"  {k}: {len(v)}", flush=True)
    shift, scale = T.fit_normalization(eps["train"])
    np.savez(OUT / "normalization.npz", shift=shift, scale=scale)
    np.savez_compressed(OUT / "episodes.npz",
                        **{f"{k}_{s}": x for k, v in eps.items() for s, x in v})

    Xtr, Ytr = T.build_windows(eps["train"], HORIZON, STRIDE_TR, shift, scale)
    Xva, Yva = T.build_windows(eps["val"], HORIZON, STRIDE_TR, shift, scale)
    Xte, Yte = T.build_windows(eps["eval"], HORIZON, STRIDE_EV, shift, scale)
    print(f"windows: train {len(Xtr)}, val {len(Xva)}, eval {len(Xte)} "
          f"({time.time()-t0:.0f}s)", flush=True)

    # per-window RMSE in normalized units (uniform scale; declared aggregation:
    # mean over windows; pooled also reported)
    def rmse_w(pred):
        return np.sqrt(np.mean((pred - Yte).reshape(len(Yte), -1) ** 2, axis=1))

    r_pers = rmse_w(np.zeros_like(Yte))
    rng = np.random.default_rng(0)
    bidx = rng.integers(0, len(Yte), size=(N_BOOT, len(Yte)))

    def skill_ci(r_model):
        pt = r_pers.mean() / r_model.mean()
        bs = [r_pers[i].mean() / r_model[i].mean() for i in bidx]
        return pt, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

    report = {"horizon": HORIZON, "n_windows": {"train": int(len(Xtr)),
              "val": int(len(Xva)), "eval": int(len(Xte))},
              "persistence_rmse_mean": float(r_pers.mean()), "models": {}}

    lam = 10.0
    A = np.concatenate([Xtr, np.ones((len(Xtr), 1), np.float32)], 1)
    W = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1], dtype=np.float32),
                        A.T @ Ytr)
    Ate = np.concatenate([Xte, np.ones((len(Xte), 1), np.float32)], 1)
    r = rmse_w(Ate @ W)
    pt, lo, hi = skill_ci(r)
    report["models"]["ridge"] = {"rmse": float(r.mean()), "skill": pt,
                                 "skill_ci": [lo, hi]}
    np.save(OUT / "ridge_W.npy", W)
    print(f"  ridge  skill {pt:.2f} CI [{lo:.2f},{hi:.2f}]", flush=True)

    din, dout = Xtr.shape[1], Ytr.shape[1]
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        sub = OUT / f"seed_{seed}"
        sub.mkdir(exist_ok=True)
        mlp = fit(MLP(din, dout), Xtr, Ytr, Xva, Yva, 60, 256, 1e-3,
                  f"mlp_s{seed}")
        with torch.no_grad():
            pr = np.concatenate([mlp(torch.from_numpy(Xte[i:i + 512])).numpy()
                                 for i in range(0, len(Xte), 512)])
        r = rmse_w(pr)
        pt, lo, hi = skill_ci(r)
        report["models"][f"mlp_s{seed}"] = {"rmse": float(r.mean()),
                                            "skill": pt, "skill_ci": [lo, hi]}
        torch.save(mlp.state_dict(), sub / "mlp.pt")
        print(f"  mlp_s{seed}  skill {pt:.2f} CI [{lo:.2f},{hi:.2f}]", flush=True)

        torch.manual_seed(seed)
        gru = fit(Seq2Seq(), Xtr, Ytr, Xva, Yva, 60, 256, 2e-3, f"gru_s{seed}")
        with torch.no_grad():
            pr = np.concatenate([gru(torch.from_numpy(Xte[i:i + 512])).numpy()
                                 for i in range(0, len(Xte), 512)])
        r = rmse_w(pr)
        pt, lo, hi = skill_ci(r)
        report["models"][f"gru_s{seed}"] = {"rmse": float(r.mean()),
                                            "skill": pt, "skill_ci": [lo, hi]}
        torch.save(gru.state_dict(), sub / "gru.pt")
        print(f"  gru_s{seed}  skill {pt:.2f} CI [{lo:.2f},{hi:.2f}]", flush=True)

    # protocol gate: linear AND nonlinear with CI lower bound > 1, across seeds
    lin_ok = report["models"]["ridge"]["skill_ci"][0] > 1.0
    mlp_ok = all(report["models"][f"mlp_s{s}"]["skill_ci"][0] > 1.0
                 for s in SEEDS)
    gru_ok = all(report["models"][f"gru_s{s}"]["skill_ci"][0] > 1.0
                 for s in SEEDS)
    report["admission_gate"] = {"linear_pass": bool(lin_ok),
                                "nonlinear_pass": bool(mlp_ok or gru_ok),
                                "mlp_all_seeds": bool(mlp_ok),
                                "gru_all_seeds": bool(gru_ok),
                                "PASS": bool(lin_ok and (mlp_ok or gru_ok))}
    with open(OUT / "panel_report.json", "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"\nADMISSION GATE: {report['admission_gate']}", flush=True)
    print(f"done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
