"""Task-matched predictor panel for the MBR 80-step forecasting task.

Forecasts 80 steps of the 63 tracked state dimensions from 20 steps of
history plus the future 80 steps of actions and disturbances.

Targets are increments from the last observed state normalized by the
project's scale_x, so persistence is exactly the zero prediction and the
training loss is aligned with the reported metric.

Split is by sequence over draw.pt and declared before training.
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

WW_ROOT = os.environ.get("MBR_PROJECT_ROOT")
if not WW_ROOT:
    raise SystemExit("set MBR_PROJECT_ROOT to the MBR project checkout; see README")
HERE = Path(__file__).resolve().parent
OUT = HERE / "artifacts"

SEED = 42
HIST, HORIZON = 20, 80
STRIDE = 10
TRAIN_SEQS = list(range(0, 24))
VAL_SEQS = list(range(24, 30))
TEST_SEQS = list(range(30, 38))

def build(seqs, states, actions, dist, track, sx, cx, su, cu, stride):
    """Feature/target tensors for the given sequence indices."""
    F, Y, last = [], [], []
    need = HIST + HORIZON
    for si in seqs:
        st, ac = states[si], actions[si]
        T = st.shape[0]
        sn = (st - sx) / cx
        an = (ac - su[:4]) / cu[:4]
        dn = (dist[:T] - su[4:4 + dist.shape[1]]) / (cu[4:4 + dist.shape[1]] + 1e-8)
        t = 0
        while t + need <= T:
            h = slice(t, t + HIST)
            f = slice(t + HIST, t + HIST + HORIZON)
            hs = sn[h][:, track]
            F.append(np.concatenate([
                hs.reshape(-1), an[h].reshape(-1), dn[h].reshape(-1),
                an[f].reshape(-1), dn[f].reshape(-1)]))
            Y.append((sn[f][:, track] - hs[-1]).reshape(-1))
            last.append(hs[-1])
            t += stride
    return (np.asarray(F, np.float32), np.asarray(Y, np.float32),
            np.asarray(last, np.float32))

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
    """GRU encoder over history, GRU decoder driven by future inputs."""

    def __init__(self, n_state, n_exo, hid=192):
        super().__init__()
        self.n_state, self.n_exo, self.hid = n_state, n_exo, hid
        self.enc = nn.GRU(n_state + n_exo, hid, batch_first=True)
        self.dec = nn.GRU(n_exo, hid, batch_first=True)
        self.head = nn.Linear(hid, n_state)

    def forward(self, x):
        n = x.shape[0]
        ns, ne = self.n_state, self.n_exo
        i = 0
        hs = x[:, i:i + HIST * ns].reshape(n, HIST, ns); i += HIST * ns
        ha = x[:, i:i + HIST * 4].reshape(n, HIST, 4); i += HIST * 4
        hd = x[:, i:i + HIST * 10].reshape(n, HIST, 10); i += HIST * 10
        fa = x[:, i:i + HORIZON * 4].reshape(n, HORIZON, 4); i += HORIZON * 4
        fd = x[:, i:i + HORIZON * 10].reshape(n, HORIZON, 10)
        _, h = self.enc(torch.cat([hs, ha, hd], 2))
        out, _ = self.dec(torch.cat([fa, fd], 2), h)
        return self.head(out).reshape(n, HORIZON * ns)

def fit(model, Xtr, Ytr, Xva, Yva, epochs, bs, lr, tag, wd=1e-4):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
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
            v = float(np.mean([nn.functional.mse_loss(model(Xv[i:i + 256]),
                                                      Yv[i:i + 256]).item()
                               for i in range(0, len(Xv), 256)]))
        if v < best:
            best, state = v, {k: p.detach().clone()
                              for k, p in model.state_dict().items()}
        if (ep + 1) % 10 == 0:
            print(f"    {tag} ep{ep+1}: val_mse={v:.5e} best={best:.5e}", flush=True)
    model.load_state_dict(state)
    return model

def main():
    os.chdir(WW_ROOT)
    sys.path.insert(0, WW_ROOT)
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    import contextlib
    import io

    import args_new as new_args
    from experiments.process_disturbances import load_model_and_data
    from utils.disturbance_handler import get_default_disturbance_handler

    # pred_tracking is injected by the project loader, not present in raw args
    args = dict(new_args.args)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _m, _td, _env, _nm = load_model_and_data("mamba", "MBR", args)
    del _m, _td, _env, _nm
    track = np.asarray(args["pred_tracking"], dtype=int)
    d = torch.load("dataset/MBR/0/draw.pt", weights_only=False)
    states = [s.numpy().astype(np.float32) for s, _ in d]
    actions = [a.numpy().astype(np.float32) for _, a in d]
    f = "save_model/mamba/MBR"
    sx, cx, su, cu = (np.loadtxt(f"{f}/{k}.txt", dtype=np.float32)
                      for k in ("shift_x", "scale_x", "shift_u", "scale_u"))
    dh = get_default_disturbance_handler()
    dist = dh.sample_window(window_length=states[0].shape[0], start_idx=0,
                            normalized=False).astype(np.float32)[:, :10]

    t0 = time.time()
    Xtr, Ytr, _ = build(TRAIN_SEQS, states, actions, dist, track,
                        sx, cx, su, cu, STRIDE)
    Xva, Yva, _ = build(VAL_SEQS, states, actions, dist, track,
                        sx, cx, su, cu, STRIDE)
    Xte, Yte, _ = build(TEST_SEQS, states, actions, dist, track,
                        sx, cx, su, cu, STRIDE * 2)
    print(f"windows: train {Xtr.shape[0]}, val {Xva.shape[0]}, test {Xte.shape[0]}")
    print(f"features {Xtr.shape[1]}, targets {Ytr.shape[1]}  ({time.time()-t0:.1f}s)")

    scale_tr = cx[track]

    def phys_rmse(pred_inc, Y):
        """Physical-unit RMSE from normalized increments."""
        e = (pred_inc - Y).reshape(-1, HORIZON, len(track)) * scale_tr
        return float(np.sqrt(np.mean(e ** 2)))

    report, preds = {}, {}
    pers = np.zeros_like(Yte)
    report["persistence"] = {"test_rmse": phys_rmse(pers, Yte), "skill": 1.0,
                             "diverged": 0}
    p0 = report["persistence"]["test_rmse"]
    print(f"  persistence test RMSE = {p0:.2f}")

    lam = 10.0
    A = np.concatenate([Xtr, np.ones((len(Xtr), 1), np.float32)], 1)
    W = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1], dtype=np.float32),
                        A.T @ Ytr)
    Ate = np.concatenate([Xte, np.ones((len(Xte), 1), np.float32)], 1)
    pr = Ate @ W
    preds["ridge"] = pr
    r = phys_rmse(pr, Yte)
    report["ridge"] = {"test_rmse": r, "skill": p0 / r, "diverged": 0}
    np.save(OUT / "ridge_W.npy", W)
    print(f"  ridge       test RMSE = {r:.2f}  skill {p0/r:.2f}")

    din, dout = Xtr.shape[1], Ytr.shape[1]
    mlp = fit(MLP(din, dout), Xtr, Ytr, Xva, Yva, 60, 128, 1e-3, "mlp")
    with torch.no_grad():
        pr = np.concatenate([mlp(torch.from_numpy(Xte[i:i + 256])).numpy()
                             for i in range(0, len(Xte), 256)])
    preds["mlp"] = pr
    r = phys_rmse(pr, Yte)
    report["mlp"] = {"test_rmse": r, "skill": p0 / r, "diverged": 0}
    torch.save(mlp.state_dict(), OUT / "mlp.pt")
    print(f"  mlp         test RMSE = {r:.2f}  skill {p0/r:.2f}")

    s2s = fit(Seq2Seq(len(track), 14), Xtr, Ytr, Xva, Yva, 60, 128, 2e-3, "gru")
    with torch.no_grad():
        pr = np.concatenate([s2s(torch.from_numpy(Xte[i:i + 256])).numpy()
                             for i in range(0, len(Xte), 256)])
    preds["gru"] = pr
    r = phys_rmse(pr, Yte)
    report["gru"] = {"test_rmse": r, "skill": p0 / r, "diverged": 0}
    torch.save(s2s.state_dict(), OUT / "gru.pt")
    print(f"  gru         test RMSE = {r:.2f}  skill {p0/r:.2f}")

    meta = {"hist": HIST, "horizon": HORIZON, "stride": STRIDE,
            "train_seqs": TRAIN_SEQS, "val_seqs": VAL_SEQS,
            "test_seqs": TEST_SEQS, "seed": SEED,
            "n_windows": {"train": int(len(Xtr)), "val": int(len(Xva)),
                          "test": int(len(Xte))},
            "report": report}
    with open(OUT / "panel_report.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    np.savez_compressed(OUT / "test_preds.npz", Yte=Yte, **preds)
    print(f"done in {time.time()-t0:.0f}s -> {OUT}")

if __name__ == "__main__":
    main()
