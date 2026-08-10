"""Does the TEP task survive a history window long enough for a foundation model?

Zero-shot forecasters are built for hundreds of context points, and this task
gives them 20. Before rebuilding the task around a longer window it is worth
knowing whether trained models do better, the same, or worse with more
history. A task that punishes long input would make any foundation-model
result on it an artifact of the setup, which is the measurement failure this
paper is about.

Abandon criterion, declared before running:

  1. the admission gate must still pass at the longest window, meaning the
     bootstrap CI lower bound clears 1 for both the linear and the nonlinear
     member;
  2. clean skill at the longest window must not sit materially below clean
     skill at 20, taken as the CI at the long window lying entirely below the
     point estimate at 20.

Failing either, the redesign is abandoned and the paper keeps its 20-sample
contract.

Scope of the comparison. Only clean skill is compared across window lengths.
The observation-domain operators act on the whole history, so a bias at
HIST 200 perturbs ten times as many input values as the same bias at HIST 20,
and the plant-side onset sits 5 samples before the forecast origin either
way. Longer windows give a different task instance, not a re-parameterized
one, and severity axes do not carry over.

Uses the stored episodes, so no simulation runs here.

Usage:
    python probe_history.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tep_task as T
from train_tep_panel import MLP, HORIZON, fit

ART = HERE / "artifacts"
OUT = HERE / "history_probe"
HISTS = [20, 60, 120, 200]
STRIDE = 10
N_BOOT = 1000
SEED = 42


def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "run.log", "a") as f:
        f.write(line + "\n")


def windows(eps, hist, horizon, stride, shift, scale):
    """T.build_windows hardcodes T.HIST, so rebuild with an explicit length."""
    X, Y = [], []
    need = hist + horizon
    for _, x in eps:
        z = (x - shift) / scale
        t = T.BURN
        while t + need <= z.shape[0]:
            h = z[t:t + hist]
            fut = z[t + hist:t + hist + horizon, :T.N_MEAS]
            X.append(h.reshape(-1))
            Y.append((fut - h[-1, :T.N_MEAS]).reshape(-1))
            t += stride
    return np.asarray(X, np.float32), np.asarray(Y, np.float32)


def main():
    norm = np.load(ART / "normalization.npz")
    shift, scale = norm["shift"], norm["scale"]
    ep = np.load(ART / "episodes.npz")
    split = {k: [(int(s), ep[f"{k}_{s}"]) for s in v if f"{k}_{s}" in ep.files]
             for k, v in T.SEEDS.items()}

    rows = {}
    for hist in HISTS:
        t0 = time.time()
        Xtr, Ytr = windows(split["train"], hist, HORIZON, STRIDE, shift, scale)
        Xva, Yva = windows(split["val"], hist, HORIZON, STRIDE, shift, scale)
        Xte, Yte = windows(split["eval"], hist, HORIZON, STRIDE * 2, shift, scale)

        def rmse_w(pred):
            return np.sqrt(np.mean(
                (pred - Yte).reshape(len(Yte), -1) ** 2, axis=1))

        r_pers = rmse_w(np.zeros_like(Yte))
        rng = np.random.default_rng(0)
        bidx = rng.integers(0, len(Yte), size=(N_BOOT, len(Yte)))

        def skill_ci(r):
            pt = r_pers.mean() / r.mean()
            bs = [r_pers[i].mean() / r[i].mean() for i in bidx]
            return (pt, float(np.percentile(bs, 2.5)),
                    float(np.percentile(bs, 97.5)))

        lam = 10.0
        A = np.concatenate([Xtr, np.ones((len(Xtr), 1), np.float32)], 1)
        W = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1], dtype=np.float32),
                            A.T @ Ytr)
        Ate = np.concatenate([Xte, np.ones((len(Xte), 1), np.float32)], 1)
        ridge = skill_ci(rmse_w(Ate @ W))

        torch.manual_seed(SEED)
        np.random.seed(SEED)
        m = fit(MLP(Xtr.shape[1], Ytr.shape[1]), Xtr, Ytr, Xva, Yva,
                60, 256, 1e-3, f"mlp_h{hist}")
        m.eval()
        with torch.no_grad():
            pr = np.concatenate([m(torch.from_numpy(Xte[i:i + 512])).numpy()
                                 for i in range(0, len(Xte), 512)])
        mlp = skill_ci(rmse_w(pr))

        rows[hist] = {"n_train": int(len(Xtr)), "n_eval": int(len(Yte)),
                      "dim": int(Xtr.shape[1]),
                      "ridge": ridge, "mlp": mlp}
        log(f"HIST {hist:3d}  windows {len(Xtr):5d}  dim {Xtr.shape[1]:5d}  "
            f"ridge {ridge[0]:.2f} [{ridge[1]:.2f},{ridge[2]:.2f}]  "
            f"mlp {mlp[0]:.2f} [{mlp[1]:.2f},{mlp[2]:.2f}]  "
            f"({time.time()-t0:.0f}s)")

    base = rows[HISTS[0]]
    long = rows[HISTS[-1]]
    gate = long["ridge"][1] > 1.0 and long["mlp"][1] > 1.0
    worse = (long["ridge"][2] < base["ridge"][0] and
             long["mlp"][2] < base["mlp"][0])
    log("")
    log(f"criterion 1, gate passes at HIST {HISTS[-1]}: {gate}")
    log(f"criterion 2, materially worse than HIST {HISTS[0]}: {worse}")
    log("VERDICT: " + ("proceed with the long-window redesign"
                      if gate and not worse else "abandon the redesign"))
    with open(OUT / "history_probe.json", "w") as f:
        json.dump({"rows": {str(k): v for k, v in rows.items()},
                   "gate_passes": bool(gate), "materially_worse": bool(worse),
                   "verdict": "proceed" if gate and not worse else "abandon"},
                  f, indent=2, default=float)


if __name__ == "__main__":
    main()
