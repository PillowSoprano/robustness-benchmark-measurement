"""Train the MLP and GRU panel members under additional seeds.

Seed 42 already exists in artifacts/. This adds seeds 123 and 7, writing to
artifacts/seed_{s}/. Ridge and persistence are deterministic and shared.

Usage:
    PYTORCH_ENABLE_MPS_FALLBACK=1 python multi_seed_train.py
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
SEEDS = [123, 7]

def main():
    os.chdir(WW_ROOT)
    sys.path.insert(0, WW_ROOT)
    sys.path.insert(0, str(HERE))
    import train_panel as T
    import args_new as new_args
    from experiments.process_disturbances import load_model_and_data
    from utils.disturbance_handler import get_default_disturbance_handler

    args = dict(new_args.args)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        load_model_and_data("mamba", "MBR", args)
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

    Xtr, Ytr, _ = T.build(T.TRAIN_SEQS, states, actions, dist, track,
                          sx, cx, su, cu, T.STRIDE)
    Xva, Yva, _ = T.build(T.VAL_SEQS, states, actions, dist, track,
                          sx, cx, su, cu, T.STRIDE)
    din, dout = Xtr.shape[1], Ytr.shape[1]
    print(f"train {Xtr.shape} val {Xva.shape}", flush=True)

    for seed in SEEDS:
        out = HERE / "artifacts" / f"seed_{seed}"
        out.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(seed)
        np.random.seed(seed)
        t0 = time.time()
        mlp = T.fit(T.MLP(din, dout), Xtr, Ytr, Xva, Yva, 60, 128, 1e-3,
                    f"mlp_s{seed}")
        torch.save(mlp.state_dict(), out / "mlp.pt")
        gru = T.fit(T.Seq2Seq(len(track), 14), Xtr, Ytr, Xva, Yva, 60, 128,
                    2e-3, f"gru_s{seed}")
        torch.save(gru.state_dict(), out / "gru.pt")
        print(f"seed {seed} done in {time.time()-t0:.0f}s", flush=True)
        with open(out / "meta.json", "w") as fh:
            json.dump({"seed": seed}, fh)

if __name__ == "__main__":
    main()
