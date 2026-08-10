"""Where does the regime failure live, and does recurrence track it?

Section 6.2 reports that a 2.5 degree setpoint offset drops every learned
member below persistence while the GRU retains an order of magnitude more
skill than the rest. Two questions this script answers from one cheap run,
with no new training:

  1. Per channel: is the failure concentrated in the temperature-coupled
     channels (the operating point moved) or uniform across all 22 (the
     encoded dynamics are void everywhere)?
  2. Per horizon step: does the GRU's error start where the MLP's does and
     fall as the horizon proceeds (recurrent state tracking the shifted
     trajectory) or is it uniformly lower (a better prior, no tracking)?

Plus the sharpest single number: the mean signed error on the reactor
temperature channel in physical units. A model whose forecast sits near the
OLD setpoint shows a signed bias of about minus the offset; a model tracking
the new operating point shows a bias near zero.

Severity: s = 0.25 (offset 2.5 C), the first grid level past every member's
utility crossing. Nominal arms at s = 0 give the clean baseline.

Usage:
    RBM_SEEDS=42,123,7,2024,31337 python tep_regime_diagnostic.py
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
from shutdown_scan import freeze_onset

ART = HERE / "artifacts"
OUT = HERE / "retention" / "regime_diagnostic.json"

T_ORIGIN = 300
SPAN_END = T_ORIGIN + HORIZON
EVAL_SEEDS = list(range(4000, 4080))
SEEDS = [int(s) for s in
         os.environ.get("RBM_SEEDS", "42,123,7,2024,31337").split(",")]
SEV = 0.25                       # offset = 2.5 C on SETPT(18)
CH_RT = 8                        # XMEAS(9), reactor temperature, 0-based

XMEAS_NAMES = [
    "A feed", "D feed", "E feed", "A+C feed", "recycle flow",
    "reactor feed rate", "reactor pressure", "reactor level",
    "reactor temperature", "purge rate", "sep temperature", "sep level",
    "sep pressure", "sep underflow", "stripper level", "stripper pressure",
    "stripper underflow", "stripper temperature", "steam flow",
    "compressor work", "CW reactor outlet T", "CW condenser outlet T"]


def main():
    t0 = time.time()
    norm = np.load(ART / "normalization.npz")
    shift, scale = norm["shift"], norm["scale"]
    meas_scale = scale[:T.N_MEAS]
    meas_shift = shift[:T.N_MEAS]
    ep = np.load(ART / "episodes.npz")
    nominal = {int(s): ep[f"eval_{s}"][:, :T.N_MEAS + T.N_MV].astype(np.float32)
               for s in EVAL_SEEDS}

    spoff = np.zeros(20)
    spoff[17] = SEV * 10.0
    jobs = [(T.nominal_idata(), sd, T.nominal_dsev(), spoff)
            for sd in EVAL_SEEDS]
    print(f"simulating {len(jobs)} regime arms at s={SEV}", flush=True)
    outs = simulate_batch(jobs, workers=8)
    shifted = {}
    for sd, x in zip(EVAL_SEEDS, outs):
        xx = np.asarray(x)[:, :T.N_MEAS + T.N_MV].astype(np.float32)
        if np.isfinite(xx).all():
            f0 = freeze_onset(xx)
            if f0 is None or f0 >= SPAN_END:
                shifted[sd] = xx
    print(f"  {len(shifted)}/80 arms valid ({time.time()-t0:.0f}s)", flush=True)

    din = T.HIST * (T.N_MEAS + T.N_MV)
    dout = HORIZON * T.N_MEAS
    W = np.load(ART / "ridge_W.npy")
    nets = {}
    for sd in SEEDS:
        m = MLP(din, dout)
        m.load_state_dict(torch.load(ART / f"seed_{sd}" / "mlp.pt"))
        m.eval()
        g = Seq2Seq()
        g.load_state_dict(torch.load(ART / f"seed_{sd}" / "gru.pt"))
        g.eval()
        nets[f"mlp_s{sd}"], nets[f"gru_s{sd}"] = m, g

    def predict(model, F, L):
        """(episodes, HORIZON, N_MEAS) in normalised units."""
        if model == "persistence":
            return np.repeat(L[:, None, :], HORIZON, axis=1)
        if model == "ridge":
            inc = np.concatenate([F, np.ones((len(F), 1), np.float32)], 1) @ W
            return inc.reshape(-1, HORIZON, T.N_MEAS) + L[:, None, :]
        with torch.no_grad():
            inc = nets[model](torch.from_numpy(F)).numpy()
        return inc.reshape(-1, HORIZON, T.N_MEAS) + L[:, None, :]

    def windows(traj):
        F, L, G = [], [], []
        for sd, x in traj.items():
            z = (x - shift) / scale
            h = z[T_ORIGIN - T.HIST:T_ORIGIN]
            F.append(h.reshape(-1))
            L.append(h[-1, :T.N_MEAS])
            G.append(z[T_ORIGIN:T_ORIGIN + HORIZON, :T.N_MEAS])
        return (np.asarray(F, np.float32), np.asarray(L, np.float32),
                np.asarray(G, np.float32))

    def family(model_key):
        return model_key.split("_")[0]

    MEMBERS = ["persistence", "ridge"] + sorted(nets)
    result = {"severity": SEV, "offset_C": SEV * 10.0,
              "episodes_valid": len(shifted), "seeds": SEEDS,
              "channels": XMEAS_NAMES, "per_member": {}}

    for arm_name, traj in (("clean", nominal), ("regime", shifted)):
        F, L, G = windows(traj)
        for m in MEMBERS:
            P = predict(m, F, L)
            E = P - G                                   # normalised error
            d = result["per_member"].setdefault(m, {})
            d[f"{arm_name}_rmse_per_step"] = [
                round(float(np.sqrt(np.mean(E[:, t, :] ** 2))), 4)
                for t in range(HORIZON)]
            d[f"{arm_name}_rmse_per_channel"] = [
                round(float(np.sqrt(np.mean(E[:, :, c] ** 2))), 4)
                for c in range(T.N_MEAS)]
            # signed bias in physical units on the reactor temperature
            phys_err = E[:, :, CH_RT] * meas_scale[CH_RT]
            d[f"{arm_name}_rt_bias_C"] = round(float(np.mean(phys_err)), 4)

    # family aggregates over seeds (mean of per-seed values)
    agg = {}
    for fam in ("mlp", "gru"):
        keys = [k for k in MEMBERS if k.startswith(fam + "_s")]
        agg[fam] = {
            stat: [round(float(np.mean([result["per_member"][k][stat][i]
                                        for k in keys])), 4)
                   for i in range(len(result["per_member"][keys[0]][stat]))]
            for stat in ("regime_rmse_per_step", "regime_rmse_per_channel",
                         "clean_rmse_per_channel")}
        agg[fam]["regime_rt_bias_C"] = round(float(np.mean(
            [result["per_member"][k]["regime_rt_bias_C"] for k in keys])), 4)
    result["family"] = agg

    # concentration: how much of each member's squared regime error sits in
    # the five temperature-coupled channels (9, 11, 18, 21, 22 -> 0-based)
    TEMP = [8, 10, 17, 20, 21]
    conc = {}
    for m in ("persistence", "ridge"):
        pc = result["per_member"][m]["regime_rmse_per_channel"]
        conc[m] = round(sum(pc[c] ** 2 for c in TEMP)
                        / sum(v ** 2 for v in pc), 4)
    for fam in ("mlp", "gru"):
        pc = agg[fam]["regime_rmse_per_channel"]
        conc[fam] = round(sum(pc[c] ** 2 for c in TEMP)
                          / sum(v ** 2 for v in pc), 4)
    result["temp_channel_share_of_sq_error"] = conc
    result["temp_channels"] = [XMEAS_NAMES[c] for c in TEMP]

    OUT.write_text(json.dumps(result, indent=1))

    print(f"\nreactor-temperature signed bias under the shift (physical C):")
    print(f"  persistence {result['per_member']['persistence']['regime_rt_bias_C']:+.2f}"
          f"   ridge {result['per_member']['ridge']['regime_rt_bias_C']:+.2f}"
          f"   mlp {agg['mlp']['regime_rt_bias_C']:+.2f}"
          f"   gru {agg['gru']['regime_rt_bias_C']:+.2f}")
    print(f"temperature-channel share of squared error: {conc}")
    for fam in ("mlp", "gru"):
        ps = agg[fam]["regime_rmse_per_step"]
        print(f"{fam} per-step rmse: t1 {ps[0]:.3f}  t10 {ps[9]:.3f}  "
              f"t20 {ps[19]:.3f}  t40 {ps[39]:.3f}")
    print(f"wrote {OUT}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
