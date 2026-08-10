"""Figure 3: the ranking inversion, stated as plainly as it can be.

Left, the relative gate's failure radius: longer is "more robust", and the
grey baseline wins. Right, the absolute error at matched severity: shorter is
better, and the ordering reverses. Horizontal bars so the value labels sit at
the bar ends with room to breathe.
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import figstyle as fs
from artifacts import resolve

fs.apply()
import matplotlib.pyplot as plt

R = json.load(open(resolve(HERE / "fingerprint/raw_results.json")))
M = ["persistence", "ridge", "mlp", "gru"]
SKILL = {"persistence": "1.00", "ridge": "1.43", "mlp": "1.59", "gru": "3.80"}
KEYS = ["obs_bias", "obs_noise", "process", "actuation"]
ROWLAB = ["obs / bias", "obs / noise", "process", "actuation"]
NORM = {"obs_bias": 1.0, "obs_noise": 1.0, "process": 100.0, "actuation": 1.0}
SEV = {"obs_bias": "0.12", "obs_noise": "0.12", "process": "50",
       "actuation": "0.2"}
SEVLAB = ["at 0.12", "at 0.12", "at +50%", "at 0.2"]

fig, (axl, axr) = plt.subplots(1, 2, figsize=(fs.TEXT, 2.35))
h = 0.19
base = np.arange(len(KEYS))[::-1]          # top row first

for i, m in enumerate(M):
    off = (1.5 - i) * h
    rad = [(R["models"][m]["domains"][k]["radius"] or np.nan) / NORM[k]
           for k in KEYS]
    err = [R["models"][m]["domains"][k]["curve"][SEV[k]]["mean_rmse"]
           for k in KEYS]
    col = fs.MODEL[m]["color"]
    bl = axl.barh(base + off, rad, h, color=col, linewidth=0,
                  label=f"{fs.LABEL[m]} (skill {SKILL[m]})")
    br = axr.barh(base + off, err, h, color=col, linewidth=0)
    for b, v in zip(bl, rad):
        if np.isfinite(v):
            axl.text(v + 0.008, b.get_y() + b.get_height() / 2, f"{v:.3f}",
                     va="center", fontsize=6, color="0.25")
    for b, v in zip(br, err):
        axr.text(v + 2.5, b.get_y() + b.get_height() / 2, f"{v:.0f}",
                 va="center", fontsize=6, color="0.25")

for ax, lab in ((axl, ROWLAB), (axr, [f"{a}\n{b}" for a, b in
                                      zip(ROWLAB, SEVLAB)])):
    ax.set_yticks(base)
    ax.set_yticklabels(lab)
    ax.grid(axis="x", alpha=0.22)
    ax.set_axisbelow(True)

axl.set_xlim(0, 0.46)
axl.set_xlabel("failure radius under the RELATIVE gate (normalized)")
axl.set_title("The least accurate member survives the most severity",
              pad=4)
axr.set_xlim(0, 205)
axr.set_xlabel("mean RMSE under intervention (ABSOLUTE error)")
axr.set_title("At matched severity the ordering reverses", pad=4)
# upper right: the two observation rows have short bars, so that corner is
# empty and the legend covers no value label
axl.legend(loc="upper right", handlelength=1.2, borderpad=0.2,
           labelspacing=0.25)

fig.tight_layout(w_pad=1.6)
fs.save(fig, HERE / "fingerprint/gate_ranks")
