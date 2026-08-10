"""Figure 4: skill retention under intervention, MBR 80-step task.

Layering, lightest to heaviest: bootstrap interval as a pale fill, seed
spread as a slightly stronger fill, and the point curve on top. Filled
intervals keep overlapping uncertainty bands readable.

The persistence line is the figure's spine: above it a model earns its
complexity, below it a model is worse than holding the last observation, and
that region carries a tint so it reads before the caption does.
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

R = json.load(open(resolve(HERE / "skill_retention/raw_results.json")))
SEEDS = R["seeds"]
DOM = [("obs_bias", "observation / bias", "severity (scale units)", 0.7),
       ("obs_noise", "observation / noise", "severity (scale units)", 0.7),
       ("process", "process input", "influent scaling (%)", None),
       ("actuation", "actuation", "authority loss $s$", None)]
FAM = ["ridge", "mlp", "gru"]

fig, axes = plt.subplots(1, 4, figsize=(fs.TEXT, 2.5), sharey=True)

for j, (key, title, xlab, xmax) in enumerate(DOM):
    ax = axes[j]
    ax.set_ylim(0.0, 4.6)
    fs.baseline(ax, 1.0, label="persistence" if j == 0 else None)

    ends = []
    for fam in FAM:
        st = fs.MODEL[fam]
        col = st["color"]
        if fam == "ridge":
            m = R["models"]["ridge"][key]
            xs = np.array(m["levels"], float)
            pt = np.array(m["skill"])
            bands = [(np.array(m["ci_lo"]), np.array(m["ci_hi"]), 0.13)]
        else:
            curves = np.stack([R["models"][f"{fam}_s{sd}"][key]["skill"]
                               for sd in SEEDS])
            m42 = R["models"][f"{fam}_s42"][key]
            xs = np.array(m42["levels"], float)
            pt = curves.mean(0)
            bands = [(np.array(m42["ci_lo"]), np.array(m42["ci_hi"]), 0.10),
                     (curves.min(0), curves.max(0), 0.22)]
        # clip to the displayed range so the curve ends where its label sits
        k = xs <= xmax if xmax else np.ones_like(xs, bool)
        for lo, hi, a in bands:
            ax.fill_between(xs[k], lo[k], hi[k], color=col, alpha=a, lw=0,
                            zorder=1)
        ax.plot(xs[k], pt[k], color=col, marker=st["marker"], lw=st["lw"],
                zorder=3)

        # crossing drawn only when most seeds of the family show one
        ux = fs.family_crossing(R["models"], fam, SEEDS, key)
        if ux is not None:
            ax.axvline(ux, color=col, ls="--", lw=0.9, alpha=0.55, zorder=2)
        ends.append((xs[k][-1], pt[k][-1], fs.LABEL[fam], col))

    ax.set_xlim(-0.02 if xmax else None, xmax)
    ax.set_title(title, pad=3)
    ax.set_xlabel(xlab)
    ax.grid(alpha=0.22)
    if j == 0:
        ax.set_ylabel("skill$(s)$ vs persistence")
    fs.end_labels(ax, ends)

fig.tight_layout(w_pad=1.9)
fs.save(fig, HERE / "skill_retention/skill_retention")
