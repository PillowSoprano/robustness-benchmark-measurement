"""Figure 5: skill retention on TEP, seven operators.

Six disturbance-type operators leave every member above persistence. The
seventh, the regime operator, drops all of them through it. The regime panel
is visually distinguished to make that contrast easy to locate.

Same layering and the same colour semantics as the MBR figure, so a reader
who learned the mapping there keeps it here.
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

R = json.load(open(resolve(HERE / "retention/results_v2.json")))
try:
    R2 = json.load(open(resolve(HERE / "retention/regime_act2_results.json")))
    for m, v in R2["models"].items():
        R["models"][m].update(v)
except FileNotFoundError:
    pass

SEEDS = sorted({int(name.rsplit("_s", 1)[1])
                for name in R["models"] if name.startswith("mlp_s")})
FAM = ["ridge", "mlp", "gru"]
DOM = [("obs_bias", "observation / bias", "severity (std units)"),
       ("obs_noise", "observation / noise", "severity (std units)"),
       ("process_idv1", "process / IDV(1) feed ratio", "dsev"),
       ("actuation_idv6", "actuation / IDV(6) feed loss", "dsev"),
       ("actuation_idv14", "actuation / IDV(14) valve sticking", "dsev"),
       ("dynamics_idv13", "dynamics / IDV(13) kinetics", "dsev"),
       ("regime_setpt18", "regime / reactor T setpoint", "offset ($^\\circ$C)")]
DOM = [d for d in DOM if d[0] in R["models"]["ridge"]]
HERO = "regime_setpt18"
DEG_PER_UNIT = 10.0        # severity s maps to s * 10 degrees C

n = len(DOM)
ncol = (n + 1) // 2
fig, axgrid = plt.subplots(2, ncol, figsize=(fs.TEXT, 4.3), sharey=True)
axes = axgrid.ravel()
for ax in axes[n:]:
    ax.axis("off")

for j, (key, title, xlab) in enumerate(DOM):
    ax = axes[j]
    hero = key == HERO
    ax.set_ylim(0.0, 2.35)
    fs.baseline(ax, 1.0, label="persistence" if j == 0 else None)

    for fam in FAM:
        st = fs.MODEL[fam]
        col = st["color"]
        if fam == "ridge":
            m = R["models"]["ridge"][key]
            xs = np.array(m["levels"], float)
            pt = np.array(m["skill"])
            lo, hi = m["ci_lo"], m["ci_hi"]
        else:
            curves = np.stack([R["models"][f"{fam}_s{sd}"][key]["skill"]
                               for sd in SEEDS])
            m42 = R["models"][f"{fam}_s42"][key]
            xs = np.array(m42["levels"], float)
            pt = curves.mean(0)
            lo, hi = curves.min(0), curves.max(0)
        if hero:
            xs = xs * DEG_PER_UNIT
        ax.fill_between(xs, lo, hi, color=col, alpha=0.16, lw=0, zorder=1)
        ax.plot(xs, pt, color=col, marker=st["marker"], lw=st["lw"], zorder=3)

        ux = fs.family_crossing(R["models"], fam, SEEDS, key)
        if ux is not None:
            ax.axvline(ux * DEG_PER_UNIT if hero else ux, color=col,
                       ls="--", lw=0.9, alpha=0.55, zorder=2)

    ax.set_title(title, pad=3,
                 fontweight="bold" if hero else "normal")
    ax.set_xlabel(xlab)
    ax.grid(alpha=0.22)
    if j % ncol == 0:
        ax.set_ylabel("skill$(s)$ vs persistence")
    if hero:
        # Distinguish the regime panel from the six disturbance panels.
        ax.set_facecolor("#FFF6E5")
        for side in ("top", "right"):
            ax.spines[side].set_visible(True)
        for s in ax.spines.values():
            s.set_color("#C8922A")
            s.set_linewidth(1.0)
        ax.annotate("every member falls\nbelow persistence",
                    xy=(0.97, 0.60), xycoords="axes fraction",
                    ha="right", va="top", fontsize=6.8, color="#8A5A00")

handles = [plt.Line2D([], [], color=fs.MODEL[f]["color"],
                      marker=fs.MODEL[f]["marker"], lw=fs.MODEL[f]["lw"])
           for f in FAM]
axes[0].legend(handles, [fs.LABEL[f] for f in FAM], loc="lower left",
               handlelength=1.6, borderpad=0.2, labelspacing=0.25)

fig.tight_layout(w_pad=1.5, h_pad=1.4)
fs.save(fig, HERE / "retention/tep_retention_v2")
