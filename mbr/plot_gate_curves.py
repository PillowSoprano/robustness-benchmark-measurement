"""Figure 2: the same predictions seen through two gates.

Top row is the relative gate, failure probability against its own nominal
error. Bottom row is the absolute error under the same intervention. The two
rows share an x axis per column and sit flush against each other, because the
point is that one set of predictions produces both.

Drawn at final print width, so the point sizes here are the point sizes on
paper.
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import figstyle as fs
from artifacts import resolve

fs.apply()
import matplotlib.pyplot as plt

R = json.load(open(resolve(HERE / "fingerprint/raw_results.json")))
M = ["persistence", "ridge", "mlp", "gru"]
SKILL = {"persistence": "1.00", "ridge": "1.43", "mlp": "1.59", "gru": "3.80"}
DOM = [("obs_bias", "observation / bias", "severity (scale units)"),
       ("obs_noise", "observation / noise", "severity (scale units)"),
       ("process", "process input", "influent scaling (%)"),
       ("actuation", "actuation", "authority loss $s$")]

fig, axes = plt.subplots(2, 4, figsize=(fs.TEXT, 3.5), sharex="col")

for j, (key, title, xlab) in enumerate(DOM):
    axp, axa = axes[0, j], axes[1, j]
    for m in M:
        c = R["models"][m]["domains"][key]["curve"]
        items = sorted(((float(k), v) for k, v in c.items()), key=lambda p: p[0])
        xs = [k for k, _ in items]
        st = dict(fs.MODEL[m])
        st.pop("zorder", None)
        axp.plot(xs, [v["p_fail"] for _, v in items], **st)
        axa.plot(xs, [v["mean_rmse"] for _, v in items], **st)
    axp.axhline(0.10, color="0.35", ls=":", lw=0.8)
    axp.set_title(title, pad=3)
    axp.set_ylim(-0.04, 1.06)
    axa.set_xlabel(xlab)
    for ax in (axp, axa):
        ax.grid(alpha=0.22)
    if j == 0:
        axp.set_ylabel("P(degradation > 10%)\nRELATIVE gate")
        axa.set_ylabel("mean RMSE\nABSOLUTE error")
        axp.text(0.99, 0.12, r"$\alpha=0.10$",
                 transform=axp.get_yaxis_transform(), ha="right",
                 va="bottom", fontsize=6.5, color="0.35")
    else:
        axp.tick_params(labelleft=True)

# One legend for the figure, laid out horizontally above it. Inside a panel
# it sat on the tick labels, and eight panels have no spare corner.
handles = [plt.Line2D([], [], **{k: v for k, v in fs.MODEL[m].items()
                                 if k != "zorder"}) for m in M]
fig.legend(handles, [f"{fs.LABEL[m]} (skill {SKILL[m]})" for m in M],
           loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.045),
           handlelength=1.8, columnspacing=1.8)

# breathing room between columns, none between the two rows: the rows
# describe the same predictions and should read as one block
fig.tight_layout(w_pad=1.7, h_pad=0.0)
fig.subplots_adjust(hspace=0.0)

fs.save(fig, HERE / "fingerprint/gate_curves")
