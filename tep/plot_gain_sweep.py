"""Read the control-authority sweep and say whether the mechanism claim holds.

The claim: TEP's disturbance domains withhold utility crossings because the
regulatory layer absorbs those interventions. If that is right, the minimum
skill in each disturbance domain should fall as the controller gains are
scaled down, with everything else about the plant, task, and panel fixed.

Prints the test first, in the form that decides it: minimum skill against
gain, per domain, with the sign of the trend. Writes the figure second.
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

R = json.load(open(resolve(HERE / "gain_sweep/gain_sweep.json")))
GAINS = sorted((float(g) for g in R), reverse=True)
MODELS = ["ridge", "mlp", "gru"]
NICE = {"obs_bias": "observation / bias", "obs_noise": "observation / noise",
        "process_idv1": "process / IDV(1)", "actuation_idv6": "actuation / IDV(6)",
        "dynamics_idv13": "dynamics / IDV(13)"}
DOMS = [d for d in NICE if d in R[str(GAINS[0])]["curves"]]

# ---- the test, printed before anything is drawn.
# Compare on the severity levels present at EVERY gain. Censoring drops the
# strongest level at low gain, and taking each gain's own minimum would then
# compare a mild severity against a harsh one.
COMMON = {d: sorted(set.intersection(*[
              set(R[str(g)]["curves"][d]["levels"]) for g in GAINS]) - {0.0})
          for d in DOMS}
mins = {d: {m: [] for m in MODELS} for d in DOMS}
for g in GAINS:
    for d in DOMS:
        c = R[str(g)]["curves"][d]
        idx = [c["levels"].index(v) for v in COMMON[d]]
        for m in MODELS:
            mins[d][m].append(min(c[m][i] for i in idx))

print("minimum skill on the severity levels common to every gain")
print(f"{'domain':22s} " + " ".join(f"g={g:<5.2f}" for g in GAINS)
      + "  max sev  trend")
crossed = []
for d in DOMS:
    worst = [min(mins[d][m][i] for m in MODELS) for i in range(len(GAINS))]
    # gains descend, so a negative slope in this order means detuning lowers skill
    slope = np.polyfit(GAINS, worst, 1)[0]
    # GAINS descend, so slope > 0 means skill drops as the gain drops
    tag = "falls with detuning" if slope > 0 else "rises with detuning"
    span = max(worst) - min(worst)
    hit = [g for g, w in zip(GAINS, worst) if w < 1]
    crossed += [(d, g) for g in hit]
    print(f"{d:22s} " + " ".join(f"{w:7.2f}" for w in worst)
          + f" {max(COMMON[d]):8.2f}  {tag}"
          + (f", crosses at g={min(hit):.2f}" if hit else ""))

n_eval = {g: R[str(g)]["n_eval"] for g in GAINS}
print("\ncensored levels (dropped below 50% surviving episodes):")
for d in DOMS:
    for g in GAINS:
        miss = set(R[str(GAINS[0])]["curves"][d]["levels"]) - \
               set(R[str(g)]["curves"][d]["levels"])
        if miss:
            print(f"  {d} at g={g}: {sorted(miss)}")
print(f"\nscorable episodes per gain: "
      + ", ".join(f"{g}: {n_eval[g]}/80" for g in GAINS))
print("crossings found" if crossed else
      "no crossing inside the viable gain range; the trend is the result")

# ---- figure
fig, axes = plt.subplots(1, len(DOMS), figsize=(fs.TEXT, 2.4), sharey=True)
axes = np.atleast_1d(axes)
for j, d in enumerate(DOMS):
    ax = axes[j]
    ax.set_ylim(0.0, 2.4)
    fs.baseline(ax, 1.0, label="persistence" if j == 0 else None)
    ends = []
    for m in MODELS:
        st = fs.MODEL[m]
        ax.plot(GAINS, mins[d][m], color=st["color"], marker=st["marker"],
                lw=st["lw"], zorder=3)
        ends.append((GAINS[-1], mins[d][m][-1], fs.LABEL[m], st["color"]))
    ax.invert_xaxis()          # weaker control to the right
    ax.set_title(NICE[d], pad=3)
    ax.set_xlabel("controller gain multiplier")
    ax.grid(alpha=0.22)
    if j == 0:
        ax.set_ylabel("minimum skill$(s)$ in domain")
    fs.end_labels(ax, ends)

fs.save(fig, HERE / "gain_sweep/gain_sweep")
