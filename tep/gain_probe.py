"""Viability probe for the control-authority sweep.

TEP is open-loop unstable, so weakening the regulatory layer eventually trips
the shutdown interlock. This finds how far the gain can be reduced before the
plant stops producing scorable episodes.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import tep_task as T
from tep_sim import simulate_batch
from shutdown_scan import freeze_onset

SEEDS = list(range(5000, 5016))
LEVELS = [1.0, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]


def main():

    jobs, keys = [], []
    for g in LEVELS:
        for s in SEEDS:
            jobs.append((T.nominal_idata(), s, T.nominal_dsev(), None, g))
            keys.append((g, s))
    outs = simulate_batch(jobs, workers=8)

    print(f"{'gain':>6s} {'shutdown':>9s} {'P sd':>8s} {'T sd':>7s} {'lvl sd':>7s}")
    for g in LEVELS:
        bad = ps = ts = ls = 0
        n = 0
        for (gg, s), x in zip(keys, outs):
            if gg != g:
                continue
            x = np.asarray(x)
            if not np.isfinite(x).all() or (freeze_onset(x[:, :33]) is not None):
                bad += 1
                continue
            ps += x[80:, 6].std(); ts += x[80:, 8].std(); ls += x[80:, 7].std()
            n += 1
        d = max(n, 1)
        print(f"{g:6.2f} {bad:4d}/{len(SEEDS):<4d} {ps/d:8.2f} {ts/d:7.3f} {ls/d:7.3f}")


if __name__ == "__main__":
    main()
