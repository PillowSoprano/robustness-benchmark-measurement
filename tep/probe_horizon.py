"""Horizon probe: persistence vs ridge skill before committing the task.

Lesson from the CSTR failure: verify the task is not persistence-dominated
BEFORE building the full pipeline.
"""

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tep_task as T

def main():
    t0 = time.time()
    train = T.gen_episodes(range(1000, 1060))       # 60 episodes for the probe
    val = T.gen_episodes(range(2000, 2020))
    print(f"episodes: {len(train)} train, {len(val)} val "
          f"({time.time()-t0:.0f}s)", flush=True)
    shift, scale = T.fit_normalization(train)

    print(f"{'horizon':>8}{'hours':>7} | {'persist':>9}{'ridge':>9}{'skill':>7}")
    for H in [10, 20, 40, 80]:
        Xtr, Ytr = T.build_windows(train, H, 10, shift, scale)
        Xva, Yva = T.build_windows(val, H, 20, shift, scale)
        pers = float(np.sqrt(np.mean(Yva ** 2)))
        lam = 10.0
        A = np.concatenate([Xtr, np.ones((len(Xtr), 1), np.float32)], 1)
        W = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1], dtype=np.float32),
                            A.T @ Ytr)
        Av = np.concatenate([Xva, np.ones((len(Xva), 1), np.float32)], 1)
        ridge = float(np.sqrt(np.mean((Av @ W - Yva) ** 2)))
        print(f"{H:>8}{H*0.05:>7.1f} | {pers:>9.4f}{ridge:>9.4f}"
              f"{pers/ridge:>7.2f}", flush=True)

if __name__ == "__main__":
    main()
