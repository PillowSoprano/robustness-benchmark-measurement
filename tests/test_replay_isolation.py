"""Regression check: batch context must not change a seeded TEP trajectory.

Requires the compiled TEP extension in tep/build or on PYTHONPATH.
Run: python tests/test_replay_isolation.py
"""
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tep'))
from tep_sim import simulate,simulate_batch

def main():
    switches=np.zeros((480,20),dtype=int)
    switches[:,7]=1
    switches[295:,13]=1
    severity=np.ones(20);severity[13]=0.5
    independent=simulate(switches,4001,severity)
    # With one worker the former default chunking runs three simulations in
    # each process, exposing the retained valve-sticking state.
    batch=simulate_batch([(switches,4000+i%2,severity) for i in range(12)],workers=1)
    for result in batch[1::2]:
        np.testing.assert_array_equal(result,independent)
    print('PASS: all six repeated sensitive seeds match independent replay')

if __name__=='__main__':main()
