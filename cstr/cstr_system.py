"""Nonlinear CSTR plant used by the admission study.

State x = [C_A, T], exogenous d = [C_Af, T_f], actuation u = T_c (applied).
Parameters and operating anchors are from the MathWorks CSTR documentation;
discretization and sampling choices are benchmark-design decisions. The
implementation-validation record is `CARD_DEVIATIONS.md`.
"""

import numpy as np
from scipy.integrate import solve_ivp

# --- source-backed parameters
F = 1.0            # m3/h
V = 1.0            # m3
R = 1.985875       # kcal/(kmol K)
DH = -5960.0       # kcal/kmol
E = 11843.0        # kcal/kmol
K0 = 34930800.0    # 1/h
RHO_CP = 500.0     # kcal/(m3 K)
UA_NOM = 150.0     # kcal/(K h)

# --- operating anchors
ANCHOR_L = dict(CA=8.5698, T=311.2639, CAf=10.0, Tf=300.0, Tc=292.0)
ANCHOR_H = dict(CA=2.0, T=373.0, CAf=10.0, Tf=300.0, Tc=299.0)

# --- time base
DT = 0.05          # h, recorded interval
HIST = 20          # observation history samples
HORIZON = 10       # forecast horizon samples
PREFIX = 5         # intervention onset before forecast origin
BURN_IN_H = 4.0

# --- envelope and reference scales, revised after implementation validation
# The initial protocol took T in [310, 390] from the source's reported modeled
# range, but that interval is inconsistent with the L anchor (T = 311.26) plus
# the nominal sampling spread: 14.5% of unperturbed episodes violate it
# (measured T minimum 308.34 over 400 episodes). An admissibility event that
# fires on nominal operation cannot serve as a failure signal, so the lower
# bound is moved to 305 K, which is below the measured nominal support with
# margin and yields 0/300 nominal violations. The upper bound and the C_A and
# T_c bounds are unchanged. Reference scales r_j are unchanged so the fidelity
# loss stays comparable to the original definition.
CA_RANGE = (0.0, 10.0)
T_RANGE = (305.0, 390.0)
T_RANGE_CARD_V01 = (310.0, 390.0)   # retained for the validation record
TC_RANGE = (273.0, 322.0)
R_CA, R_T = 10.0, 80.0


def rhs(t, x, CAf, Tf, Tc, UA):
    CA, T = x
    T = max(T, 1e-6)
    r = K0 * np.exp(-E / (R * T)) * CA
    dCA = F / V * (CAf - CA) - r
    dT = (F / V * (Tf - T)
          - DH / RHO_CP * r
          - UA / (RHO_CP * V) * (T - Tc))
    return [dCA, dT]


def step(x, CAf, Tf, Tc, UA=UA_NOM, dt=DT, rtol=1e-9, atol=1e-11):
    """Advance one recorded interval with an adaptive solver."""
    sol = solve_ivp(rhs, (0.0, dt), np.asarray(x, dtype=float),
                    args=(CAf, Tf, Tc, UA), method="RK45",
                    rtol=rtol, atol=atol, dense_output=False)
    if not sol.success or not np.all(np.isfinite(sol.y[:, -1])):
        return np.array([np.nan, np.nan]), False
    return sol.y[:, -1], True


def rollout(x0, CAf_seq, Tf_seq, Tc_seq, UA=UA_NOM):
    """Simulate len(Tc_seq) intervals. Returns (states[T+1,2], ok_flags)."""
    n = len(Tc_seq)
    xs = np.zeros((n + 1, 2))
    xs[0] = x0
    ok = True
    for k in range(n):
        xn, good = step(xs[k], CAf_seq[k], Tf_seq[k], Tc_seq[k], UA)
        if not good:
            xs[k + 1:] = np.nan
            ok = False
            break
        xs[k + 1] = xn
    return xs, ok


def admissible(states, Tc_applied=None):
    """Card 6 hard trajectory events. Returns (bool, first_violation_index)."""
    st = np.asarray(states, dtype=float)
    bad = ~np.isfinite(st).all(axis=1)
    bad |= (st[:, 0] < CA_RANGE[0]) | (st[:, 0] > CA_RANGE[1])
    bad |= (st[:, 1] < T_RANGE[0]) | (st[:, 1] > T_RANGE[1])
    if Tc_applied is not None:
        tc = np.asarray(Tc_applied, dtype=float)
        tcbad = (tc < TC_RANGE[0]) | (tc > TC_RANGE[1])
        if tcbad.any():
            return False, int(np.argmax(tcbad))
    if bad.any():
        return False, int(np.argmax(bad))
    return True, -1


def j_fid(pred, truth):
    """Card 6 range-normalized RMS over [C_A, T]."""
    p, t = np.asarray(pred, dtype=float), np.asarray(truth, dtype=float)
    e = np.stack([(p[:, 0] - t[:, 0]) / R_CA, (p[:, 1] - t[:, 1]) / R_T], axis=1)
    return float(np.sqrt(np.mean(e ** 2)))


def regime_center(s):
    """Card 7.2 regime path for [C_A, T, T_c]."""
    lo = np.array([ANCHOR_L["CA"], ANCHOR_L["T"], ANCHOR_L["Tc"]])
    hi = np.array([ANCHOR_H["CA"], ANCHOR_H["T"], ANCHOR_H["Tc"]])
    return (1 - s) * lo + s * hi


def sample_episode(rng, n_samples, regime_s=0.0, hold_h=0.25):
    """Card 7.1 nominal distributions, translated along the regime path."""
    c = regime_center(regime_s)
    hold = max(1, int(round(hold_h / DT)))
    n_holds = int(np.ceil(n_samples / hold))

    def pw(mean, sd, lo=None, hi=None):
        v = rng.normal(mean, sd, size=n_holds)
        if lo is not None:
            v = np.clip(v, lo, hi)
        return np.repeat(v, hold)[:n_samples]

    CAf = pw(10.0, 0.2)
    Tf = pw(300.0, 0.5)
    Tc = pw(c[2], 1.0, c[2] - 5.0, c[2] + 5.0)
    x0 = np.array([rng.normal(c[0], 0.15), rng.normal(c[1], 1.0)])
    return x0, CAf, Tf, Tc
