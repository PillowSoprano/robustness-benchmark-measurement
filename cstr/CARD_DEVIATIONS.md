# CSTR implementation-validation record

Date: 2026-08-07
Scope: assumptions checked before predictor comparison

Two pre-implementation assumptions did not survive numerical validation. Both
are recorded here, together with their consequences and corrections, to retain
a complete audit trail.

## D1 — the high-conversion anchor is not open-loop unstable

The initial specification described anchor H as an open-loop-unstable,
high-conversion point and used that description to motivate the regime
endpoint at `s = 1`.

Measured Jacobian eigenvalues under the implemented parameter set:

| Point | Eigenvalues | Verdict |
|---|---|---|
| L (8.5698, 311.2639), Tc = 292 | -0.895, -0.522 | stable node |
| H (2.0, 373), Tc = 299 | -1.107 +/- 1.087i | stable spiral |

The true equilibrium at Tc = 299 nearest H is (2.0017, 373.1075), also a stable
spiral. A 0.1 K perturbation from H does not diverge over 10 h.

Cause: provenance mixing. The parameter values and the instability claim came
from two MathWorks pages describing differently parameterized CSTR models and
were initially treated as one source set.

Consequences:
- The instability motivation for the regime endpoint is void. The regime
  operator remains valid as a distribution shift outside training support
  (measured: nominal T support is [308.3, 313.6]; at `s = 1` the episode T
  support is [365.4, 379.9], entirely disjoint), but no claim about
  open-loop instability may be made.
- Any planned analysis that depended on an unstable operating point, in
  particular a stability-related decision gate, needs a different construction
  or a different parameter set.
- The initial claim that the same parameterization contains both a stable and
  an unstable operating point is withdrawn.

## D2 — the admissibility envelope fires on nominal operation

The initial specification used T in [310, 390] K as a benchmark admissibility
bound, based on the source's reported modeled range.

Measured on unperturbed episodes drawn from the declared nominal distribution:
29 of 200 episodes violate the envelope, and the minimum
temperature reached over 400 episodes is 308.34 K. The L anchor sits 1.26 K
above the bound while T(0) has standard deviation 1.0 K and the coolant command
varies by +/- 5 K, so the bound is inside the nominal operating support.

An admissibility event that fires 14.5% of the time with no intervention
applied cannot function as a failure signal, and would contaminate every
radius estimate through the inadmissible-trajectory channel.

Correction applied in `cstr_system.py`: lower T bound moved from 310 K to
305 K, below the measured nominal support with margin. Verified 0 of 300
nominal episodes violate. Upper T bound, C_A bounds, T_c bounds, and the
reference scales r_j are unchanged, so the fidelity loss remains as specified.

The original bound is retained in code as `T_RANGE_CARD_V01` so the deviation
is inspectable.

## Role of implementation validation

Both defects were found before any predictor was trained or any severity was
swept, which is the purpose of the calibration stage. D2 in particular would have
produced a spurious inadmissibility channel in every CSTR result. Neither is
visible from specification review alone; both required numerical evaluation.
