"""The paper's load-bearing numbers, checked against the committed artifacts.

Each expectation below is a number the paper states. The value on the left is
what the paper says; the value on the right is read out of
`frozen_predictions/`. A failure means the two have drifted, which is the
only way a reader can tell that a regenerated artifact no longer supports the
sentence it was cited for.

This is deliberately not a test of the analysis code. It is a test that the
committed artifacts still say what the paper claims they say.

Usage:
    python tests/test_numbers.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "frozen_predictions"

checks = []


def expect(label, claimed, actual, tol=5e-3):
    if isinstance(claimed, (str, bool, int, list, set, tuple)):
        ok = claimed == actual
    else:
        ok = abs(float(claimed) - float(actual)) <= tol
    checks.append((label, ok, f"paper {claimed}  artifact {actual}"))


def load(rel):
    p = FROZEN / rel
    if not p.exists():
        checks.append((f"artifact {rel}", False, "missing"))
        return None
    return json.loads(p.read_text())


# ---------------------------------------------------------------- Section 4.4
d = load("cstr/horizon_excitation_sweep.json")
if d:
    hs, es = d["horizon_sweep"], d["excitation_sweep"]
    expect("CSTR skill at the card horizon", 1.13, hs["10"]["panel_best_skill"])
    expect("CSTR skill at 6 h", 1.47, hs["120"]["panel_best_skill"])
    exc = [es[e]["panel_best_skill"] for e in ("1.0", "2.0", "4.0", "8.0")]
    expect("CSTR excitation floor", 1.12, min(exc))
    expect("CSTR excitation ceiling", 1.13, max(exc))
    expect("CSTR configurations tried", 10, len(hs) - 1 + len(es))
    spreads = [max(r["ridge_skill"], r["mlp_skill"], r["gru_skill"])
               - min(r["ridge_skill"], r["mlp_skill"], r["gru_skill"])
               for r in list(hs.values()) + list(es.values())
               if isinstance(r, dict) and "ridge_skill" in r]
    expect("CSTR widest panel spread", 0.052, max(spreads), tol=5e-4)
    expect("CSTR panel is the paper's panel",
           ["persistence", "ridge", "mlp", "gru"],
           d["declared_before_running"]["panel"])

# ------------------------------------------------------------ Sections 6, 9
d = load("tep/seed_spread.json")
if d:
    pd = d["per_domain_minimum_skill"]
    mod = d.get("modern_per_domain_minimum_skill", {})
    expect("training seeds", 5, len(d["method"]["seeds"]))
    expect("per-seed curves cover all four stochastic members",
           {"mlp", "gru", "dlinear", "patchtf"},
           set(d["coverage"]["per_seed_curves"]))
    widest = max([d["summary"]["max_seed_spread_over_all_domain_minima"]]
                 + [v["spread"] for r in mod.values() for v in r.values()])
    expect("widest seed spread", 0.773, widest, tol=5e-4)
    expect("GRU observation-bias spread", 0.286,
           pd["obs_bias"]["gru"]["spread"], tol=5e-4)
    expect("GRU regime spread", 0.119, pd["regime_setpt18"]["gru"]["spread"],
           tol=5e-4)
    fl = d["ordering_flips_across_seeds"]
    expect("ordering flips", 27, fl["n_cells_flipped"])
    expect("cells checked", 64, fl["n_cells_checked"])

    def vals(row):
        return [v for m in row.values()
                for v in (m.get("per_seed") or {"": m["value"]}).values()]

    # Section 9.3 rests on these two, and they are now checked over the whole
    # grid rather than over the three members that used to have per-seed data
    PLANT = ("actuation_idv6", "actuation_idv14", "dynamics_idv13",
             "process_idv1")
    plant_min = min(v for dom in PLANT
                    for v in vals(pd[dom]) + vals(mod.get(dom, {})))
    expect("no plant-side crossing at any seed", True, plant_min >= 1.0)
    reg_max = max(vals(pd["regime_setpt18"])
                  + vals(mod.get("regime_setpt18", {})))
    expect("every member crosses in the regime domain at every seed",
           True, reg_max < 1.0)
    # the exception five seeds found and three did not
    expect("DecompLinear crosses under observation bias at one seed", 0.713,
           min(mod["obs_bias"]["dlinear"]["per_seed"].values()), tol=5e-4)

# ------------------------------------------------------------- Section 10.2
d = load("tep/history_probe.json")
if d:
    rows = d["rows"]
    for h, ridge, mlp in (("20", 1.94, 1.88), ("60", 1.82, 1.86),
                          ("120", 1.55, 1.80), ("200", 1.04, 1.67)):
        expect(f"history {h} ridge", ridge, rows[h]["ridge"][0])
        expect(f"history {h} MLP", mlp, rows[h]["mlp"][0])
    expect("long-window verdict", "abandon", d["verdict"])

# --------------------------------------------------------------- Section 8
d = load("tep/gain_sweep.json")
if d:
    expect("gain sweep artifact present", True, True)

# --------------------------------------------------------------- Section 9
d = load("tep/modern_panel.json")
if d:
    reg = d["curves"]["regime_setpt18"]
    for m, v in (("ridge", 0.02), ("mlp", 0.02), ("gru", 0.62),
                 ("dlinear", 0.02), ("patchtf", 0.07)):
        expect(f"regime minimum {m}", v, min(reg[m]), tol=6e-3)

# ------------------------------------------------------------------- report
width = max(len(c[0]) for c in checks)
failed = sum(not ok for _, ok, _ in checks)
print("=== paper numbers against committed artifacts ===\n")
for label, ok, detail in checks:
    print(f"  {'PASS' if ok else 'FAIL'}  {label:<{width}}  {detail}")
print(f"\n{len(checks) - failed}/{len(checks)} pass")
sys.exit(1 if failed else 0)
