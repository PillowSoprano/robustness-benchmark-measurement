"""Per-seed spread of the TEP retention statistics.

Section 9.2 of the journal manuscript reports minimum skill per domain
averaged over five training seeds, and Section 3.7 requires that a derived
statistic be claimed only when it survives the seed. This script computes the
spread that requirement is checked against, and writes it as an artifact for
Appendix E.

Skill is computed per seed from the per-episode error tables, not by
averaging skills that were already averaged: for each (domain, severity) the
persistence and member errors are aggregated over the surviving episodes
first, and the ratio is taken second. Averaging ratios and taking the ratio
of averages are different numbers, and the second is what the protocol
defines.

Coverage is uneven and the artifact says so. Ridge is a closed-form fit and
carries no training seed. Persistence has no parameters. The MLP and the GRU
have per-episode tables at all five seeds. The two modern architectures have
per-seed clean gate values but no per-seed curve tables, so their spread is
reported at severity zero only.

Usage:
    python seed_spread.py
"""

import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RET = HERE / "retention"
OUT = HERE / "retention" / "seed_spread.json"

TABLES = ["tables_v2.npz", "tables_ext.npz", "tables_regime_act2.npz"]
SEEDS = os.environ.get("RBM_SEEDS", "42,123,7,2024,31337").split(",")
SEEDED = ["mlp", "gru"]           # members whose tables carry a seed suffix
UNSEEDED = ["ridge"]              # deterministic given the split
SPACE = "norm"                    # the declared state-dimension aggregation

# Severity levels censored by the viability rule of Section 3.5. They enter no
# aggregate, so a minimum taken over them would be a minimum over a dead plant.
CENSORED = {("process_idv1", 3.0), ("regime_setpt18", 1.5),
            ("regime_setpt18", 2.0), ("regime_setpt18", 2.5)}


def load_tables():
    """key -> per-episode error array, merged across the three table files."""
    cells = {}
    for name in TABLES:
        path = RET / name
        if not path.exists():
            continue
        z = np.load(path, allow_pickle=True)
        for k in z.files:
            space, member, domain, level = k.split("|")
            if space != SPACE:
                continue
            cells[(member, domain, float(level))] = np.asarray(z[k], float)
    return cells


def pooled(x):
    """Pooled RMS over episodes, the declared episode estimator."""
    return float(np.sqrt(np.mean(np.square(x))))


def mean_of_window(x):
    """The alternative estimator of Section 3.6, reported alongside."""
    return float(np.mean(x))


ESTIMATORS = {"pooled_rms": pooled, "mean_of_window": mean_of_window}


def main():
    cells = load_tables()
    domains = sorted({d for (_, d, _) in cells})

    # member name as it appears in the tables, per seed
    def key(member, seed):
        return f"{member}_s{seed}" if member in SEEDED else member

    def build(agg):
        c = defaultdict(dict)       # (member, seed) -> domain -> [(s, skill)]
        for domain in domains:
            levels = sorted({s for (m, d, s) in cells if d == domain})
            for member in SEEDED + UNSEEDED:
                for seed in (SEEDS if member in SEEDED else ["-"]):
                    pts = []
                    for s in levels:
                        if (domain, s) in CENSORED:
                            continue
                        ref = cells.get(("persistence", domain, s))
                        own = cells.get((key(member, seed), domain, s))
                        if ref is None or own is None:
                            continue
                        pts.append((s, agg(ref) / agg(own)))
                    if pts:
                        c[(member, seed)][domain] = pts
        return c

    all_curves = {name: build(agg) for name, agg in ESTIMATORS.items()}
    curves = all_curves["pooled_rms"]

    result = {
        "method": {
            "space": SPACE,
            "episode_estimator": "pooled RMS over surviving episodes",
            "ratio_order": "aggregate errors first, divide second",
            "seeds": SEEDS,
            "censored_levels_excluded": sorted(
                f"{d}@{s}" for d, s in CENSORED),
            "tables": [t for t in TABLES if (RET / t).exists()],
        },
        "per_domain_minimum_skill": {},
        "clean_skill_at_zero_severity": {},
        "coverage": {
            "per_seed_curves": SEEDED,
            "no_training_seed": UNSEEDED + ["persistence"],
            "per_seed_gate_only": ["dlinear", "patchtf"],
        },
    }

    # ---- minimum skill per domain, per seed
    for domain in domains:
        row = {}
        for member in SEEDED:
            vals = []
            for seed in SEEDS:
                pts = curves[(member, seed)].get(domain)
                if pts:
                    vals.append(min(v for _, v in pts))
            if len(vals) == len(SEEDS):
                row[member] = {
                    "per_seed": {s: round(v, 4) for s, v in zip(SEEDS, vals)},
                    "mean": round(float(np.mean(vals)), 4),
                    "spread": round(float(max(vals) - min(vals)), 4),
                }
        for member in UNSEEDED:
            pts = curves[(member, "-")].get(domain)
            if pts:
                row[member] = {"value": round(min(v for _, v in pts), 4),
                               "spread": 0.0,
                               "note": "no training seed"}
        result["per_domain_minimum_skill"][domain] = row

    # ---- clean skill, and the modern members' per-seed gate
    for member in SEEDED:
        vals = []
        for seed in SEEDS:
            for domain, pts in curves[(member, seed)].items():
                z = [v for s, v in pts if s == 0.0]
                if z:
                    vals.append(z[0])
                    break
        if len(vals) == len(SEEDS):
            result["clean_skill_at_zero_severity"][member] = {
                "per_seed": {s: round(v, 4) for s, v in zip(SEEDS, vals)},
                "spread": round(float(max(vals) - min(vals)), 4),
            }

    mp = HERE / "modern" / "modern_panel.json"
    modern = {}
    if mp.exists():
        md = json.loads(mp.read_text())
        gate = md["gate"]
        for member in ("dlinear", "patchtf"):
            vals = [gate[f"{member}_s{s}"]["skill"] for s in SEEDS
                    if f"{member}_s{s}" in gate]
            if vals:
                result["clean_skill_at_zero_severity"][member] = {
                    "per_seed": {s: round(v, 4) for s, v in zip(SEEDS, vals)},
                    "spread": round(float(max(vals) - min(vals)), 4),
                }
        # per-seed curves, present since the modern panel stopped averaging
        # the errors before forming the ratio
        for domain, c in md["curves"].items():
            levels = c["levels"]
            row = {}
            for member in ("dlinear", "patchtf"):
                keys = [f"{member}_s{s}" for s in SEEDS]
                if not all(k in c for k in keys):
                    continue
                mins = [min(v for lv, v in zip(levels, c[k])
                            if (domain, lv) not in CENSORED) for k in keys]
                row[member] = {
                    "per_seed": {s: round(v, 4) for s, v in zip(SEEDS, mins)},
                    "mean": round(float(np.mean(mins)), 4),
                    "spread": round(float(max(mins) - min(mins)), 4),
                }
            if row:
                modern[domain] = row
        result["modern_per_domain_minimum_skill"] = modern
        result["coverage"]["per_seed_curves"] = SEEDED + ["dlinear", "patchtf"]
        result["coverage"].pop("per_seed_gate_only", None)

    # ---- does the member ordering flip between seeds?
    flips = []
    for domain in domains:
        levels = sorted({s for (m, d, s) in cells if d == domain})
        for s in levels:
            if (domain, s) in CENSORED:
                continue
            orders = {}
            for seed in SEEDS:
                vals = {}
                for member in SEEDED + UNSEEDED:
                    sd = seed if member in SEEDED else "-"
                    pts = curves[(member, sd)].get(domain, [])
                    hit = [v for lv, v in pts if lv == s]
                    if hit:
                        vals[member] = hit[0]
                if len(vals) == len(SEEDED) + len(UNSEEDED):
                    orders[seed] = tuple(sorted(vals, key=vals.get,
                                                reverse=True))
            if len(set(orders.values())) > 1:
                flips.append({"domain": domain, "severity": s,
                              "orders": {k: list(v)
                                         for k, v in orders.items()}})
    result["ordering_flips_across_seeds"] = {
        "n_cells_checked": sum(
            1 for d in domains
            for s in sorted({x for (m, dd, x) in cells if dd == d})
            if (d, s) not in CENSORED),
        "n_cells_flipped": len(flips),
        "cells": flips,
    }

    # ---- the same minima under the alternative episode estimator
    alt = {}
    for domain in domains:
        row = {}
        for member in SEEDED:
            vals = []
            for seed in SEEDS:
                pts = all_curves["mean_of_window"][(member, seed)].get(domain)
                if pts:
                    vals.append(min(v for _, v in pts))
            if len(vals) == len(SEEDS):
                row[member] = {"mean": round(float(np.mean(vals)), 4),
                               "spread": round(float(max(vals) - min(vals)), 4)}
        alt[domain] = row
    result["per_domain_minimum_skill_mean_of_window"] = alt

    spreads = [r[m]["spread"]
               for r in result["per_domain_minimum_skill"].values()
               for m in r if "spread" in r[m] and r[m]["spread"] > 0]
    worst = max(result["per_domain_minimum_skill"].items(),
                key=lambda kv: max((v["spread"] for v in kv[1].values()),
                                   default=0))
    result["summary"] = {
        "max_seed_spread_over_all_domain_minima": round(max(spreads), 4),
        "worst_domain": worst[0],
        "n_seeded_cells": len(spreads),
        "n_ordering_flips": len(flips),
        "scope_note":
            "seed spread is domain-dependent; report and interpret the "
            "per-domain values rather than applying a global bound",
    }

    OUT.write_text(json.dumps(result, indent=2))

    print(f"{'domain':<18} {'member':<6} {'per seed':<26} mean   spread")
    for domain, row in result["per_domain_minimum_skill"].items():
        for member, d in row.items():
            if "per_seed" in d:
                ps = " ".join(f"{v:.3f}" for v in d["per_seed"].values())
                print(f"{domain:<18} {member:<6} {ps:<26} "
                      f"{d['mean']:.3f}  {d['spread']:.4f}")
            else:
                print(f"{domain:<18} {member:<6} {d['value']:.3f} "
                      f"{'(no seed)':<17} {'':6} {d['spread']:.4f}")
    print("\nclean skill spread:")
    for m, d in result["clean_skill_at_zero_severity"].items():
        print(f"  {m:<10} {list(d['per_seed'].values())}  spread "
              f"{d['spread']:.4f}"
              + ("  [gate only]" if "note" in d else ""))
    print(f"\nordering flips: {len(flips)} of "
          f"{result['ordering_flips_across_seeds']['n_cells_checked']} cells")
    for f in flips[:8]:
        print(f"  {f['domain']}@{f['severity']}: "
              + " | ".join(f"{k}:{'>'.join(v)}" for k, v in f['orders'].items()))
    print(f"\nmax spread {max(spreads):.4f} in {worst[0]}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
