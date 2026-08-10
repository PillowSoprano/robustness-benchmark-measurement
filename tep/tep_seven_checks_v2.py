"""Seven-checks re-examination on the corrected + completed TEP tables.

Differences from tep_seven_checks.py (v1):
  - consumes retention/tables_v2.npz (shutdown-masked, extended grids) and
    merges retention/tables_regime_act2.npz when present (regime domain,
    second actuation operator);
  - reports shutdown-censored levels explicitly;
  - operator-choice check now has two pairs: obs bias vs noise, and
    actuation idv6 vs idv14.

Outputs retention/SEVEN_CHECKS_v2.md and seven_checks_v2.json.
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
OUT = HERE / "retention"

MODELS = ["persistence", "ridge", "mlp_s42", "gru_s42",
          "mlp_s123", "gru_s123", "mlp_s7", "gru_s7"]
CORE = ["persistence", "ridge", "mlp_s42", "gru_s42"]
TAU, ALPHA = 0.10, 0.10

tables = {"norm": {}, "phys": {}}
domains = {}
for fname in ("tables_v2.npz", "tables_regime_act2.npz"):
    p = OUT / fname
    if not p.exists():
        continue
    Z = np.load(p)
    for key in Z.files:
        space, m, d, s = key.split("|")
        s = float(s)
        tables[space].setdefault(m, {})[(d, s)] = Z[key]
        domains.setdefault(d, set()).add(s)
domains = {d: sorted(v) for d, v in domains.items()}

out = {"gate": {}, "window_agg": [], "state_agg": []}
lines = ["# TEP seven-checks v2: corrected tables, all seven domains\n",
         f"Domains and grids: "
         + "; ".join(f"{d} ({len(v)} levels, max {v[-1]})"
                     for d, v in domains.items()) + "\n"]

lines.append("## Gate semantics\n")
acc = {m: np.nanmean(tables["norm"][m][("obs_bias", 0.0)]) for m in CORE}
lines.append("clean accuracy (norm RMSE): "
             + ", ".join(f"{m}={acc[m]:.3f}" for m in CORE) + "\n")
lines.append("| domain | member | relative radius | error@max sev |")
lines.append("|---|---|---:|---:|")
rows = {}
for d, levels in domains.items():
    for m in CORE:
        r0 = tables["norm"][m][(d, 0.0)]
        radius = None
        for s in levels[1:]:
            rs = tables["norm"][m][(d, s)]
            ok = np.isfinite(rs) & np.isfinite(r0)
            p = np.mean(rs[ok] / r0[ok] - 1 > TAU)
            if p >= ALPHA and radius is None:
                prev = levels[levels.index(s) - 1]
                radius = 0.5 * (prev + s)
        emax = float(np.nanmean(tables["norm"][m][(d, levels[-1])]))
        rows[(d, m)] = (radius, emax)
        lines.append(f"| {d} | {m} | "
                     f"{'censored' if radius is None else round(radius,3)} | "
                     f"{emax:.3f} |")
lines.append("")
accv = [-acc[m] for m in CORE]
for d in domains:
    rad = [rows[(d, m)][0] for m in CORE]
    err = [-rows[(d, m)][1] for m in CORE]
    if all(r is not None for r in rad):
        rad_note = f"{spearmanr(accv, rad).statistic:+.2f}"
    else:
        rad_note = f"censored for {sum(r is None for r in rad)}/4 members"
    rho_err = spearmanr(accv, err).statistic
    out["gate"][d] = {"rho_relative_radius": rad_note,
                      "rho_abs_error": float(rho_err)}
    lines.append(f"Spearman(accuracy, rel-radius) {d}: {rad_note}; "
                 f"Spearman(accuracy, error@max): {rho_err:+.2f}")
lines.append("")

lines.append("## Aggregation axes\n")

def ranking(space, agg, d, s):
    vals = {}
    for m in MODELS:
        if m == "persistence":
            continue
        r = tables[space][m][(d, s)]
        r = r[np.isfinite(r)]
        vals[m] = r.mean() if agg == "mean" else np.sqrt(np.mean(r ** 2))
    return sorted(vals, key=vals.get)

n_checked = n_wswap = n_sswap = 0
for d, levels in domains.items():
    for s in levels:
        n_checked += 1
        rk_mean = ranking("norm", "mean", d, s)
        rk_pool = ranking("norm", "pool", d, s)
        rk_phys = ranking("phys", "mean", d, s)
        if rk_mean != rk_pool:
            n_wswap += 1
            out["window_agg"].append({"domain": d, "sev": s,
                                      "mean": rk_mean[:3],
                                      "pooled": rk_pool[:3]})
        if rk_mean != rk_phys:
            n_sswap += 1
            out["state_agg"].append({"domain": d, "sev": s,
                                     "norm": rk_mean[:3],
                                     "phys": rk_phys[:3]})
lines.append(f"window aggregation (mean vs pooled): ranking changed in "
             f"{n_wswap}/{n_checked} (domain, severity) cells")
lines.append(f"state aggregation (normalized vs physical): ranking changed in "
             f"{n_sswap}/{n_checked} cells\n")

lines.append("## Operator choice within domain\n")
out["operator_choice"] = {}
for label, da, db in (("observation: bias vs noise", "obs_bias", "obs_noise"),
                      ("actuation: idv6 vs idv14",
                       "actuation_idv6", "actuation_idv14")):
    if da not in domains or db not in domains:
        continue
    sa = domains[da][min(3, len(domains[da]) - 1)]
    sb = domains[db][min(3, len(domains[db]) - 1)]
    rka = ranking("norm", "mean", da, domains[da][-1])
    rkb = ranking("norm", "mean", db, domains[db][-1])
    out["operator_choice"][label] = {"rank_a": rka, "rank_b": rkb,
                                     "reorders": rka != rkb}
    lines.append(f"{label}: max-severity ranking "
                 f"{'DIFFERS' if rka != rkb else 'identical'}")
    lines.append(f"  {da}@{domains[da][-1]}: {rka}")
    lines.append(f"  {db}@{domains[db][-1]}: {rkb}\n")

text = "\n".join(lines)
(OUT / "SEVEN_CHECKS_v2.md").write_text(text)
with open(OUT / "seven_checks_v2.json", "w") as f:
    json.dump(out, f, indent=2, default=str)
print(text)
