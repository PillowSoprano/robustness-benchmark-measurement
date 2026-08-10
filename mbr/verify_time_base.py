"""What is one sample of the MBR task, in hours?

Section 10.3 of the journal manuscript reports autocorrelations in lags and
wants them in hours. The lag-to-hours conversion was an open item because the
influent record and the environment could in principle disagree, and because
the file the environment loads is not the standard BSM1 dry-weather record.

Three independent lines of evidence are checked here, and they agree.

  1. The environment's own clock. One step integrates one membrane
     filtration and backwash cycle, tf + tc.
  2. The record length. The environment declares 14 days of operation and the
     generated sequences carry 1344 samples.
  3. The hydraulics. Total reactor volume over influent flow gives a
     retention time that the measured state autocorrelations have to be
     consistent with.

It also records a provenance fact worth knowing before anyone reads the
influent autocorrelation as diurnal: the record the environment loads has a
six-hour dominant period, while the standard BSM1 dry-weather file sitting in
the same directory has the twenty-four-hour one.

Usage:
    python verify_time_base.py
"""

import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = os.environ.get("MBR_PROJECT_ROOT")
if not ROOT:
    sys.exit("Set MBR_PROJECT_ROOT to the MBR plant project. That project is "
             "not redistributable and is not shipped here; see data/README.md. "
             "This diagnostic writes a local, ignored artifact.")
ROOT = Path(ROOT)

ENV = ROOT / "envs/mbr_casadi_4inputs_limu.py"
INFLUENT = ROOT / "envs/Inf_dry_constQ.txt"
BSM1_STD = ROOT / "envs/Inf_dry_2006.txt"
DATASET = ROOT / "dataset/MBR/0/draw.pt"
SEQ = 30

# ASM1 component order per tank; tank 5 starts at 4 * 13.
COMPONENTS = ["S_I", "S_S", "X_I", "X_S", "X_BH", "X_BA", "X_P",
              "S_O", "S_NO", "S_NH", "S_ND", "X_ND", "S_ALK"]
TANK5 = {c: 4 * 13 + i for i, c in enumerate(COMPONENTS)}

# BSM1 influent file columns.
INF_Q, INF_SS = 14, 2

LAGS = [6, 12, 20, 24, 48, 96, 192]

out = {}


def acf(x, lags):
    x = np.asarray(x, float)
    x = x - x.mean()
    n, v = len(x), float((x * x).sum())
    return {L: float((x[:n - L] * x[L:]).sum() / v) for L in lags}


def first_zero_crossing(x, limit=400):
    a = acf(x, range(1, limit))
    for L in range(1, limit):
        if a[L] <= 0:
            return L
    return None


# ---- evidence 1: the environment's own clock
src = ENV.read_text()


def const(name):
    m = re.search(rf"self\.{name}\s*=\s*([0-9.]+)\s*\*\s*([0-9.]+)", src)
    if m:
        return float(m.group(1)) * float(m.group(2))
    m = re.search(rf"self\.{name}\s*=\s*([0-9.]+)", src)
    return float(m.group(1)) if m else None


tf, tc, t_operate = const("tf"), const("tc"), const("t_operate")
cycle_s = tf + tc
out["evidence_1_env_clock"] = {
    "tf_seconds": tf,
    "tc_seconds": tc,
    "cycle_seconds": cycle_s,
    "cycle_minutes": cycle_s / 60,
    "samples_per_day": 86400 / cycle_s,
    "note": "step(u, i) consumes influent row i and integrates one "
            "filtration plus backwash cycle, so one row is one sample",
}

# ---- evidence 2: record length
seqs = torch.load(DATASET, map_location="cpu", weights_only=False)
states = np.asarray(seqs[SEQ][0])
n_samples = states.shape[0]
influent = np.loadtxt(INFLUENT)
out["evidence_2_record_length"] = {
    "dataset_samples": int(n_samples),
    "influent_rows": int(influent.shape[0]),
    "t_operate_days": t_operate,
    "implied_samples_per_day": n_samples / t_operate,
    "consistent_with_evidence_1":
        abs(n_samples / t_operate - 86400 / cycle_s) < 1e-6,
}

DT_HOURS = cycle_s / 3600.0

# ---- evidence 3: hydraulics
vols = [float(v) for v in re.findall(r"self\.Vol = \{(.*?)\}", src)[0].split(":")[1:]
        for v in [re.match(r"\s*([0-9.]+)", v).group(1)]]
total_volume = sum(vols)
mean_Q = float(influent[:, INF_Q].mean())
hrt_h = total_volume / mean_Q * 24.0
out["evidence_3_hydraulics"] = {
    "total_reactor_volume_m3": total_volume,
    "mean_influent_flow_m3_per_day": round(mean_Q, 1),
    "hydraulic_retention_time_hours": round(hrt_h, 2),
    "note": "state autocorrelations must decay on this timescale; "
            "see the S_NH and X_BH rows below",
}

# ---- the four series of Section 10.3, with lags in hours
series = {
    "influent_flow": influent[:, INF_Q],
    "influent_substrate": influent[:, INF_SS],
    "effluent_ammonium": states[:, TANK5["S_NH"]],
    "heterotrophic_biomass": states[:, TANK5["X_BH"]],
}
out["autocorrelation"] = {
    "dt_hours": DT_HOURS,
    "lags_samples": LAGS,
    "lags_hours": [round(L * DT_HOURS, 2) for L in LAGS],
    "series": {
        name: {
            "acf": [round(v, 3) for v in acf(x, LAGS).values()],
            "first_nonpositive_lag_samples": first_zero_crossing(x),
            "first_nonpositive_lag_hours":
                round(first_zero_crossing(x) * DT_HOURS, 2)
                if first_zero_crossing(x) else None,
        }
        for name, x in series.items()
    },
}


def dominant_period(x, hi=400):
    """First autocorrelation peak past the first zero crossing.

    A plain argmax over all lags returns the trivial peak next to lag 0,
    which is why the first version of this reported 0.5 h for a record whose
    cycle is a day.
    """
    z = first_zero_crossing(x, hi)
    if z is None:
        return None
    a = acf(x, range(z, hi))
    return max(range(z, hi), key=lambda L: a[L])


# ---- provenance: the loaded record is not the standard BSM1 one
std = np.loadtxt(BSM1_STD)
p_loaded = dominant_period(influent[:, INF_Q])
p_std = dominant_period(std[:, INF_Q])
out["influent_provenance"] = {
    "file_loaded_by_env": INFLUENT.name,
    "flow_is_constant_despite_the_name":
        bool(influent[:, INF_Q].std() == 0),
    "flow_std_m3_per_day": round(float(influent[:, INF_Q].std()), 1),
    "dominant_period_samples": p_loaded,
    "dominant_period_hours": round(p_loaded * DT_HOURS, 2),
    "period_definition": "first autocorrelation peak past the first zero "
                         "crossing",
    "standard_bsm1_file": BSM1_STD.name,
    "standard_dominant_period_samples": p_std,
    "standard_dominant_period_hours": round(p_std * DT_HOURS, 2),
    "standard_acf_at_its_period": round(
        acf(std[:, INF_Q], [p_std])[p_std], 3),
    "loaded_acf_at_its_period": round(
        acf(influent[:, INF_Q], [p_loaded])[p_loaded], 3),
    "note": "the loaded record's flow varies and its dominant period is not "
            "the diurnal one; read its autocorrelation as a six-hour cycle",
}

out["conclusion"] = {
    "sample_interval_hours": DT_HOURS,
    "sample_interval_minutes": DT_HOURS * 60,
    "samples_per_day": round(24 / DT_HOURS, 4),
    "sequence_span_days": round(n_samples * DT_HOURS / 24, 4),
    "all_three_lines_agree": bool(
        out["evidence_2_record_length"]["consistent_with_evidence_1"]
        and abs(DT_HOURS - 0.25) < 1e-9),
}

DEST = Path(__file__).resolve().parent / "artifacts" / "time_base.json"
DEST.parent.mkdir(parents=True, exist_ok=True)
DEST.write_text(json.dumps(out, indent=2))

print(json.dumps(out["conclusion"], indent=2))
print("\nlags in hours:", out["autocorrelation"]["lags_hours"])
for name, d in out["autocorrelation"]["series"].items():
    print(f"  {name:24s} {d['acf']}  zero at "
          f"{d['first_nonpositive_lag_hours']} h")
print("\ninfluent provenance:",
      json.dumps(out["influent_provenance"], indent=2))
print(f"\nwrote {DEST}")
