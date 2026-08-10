"""Resolve a result file to the frozen copy, or to a live run if one exists.

During a live experiment, each analysis writes beside its own script. In the
release, the corresponding results live under `frozen_predictions/` with
flatter names. This maps one layout to the other and prefers a newly generated
result when one is present.
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FROZEN = ROOT / "frozen_predictions"

# live-result relative path  ->  frozen name
MAP = {
    "fingerprint/raw_results.json": "mbr/fingerprint.json",
    "skill_retention/raw_results.json": "mbr/skill_retention.json",
    "skill_retention/tau_abs_check.json": "mbr/tau_abs.json",
    "artifacts/panel_report_dual_aggregation.json": "mbr/panel.json",
    "retention/results_v2.json": "tep/retention.json",
    "retention/regime_act2_results.json": "tep/regime_act2.json",
    "retention/tau_abs_check.json": "tep/tau_abs.json",
    "retention/seven_checks_v2.json": "tep/seven_checks.json",
    "artifacts/panel_report.json": "tep/panel.json",
    "modern/modern_panel.json": "tep/modern_panel.json",
    "gain_sweep/gain_sweep.json": "tep/gain_sweep.json",
    "history_probe/history_probe.json": "tep/history_probe.json",
}


def resolve(path):
    """Live result if present, else the frozen copy. Accepts str or Path."""
    p = Path(path)
    if p.exists():
        return p
    for rel, frozen in MAP.items():
        if str(p).replace(os.sep, "/").endswith(rel):
            f = FROZEN / frozen
            if f.exists():
                return f
    return p          # let the caller raise with the original name
