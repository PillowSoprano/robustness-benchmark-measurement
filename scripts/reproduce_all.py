"""Regenerate released figures and verify artifact-backed tables.

Tier 1 of REPRODUCIBILITY.md: Python only, no simulator, no Fortran
compiler, and no external plant data. Released per-window errors are committed
under frozen_predictions/, so downstream results are recomputed rather than
rerun.

Verifies the manifest checksums first, because a figure regenerated from a
silently edited artifact is worse than no figure.

Usage:
    python scripts/reproduce_all.py [--figures-only|--tables-only]
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FROZEN = ROOT / "frozen_predictions"
OUT = ROOT / "reproduced"

# Labels follow the released manuscript numbering.
FIGURES = [
    ("Fig. 3         TEP retention, seven operators",
     "tep/plot_tep_retention_v2.py"),
    ("Fig. 4         control-authority sweep", "tep/plot_gain_sweep.py"),
]
TABLES = [
    ("Table 6   TS-Fault re-ranking",
     "reanalysis/tsfault_common_reference.py"),
]
# Artifacts the paper cites directly, listed so a tier-1 run surfaces every
# number the text rests on; each is checksum-verified by check_manifest().
ARTIFACT_TABLES = [
    ("Table 4     panel skill", "tep/panel.json"),
    ("Table 7     gain sweep minima", "tep/gain_sweep.json"),
    ("Table 8     five-architecture minima", "tep/modern_panel.json"),
    ("Table 9     regime diagnostic", "tep/regime_diagnostic.json"),
    ("Table D.9   per-seed spread", "tep/seed_spread.json"),
    ("Table F.10  CSTR admission sweep", "cstr/horizon_excitation_sweep.json"),
    ("Sec. 10.2   long-window probe", "tep/history_probe.json"),
]


def check_manifest():
    man = json.loads((FROZEN / "MANIFEST.json").read_text())
    bad = []
    for rel, rec in man.items():
        f = FROZEN / rel
        if not f.exists():
            bad.append(f"{rel}: missing")
            continue
        h = hashlib.sha256(f.read_bytes()).hexdigest()[:16]
        if h != rec["sha256"]:
            bad.append(f"{rel}: sha256 {h} != {rec['sha256']}")
    print(f"frozen predictions: {len(man)} artifacts", end="")
    if bad:
        print(" -- CHECKSUM FAILURES")
        for b in bad:
            print("  !", b)
        sys.exit(1)
    print(", checksums ok")


def main():
    check_manifest()
    OUT.mkdir(exist_ok=True)
    only = sys.argv[1] if len(sys.argv) > 1 else None
    fails = 0
    if only != "--tables-only":
        print("\nfigures")
        for label, rel in FIGURES:
            p = ROOT / rel
            if not p.exists():
                print(f"  skip  {label}  ({rel} absent)")
                continue
            r = subprocess.run([sys.executable, p.name], cwd=p.parent,
                               capture_output=True, text=True)
            ok = r.returncode == 0
            if ok:
                print(f"  ok    {label}")
            else:
                fails += 1
                print(f"  FAIL  {label}")
                print("        " + r.stderr.strip().split("\n")[-1])
    if only != "--figures-only":
        print("\ntables")
        for label, rel in TABLES:
            p = ROOT / rel
            r = subprocess.run([sys.executable, p.name], cwd=p.parent,
                               capture_output=True, text=True)
            ok = r.returncode == 0
            fails += not ok
            print(f"  {'ok  ' if ok else 'FAIL'}  {label}")
    print("\nartifact-backed tables (checksummed above)")
    for label, rel in ARTIFACT_TABLES:
        ok = (FROZEN / rel).exists()
        print(f"  {'ok' if ok else '!!':4s}  {label}  <- frozen_predictions/{rel}")
        fails += not ok
    print(f"\n{'release reproduction passed' if not fails else str(fails) + ' failed'}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
