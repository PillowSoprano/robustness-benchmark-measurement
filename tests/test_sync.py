"""Repository integrity: the committed artifacts are the ones the manifest names.

Three checks, all cheap, all run before anything else in `reproduce_all.py`:

  1. Every artifact under `frozen_predictions/` appears in the manifest, and
     every manifest entry exists on disk. A file that arrives without a
     manifest line is an artifact nobody checksummed.
  2. Checksums match. A figure regenerated from a silently edited artifact is
     worse than no figure.
  3. Nothing that should not be committed is committed: build products, byte
     code, and absolute paths out of somebody's home directory.

Usage:
    python tests/test_sync.py
"""

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "frozen_predictions"
MANIFEST = FROZEN / "MANIFEST.json"

problems = []
notes = []


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


# ---- 1 and 2: manifest coverage and checksums
manifest = json.loads(MANIFEST.read_text())
entries = {k: v for k, v in manifest.items() if not k.startswith("_")}
on_disk = {str(p.relative_to(FROZEN)) for p in FROZEN.rglob("*.json")
           if p.name != "MANIFEST.json"}

for rel in sorted(on_disk - set(entries)):
    problems.append(f"artifact not in the manifest: {rel}")
for rel in sorted(set(entries) - on_disk):
    problems.append(f"manifest names a missing artifact: {rel}")
for rel in sorted(on_disk & set(entries)):
    p, want = FROZEN / rel, entries[rel]
    got = digest(p)
    if got != want["sha256"]:
        problems.append(f"{rel}: sha256 {got} against {want['sha256']}")
    elif p.stat().st_size != want["bytes"]:
        problems.append(f"{rel}: {p.stat().st_size} bytes against "
                        f"{want['bytes']}")
notes.append(f"{len(on_disk)} artifacts, all checksummed")

# ---- 3: nothing that should not be committed is committed. "Committed"
# means tracked by git: a tier-1 run legitimately writes __pycache__/ and
# reproduced/, and .gitignore keeps them out of the index, so this check
# reads the index rather than transient generated files in the checkout.
tracked = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True,
                         text=True).stdout.splitlines()
BAD_SUFFIX = (".so", ".pyc", ".o", ".mod", ".npz", ".pt")
for rel in tracked:
    if rel.endswith(BAD_SUFFIX) or ".DS_Store" in rel:
        problems.append(f"build product committed: {rel}")
    if "__pycache__" in rel:
        problems.append(f"byte-code committed: {rel}")

# assembled from pieces so this file does not trip its own check
HOME_PATH = re.compile("|".join(["/" + "Users/", "/" + "home/[a-z]",
                                 "C:" + chr(92) * 2 + "Users"]))
for rel in tracked:
    fp = ROOT / rel
    if not fp.is_file():
        continue
    if fp.suffix not in {".py", ".md", ".json", ".txt", ".yml", ".cff", ".pyf"}:
        continue
    try:
        text = fp.read_text()
    except UnicodeDecodeError:
        continue
    if HOME_PATH.search(text):
        problems.append(f"absolute home path in {rel}")

print("=== repository integrity ===")
for n in notes:
    print("  note:", n)
print()
if problems:
    print(f"{len(problems)} PROBLEM(S):")
    for x in problems:
        print("  !", x)
    sys.exit(1)
print("no problems found")
