"""Point the pipeline at your own data locations — one command, no hand-editing.

This release is a byte-for-byte replica of the working repository, so the
original absolute data paths are still hardcoded in the sources. If you have the
dataset somewhere else, run this once instead of editing ~25 files by hand:

    python3 tools/configure_paths.py --data-root /my/DATASAT --work-root /my/work
    python3 tools/configure_paths.py --dry-run          # preview only

Only two path prefixes are rewritten:
  * the shared dataset root  (dataset.zarr, norm_stats.npz, annotations_*)
  * the work root            (data/era5/, .cdsapirc, env_proxy.sh)

Nothing else in the sources is touched. Re-running with different roots is safe.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent

# Prefixes as shipped in this release.
DATA_PREFIX = "/data1/group3/DATASAT"
WORK_PREFIX = "/home/group3/group3_work/HYLiang"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", help="your dataset root (contains dataset/ and experiments/)")
    ap.add_argument("--work-root", help="your work root (contains data/era5/, .cdsapirc)")
    ap.add_argument("--dry-run", action="store_true", help="report without writing")
    args = ap.parse_args()

    if not args.data_root and not args.work_root:
        ap.error("give at least one of --data-root / --work-root")

    rules = []
    if args.data_root:
        rules.append((DATA_PREFIX, args.data_root.rstrip("/")))
    if args.work_root:
        rules.append((WORK_PREFIX, args.work_root.rstrip("/")))

    targets = sorted((PROJECT / "ewci").glob("*.py")) + sorted((PROJECT / "tools").glob("*.py"))
    total = 0
    for p in targets:
        if p.name == Path(__file__).name:
            continue
        src = p.read_text(encoding="utf-8")
        new = src
        hits = 0
        for old, repl in rules:
            n = new.count(old)
            if n:
                new = new.replace(old, repl)
                hits += n
        if hits:
            total += hits
            rel = p.relative_to(PROJECT)
            print("%-46s %2d replacement(s)%s" % (rel, hits, "  [dry-run]" if args.dry_run else ""))
            if not args.dry_run:
                p.write_text(new, encoding="utf-8")

    print("\n%d replacement(s) across the sources." % total)
    if args.dry_run:
        print("dry-run: nothing written.")
    else:
        print("Next: python3 tools/selfcheck.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())