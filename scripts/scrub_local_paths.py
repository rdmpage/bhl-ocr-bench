#!/usr/bin/env python3
"""Rewrite machine-local absolute paths out of the tracked artifacts.

    uv run scripts/scrub_local_paths.py boards/leaderboard.json provenance/*.json
    uv run scripts/scrub_local_paths.py --check ...     # exit 1 if anything would change

`boards/leaderboard.json` and `provenance/*.json` are generated, not written by hand: the harness
scorer records `ocr_source` / `gt_source` as absolute paths, and the surya producer records where
llama.cpp found its weights. What those fields identify is provenance worth keeping. The *prefix*
is one laptop's home directory, which means nothing to a reader and ships in a public repo.

So the repo root becomes a relative path and `$HOME` becomes `~`. Everything identifying *what*
was read survives untouched — including the Hugging Face snapshot hash, which is the weights
revision and the whole point of recording the path at all.

Two deliberate choices:

  Text substitution, not a JSON round-trip. `leaderboard.py` and the producers dump with different
  options (`sort_keys`, `ensure_ascii`), so re-encoding would rewrite bytes unrelated to the paths
  and break the "re-baking must not move a number" check that `board.sh` rests on.

  Repo root before `$HOME`, because the repo lives under the home directory. The other order would
  turn the repo root into `~/Development/...` and never match it.

Idempotent: running it twice changes nothing, which is what makes it safe to wire into every bake.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
HOME = pathlib.Path.home()

# Longest-prefix first; see the module docstring.
REWRITES = [
    (f"{ROOT}/", ""),
    (str(ROOT), "."),
    (f"{HOME}/", "~/"),
    (str(HOME), "~"),
]


def scrub(text: str) -> str:
    for old, new in REWRITES:
        text = text.replace(old, new)
    return text


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", type=pathlib.Path)
    ap.add_argument("--check", action="store_true",
                    help="report what would change and exit 1, without writing")
    args = ap.parse_args()

    dirty = []
    for path in args.paths:
        if not path.exists():
            continue
        original = path.read_text(encoding="utf-8")
        scrubbed = scrub(original)
        if scrubbed == original:
            continue
        dirty.append(path)
        if not args.check:
            path.write_text(scrubbed, encoding="utf-8")

    if args.check and dirty:
        print("local paths present in: " + ", ".join(str(p) for p in dirty), file=sys.stderr)
        return 1
    if dirty:
        for path in dirty:
            print(f"  scrubbed {path.relative_to(ROOT) if path.is_absolute() else path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
