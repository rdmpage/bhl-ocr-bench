"""Quantify what markdown image placeholders cost a run's score.

Mistral emits `![img-0.jpeg](img-0.jpeg)` where it extracted a figure. That is an artefact of the
output format, not a transcription of ink on the page, so it inflates `over_extraction` and shows
up as CER insertions. The harness would normally handle this per-model in
`normalize_outputs.py`'s registry — but that file lives inside the pinned, read-only harness, so
registering a transform is not available to us and the run is scored raw (the same treatment
`tesseract-5` gets).

This does not change any score. It measures how much the artefact is worth, so the caveat on the
row can be a number rather than a hand-wave.

    uv run scripts/placeholder_impact.py runs/mistral-ocr-4-1/run.parquet
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

import pandas as pd

PLACEHOLDER = re.compile(r"!\[[^\]]*\]\([^)]*\)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=pathlib.Path)
    ap.add_argument("--gt", default="benchmark/gt/train.parquet")
    args = ap.parse_args()

    run = pd.read_parquet(args.run)
    gt = pd.read_parquet(args.gt)[["PageID", "sample_stratum", "body_text"]]
    frame = run.merge(gt, on="PageID")

    hits = frame.markdown.str.count(PLACEHOLDER)
    chars = frame.markdown.str.findall(PLACEHOLDER).map(lambda m: sum(len(x) for x in m))
    total_chars = frame.markdown.str.len().sum()
    gt_chars = frame.body_text.fillna("").str.len().sum()

    print(f"pages                     : {len(frame)}")
    print(f"pages with a placeholder  : {(hits > 0).sum()} ({(hits > 0).mean():.1%})")
    print(f"placeholders total        : {int(hits.sum())}")
    print(f"characters they occupy    : {int(chars.sum())}")
    print(f"OCR characters total      : {int(total_chars)}")
    print(f"GT body characters total  : {int(gt_chars)}")
    print(f"placeholder share of OCR  : {chars.sum() / max(total_chars, 1):.3%}")
    print(f"upper bound on CER inflation (if every placeholder char is a pure insertion):")
    print(f"                            {chars.sum() / max(gt_chars, 1):.4f} CER points")
    print()
    for stratum in ("content", "sparse_blank"):
        part = frame[frame.sample_stratum == stratum]
        part_hits = part.markdown.str.count(PLACEHOLDER)
        part_chars = part.markdown.str.findall(PLACEHOLDER).map(lambda m: sum(len(x) for x in m))
        print(f"{stratum:13s}: {(part_hits > 0).sum():5d}/{len(part):5d} pages, "
              f"{int(part_hits.sum()):5d} placeholders, {int(part_chars.sum()):7d} chars")


if __name__ == "__main__":
    sys.exit(main())
