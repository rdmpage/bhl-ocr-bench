"""Build the local scoring GT from the pinned source dataset.

`finebooks/bhl-impact-gt` is the SOURCE corpus, not the thing the scorer consumes. The published
board scored against a `prep_sample.py` derivative (`davanstrien/bhl-eval-impact-full-2165-v1`,
since removed from the Hub) which carries four columns the source does not:

    volume           the language/stratification key (a copy of BarCode)
    body_text        docling BODY layer  — what CER and recall actually score
    furniture_text   docling FURNITURE layer (page_header/page_footer)
    regions_json     per-label region token evidence

This matters more than it looks. `gt_score.score_gt_row` reads all four with `.get()`, so pointing
the scorer straight at the source dataset does NOT fail — it silently falls back to full-text
scoring and produces a number that is not the board's body-only headline. Rebuilding them here is
what makes our tesseract row comparable with the published one.

Images are deliberately omitted: the scorer drops the image column anyway, and leaving it out
turns a multi-GB local GT into a few MB. Producers read images from the source dataset directly.

    uv run benchmark/build_benchmark.py --out benchmark/gt

The derived columns come from the PINNED harness's own `gt_docling` helpers and its `prep_sample`
sampler, imported rather than reimplemented, so this stays a faithful reconstruction rather than a
second opinion about what the GT means.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

HARNESS = pathlib.Path(__file__).resolve().parent.parent / "harness"
sys.path.insert(0, str(HARNESS / "scoring"))
sys.path.insert(0, str(HARNESS / "runners"))

import gt_docling as G  # noqa: E402
import prep_sample as P  # noqa: E402

SOURCE = "finebooks/bhl-impact-gt"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", default=SOURCE)
    ap.add_argument("--source-revision", default=None,
                    help="branch/tag/commit; resolved once to an immutable commit")
    ap.add_argument("--n", type=int, default=2165, help="pages to take (2165 = the full corpus)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-text", type=int, default=80,
                    help="sparse_blank/content stratum threshold; 80 is the board's value")
    ap.add_argument("--out", default="benchmark/gt")
    args = ap.parse_args()

    import pandas as pd
    from huggingface_hub import hf_hub_download

    revision = P.resolve_source_revision(args.source, args.source_revision)
    print(f"source {args.source} @ {revision}")
    metadata = hf_hub_download(args.source, "metadata.parquet", repo_type="dataset",
                              revision=revision)
    frame = pd.read_parquet(metadata)
    print(f"{len(frame)} source rows")

    sample = P.stratified(frame, args.n, args.seed, threshold=args.min_text)
    sample["sample_stratum"] = sample["text"].map(
        lambda text: P.classify_sample_stratum(text, args.min_text)
    )
    print(f"selected {len(sample)} pages: volumes={dict(sample.BarCode.value_counts())}; "
          f"strata={dict(sample.sample_stratum.value_counts())}")

    provenance = P.sampler_provenance(
        seed=args.seed, requested_n=args.n, source_repo=args.source,
        source_revision=revision, threshold=args.min_text,
    )

    records = []
    for _, row in sample.iterrows():
        records.append({
            "PageID": int(row["PageID"]),
            "BarCode": row["BarCode"],
            "volume": row["BarCode"],
            "language": None,
            "text": row["text"],
            "body_text": G.body_text(row["docling"]),
            "furniture_text": G.furniture_text(row["docling"]),
            "regions_json": json.dumps(G.regions(row["docling"]), ensure_ascii=False),
            "sample_stratum": row["sample_stratum"],
            **provenance,
            "xml_path": row["xml_path"],
        })

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # A directory of parquet with no loader script is what `load_dataset(<path>)` expects, which is
    # the local-GT path score_dataset.py supports (resolve_dataset_revision returns None for an
    # existing path and benchmark provenance falls back to a content fingerprint).
    target = out / "train.parquet"
    pd.DataFrame(records).to_parquet(target, index=False)
    print(f"{len(records)} pages -> {target}")

    empty_body = sum(1 for r in records if not (r["body_text"] or "").strip())
    print(f"body_text empty on {empty_body} page(s); "
          f"sparse_blank stratum = {sum(1 for r in records if r['sample_stratum'] == 'sparse_blank')}")


if __name__ == "__main__":
    main()
