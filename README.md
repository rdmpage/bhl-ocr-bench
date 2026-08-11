# bhl-ocr-bench

Producer adapters that feed OCR engines into the [FineBooks IMPACT-BHL](https://huggingface.co/datasets/finebooks/bhl-impact-gt)
evaluation harness.

The harness — [`finebooks/bhl-ocr-eval`](https://github.com/finebooks/bhl-ocr-eval) — is pinned here
as a **read-only submodule** at `harness/`. Nothing in this repo edits it. The scorer (3.0), the
normalizer (frozen at `2026-07-20a`) and the post-processing registry (3) are versioned precisely so
that numbers stay comparable with the published leaderboard; changing any of them silently
invalidates that comparison, so the pin is the point.

    git submodule status        # must report the pinned SHA, clean
    git -C harness status       # must be clean

## Layout

    harness/        pinned finebooks/bhl-ocr-eval — READ ONLY
    benchmark/      build_benchmark.py -> benchmark/gt (the local scoring GT)
    producers/      one adapter per engine; common.py is the shared plumbing
    scripts/        score.sh — the scoring invocation, with its non-obvious flags
    runs/           raw OCR parquet + per-page checkpoints (gitignored)
    scorecards/     per-page scorecards from the harness (gitignored)

## The producer contract

Every adapter does exactly one job — read the pinned dataset, send each `image` to one engine,
write a parquet — and every adapter writes the **same schema**:

| column | type | meaning |
|---|---|---|
| `PageID` | int64 | join key; matches the GT page set exactly |
| `markdown` | string | the engine's text for the page, never null |
| `model` | string | the board row this output belongs to |
| `error` | string | null on success, diagnostic on failure |
| `producer_meta` | string | JSON for engine-specific extras (language pack, response id, …) |

Engine-specific detail goes in `producer_meta` so it never widens the schema.

Two rules in `producers/common.py` are load-bearing:

- **Checkpoint and resume per page.** An append-only JSONL fsynced after every page. A crash costs
  one page, not the run. This is not a nicety: scoring **fails closed**, so one errored or missing
  page makes the whole run ineligible, and for hosted engines a re-run costs money.
- **Errors go in the text column.** `score_dataset.py` decides a page failed by testing the OCR
  *string* for a `__ERR__` sentinel — it never reads a sibling `error` column. A producer that
  writes `""` on failure gets that page scored as "the engine read nothing", which quietly
  *improves* its apparent score on pages where it actually crashed.

All 2,165 pages are sent, including the 428 sparse/blank ones. Hallucination on near-empty pages is
one of the sharpest differentiators between engines (the board's sparse CER column ranges from 1.79
to 26.93), so filtering them would remove the finding.

## Running it

    uv run benchmark/build_benchmark.py          # once: builds benchmark/gt
    uv run producers/tesseract5.py --out runs/tesseract-5/run.parquet
    scripts/score.sh runs/tesseract-5

`--limit N` smoke-tests an adapter; such a run is deliberately **not** scoreable, because the page
set will not match the benchmark and the scorer fails closed on an incomplete set.

## Why benchmark/gt exists

`finebooks/bhl-impact-gt` is the **source** corpus, not the table the scorer consumes. The published
board scored against a `prep_sample.py` derivative that carries four extra columns: `volume`,
`body_text`, `furniture_text`, `regions_json`. That dataset (`davanstrien/bhl-eval-impact-full-2165-v1`)
has since been removed from the Hub.

This matters more than a missing-column error would, because it *isn't* one.
`gt_score.score_gt_row` reads all four with `.get()`, so pointing the scorer at the source dataset
does not fail — it silently falls back to full-text scoring and returns a number that is **not** the
board's body-only headline. `benchmark/build_benchmark.py` rebuilds those columns by importing the
pinned harness's own `gt_docling` helpers and `prep_sample` sampler, so it is a reconstruction rather
than a second opinion. It reproduces the documented 1,737 content / 428 sparse_blank split.

Images are omitted from `benchmark/gt` (the scorer drops them anyway); producers read images from
the source dataset directly.

## Engines

### tesseract-5 — the plumbing acceptance test

Not a result we need; a check that this repo reproduces a known row before any paid API call. The
published board's `tesseract-5` row reads **CER 0.0642**.

**The language map is the result.** Tesseract scores ~0.19 CER on this corpus with the wrong
language pack and ~0.064 with the right per-volume ones — the difference between last place and
mid-table. `producers/tesseract5.py` reproduces the map from the pinned harness's
`drivers/tesseract-port.py`, keyed on `BarCode`. A missing pack **aborts** rather than falling back
to English, because the fallback produces a number that looks plausible and is wrong.

No post-processing is applied: `tesseract-5` is deliberately absent from the harness's
`normalize_outputs.py` REGISTRY, so the published row was scored on raw `image_to_string` output.

Requires `brew install tesseract tesseract-lang`.

## After any change

    uv run pytest                            # this repo's producer tests
    cd harness && uv run pytest              # the pinned harness suite
    cd harness && uv run scoring/scorer.py       # standalone self-test
    cd harness && uv run scoring/normalizers.py  # standalone self-test
