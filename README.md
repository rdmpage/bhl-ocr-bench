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

### mistral-ocr-4-1

    export MISTRAL_API_KEY=...            # or write it to ~/.config/mistral/api_key
    uv run producers/mistral_ocr.py --limit 5 --out runs/mistral-smoke/run.parquet
    uv run producers/mistral_ocr.py --model mistral-ocr-4-1 --model-label mistral-ocr-4-1 \
        --out runs/mistral-ocr-4-1/run.parquet

`/v1/ocr` is bespoke — a document in, structured JSON with a `pages[]` array out — so the harness's
`run_openai.py` cannot drive it. There is no prompt surface at all, which is why the run provenance
records `{"prompt": null, "note": "model-native OCR mode"}` as the literal truth rather than a
placeholder.

**Pin a concrete version, not an alias.** `mistral-ocr-latest` and `mistral-ocr-4` are floating
aliases; measured on 2026-08-11 both returned byte-identical output to `mistral-ocr-4-1`, while
`mistral-ocr-4-0` and `mistral-ocr-3` are genuinely different models. Worse, the API echoes the
alias you sent back at you rather than the version it resolved to, so a run made through an alias
cannot say afterwards what actually produced it. The producer verifies `--model` against
`/v1/models` before spending anything and aborts with the list of available OCR models.

**v4 regressed badly on blank pages; v3 did not.** On the 50 blank pages where v4-1 produced most
output, under an identical request shape, v4-1 emitted 520,997 characters (median 1,841/page) and
leaked what reads as its own annotation rubric ("According to Rule 2 (UNDERSCORE & LINE RULES)") on
16 of them; `mistral-ocr-3` emitted 2,783 characters total (median 1/page) and leaked on none. This
is a model regression, not a pipeline artefact — the same adapter, the same images, the same request.
The independently-run NHM scorecard (`mistral-ocr-2512-cleaned`, same page-set fingerprint
`411a92e4a0e4`) reports sparse CER 1.93, consistent with v3 simply not having the defect. Note their
row is additionally `-cleaned`, so it is not raw output either. **If you want a usable Mistral row on
a corpus containing plates and blanks, use v3.**

**Read its Loop % as "unknown", not "zero".** The board's loop rate comes from a generating model
reporting `finish_reason == "length"`. `/v1/ocr` reports no such thing, so a repetition loop is
indistinguishable from a long transcription and the row shows 0.00. This engine demonstrably does
loop — on blank plates it emits repeated LaTeX and unrelated boilerplate. The consequence is that
Mistral's row is scored on the **whole** corpus including its own worst pages, whereas a row like
PaddleOCR-VL-1.6 (6.47% loop) is scored with 6.47% of its hardest pages removed. That makes this
row conservative relative to those, not flattered by them — but the two numbers are not
like-for-like, and the difference is a measurement artefact, not a quality finding.

### datalab-balanced

    export DATALAB_API_KEY=...            # or write it to ~/.config/datalab/api_key
    uv run producers/datalab.py --limit 5 --out runs/datalab-smoke/run.parquet
    uv run producers/datalab.py --mode balanced --workers 12 --rpm 190 \
        --out runs/datalab-balanced/run.parquet

Bespoke multipart `/api/v1/convert`, **asynchronous** (submit returns a `request_check_url` to
poll), so `run_openai.py` cannot drive it. Two undocumented details cost a smoke run to find:
the endpoint validates the upload's declared **MIME type** (WebP posted as
`application/octet-stream` is rejected as "Invalid file type"), and the completed result carries
the text in `markdown` while volunteering no `versions` object — so this row cannot even report
which build served it.

Rate limits are tiered and **polls count toward them**: free tier is 10 req/min (≈8 hours for the
corpus at ~2.8 requests/page), Team is 200/min (≈45 minutes). `--rpm` throttles client-side.

**Two undocumented form fields matter more than anything else here.** `disable_image_captions=true`
stops the engine writing AI-generated figure descriptions into the markdown (it empties the alt
text; the `![](…)` tag itself remains). `additional_config` — a JSON-encoded field — carries
`keep_pageheader_in_output` and `keep_pagefooter_in_output`. Setting all three took the row from
CER 0.1692 to **0.0630**, content 0.1375 → 0.0590, sparse 33.58 → 4.31, with recall and
over-extraction both becoming the best on the board (0.9703 / 0.0385). Both configurations are kept
as separate rows (`-apidefault` and `-configured`) because they are different measurements, not a
before-and-after of the same one.

The furniture flags did more than change policy: `page_header` token evidence went 0.23 → 0.99 and
`page_footer` 0.12 → 0.84 as expected, but `section_header` also rose 0.94 → 0.98, i.e. with header
dropping enabled the engine was discarding some material the GT counts as **body**, not just
furniture. That is why body CER improved rather than worsening — see the caveat in the git history:
captions and furniture changed in the same run, so the split between them is not cleanly attributed.

**The api-default row's headline is dominated by image captioning, not reading.** Datalab emits AI-generated image
*descriptions* as markdown alt text — `![A scientific plate showing 42 numbered eggs of various
bird species, arranged in a 7x6 grid…](…)` — averaging 135 characters across 1,287 placeholders,
4.5% of all output. The frozen reading normalizer already strips markdown pipes, headings, bold,
LaTeX and a fixed HTML tag vocabulary, but `![…](…)` alt text is prose and survives, so it is
scored as transcription. Stripping just those placeholders moves content CER from **0.1375 to
0.0919** and the headline from 0.1692 to 0.1144, with recall unchanged at 0.9691 — see
`diagnostics/` (deliberately outside the `scorecards/` glob, so it can never enter the board).

The board row stays raw, because correcting it properly means registering a transform in
`normalize_outputs.py` inside the read-only harness. That is the same policy-vs-accuracy confound
DESIGN.md describes for furniture: this row measures what the product emits, which is not the same
question as how well it reads.

Note also its furniture policy is the opposite of Mistral's: page_header evidence 0.23 and
page_footer 0.12 (vs Mistral v3's 0.90/0.65), i.e. it is a clean-reading engine that drops running
heads and page numbers. Per DESIGN.md that is policy, not quality.

### surya-ocr-2 — parked

Investigated and **blocked on macOS 13**, not merely slow: PyTorch needs macOS 14+ for Metal,
llama.cpp's Metal backend asserts on model load, and Surya 0.22 has no CPU fallback (recognition
supports only vLLM or llama.cpp). See [docs/surya-local.md](docs/surya-local.md) for the evidence,
the disk-space situation that gates the OS upgrade, and the retry recipe.

It matters because it is the one engine here whose open weights would let a row meet `RESULTS.md`'s
*"we ran the inference ourselves"* bar, which none of the three hosted-API rows do.

## After any change

    uv run pytest                            # this repo's producer tests
    cd harness && uv run pytest              # the pinned harness suite
    cd harness && uv run scoring/scorer.py       # standalone self-test
    cd harness && uv run scoring/normalizers.py  # standalone self-test
