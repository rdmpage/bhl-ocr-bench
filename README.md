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
    scripts/        board.sh — the one entry point: score rows, then rebuild the board
                    scrub_local_paths.py — keeps one laptop's paths out of the artifacts
    runs/           raw OCR parquet + per-page checkpoints (gitignored)
    scorecards/     per-page scorecards from the harness (gitignored)
    provenance/     tracked run-provenance snapshot, copied from runs/ by board.sh
    boards/         leaderboard.json, built from every scorecard at once

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

Three rules in `producers/common.py` are load-bearing:

- **Checkpoint and resume per page.** An append-only JSONL fsynced after every page. A crash costs
  one page, not the run. This is not a nicety: scoring **fails closed**, so one errored or missing
  page makes the whole run ineligible, and for hosted engines a re-run costs money.
- **Errors go in the text column.** `score_dataset.py` decides a page failed by testing the OCR
  *string* for a `__ERR__` sentinel — it never reads a sibling `error` column. A producer that
  writes `""` on failure gets that page scored as "the engine read nothing", which quietly
  *improves* its apparent score on pages where it actually crashed.
- **Wall clock accumulates across passes.** Because runs resume, timing one invocation reports only
  the *last* pass: `mistral-ocr-2512` recorded 5.5 s for 2,165 pages, having found every page
  already cached. `WallClock` appends each pass to a `passes.jsonl` beside the checkpoint and sums
  them. A pass that raises is still recorded — a crashed pass burned real time. Runs made before
  this existed report `wall_clock_s: null` rather than a confident total that omits the passes it
  cannot see; only `surya-ocr-2` (41,075 s, observed start to finish in one invocation) carries a
  trustworthy number today.

All 2,165 pages are sent, including the 428 sparse/blank ones. Hallucination on near-empty pages is
one of the sharpest differentiators between engines (the board's sparse CER column ranges from 1.79
to 26.93), so filtering them would remove the finding.

## Running it

    uv run benchmark/build_benchmark.py          # once: builds benchmark/gt
    uv run producers/tesseract5.py --out runs/tesseract-5/run.parquet
    scripts/board.sh tesseract-5                 # score that row, then rebuild the board

`--limit N` smoke-tests an adapter; such a run is deliberately **not** scoreable, because the page
set will not match the benchmark and the scorer fails closed on an incomplete set.

`scripts/board.sh` is the only entry point for everything downstream of the cached OCR. Named rows
score just those (~30 s each for tesseract, plus the bake); no arguments re-scores all six.
Either way it refreshes `provenance/` from the run dirs, regenerates `boards/leaderboard.json`, and
scrubs this machine's paths out of both — the scorer records `ocr_source` / `gt_source` as absolute
paths, so without that step every bake would put a home directory back into a public repo. `--list`
prints the registered rows. It runs no inference, so it is a pure function of (cached
raw output, run provenance, pinned scorer): **re-running it must not move a number** — re-scoring
`tesseract-5` and rebaking leaves `boards/leaderboard.json` byte-identical.

Scoring a row is not separable from baking, and that is the point: each scorecard embeds the run
provenance it was scored against, and the board reads it back out from there, so a bake that
skipped re-scoring would republish stale provenance. Adding an engine means adding it to the
`ROWS` registry at the top of the script — a run that is not a declared board row cannot be
scored, and since smoke runs are not scoreable anyway, every scoreable run is a board row.

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

## Where our rows stand

Headline is micro-averaged CER on the reading lane, body-only, over all 2,165 pages. Every row is
eligible: 2,165 pages, 0 producer errors, 0 missing.

| model | CERdip | CERread | 95% CI | content | sparse | recallµ | overµ |
|---|---:|---:|---|---:|---:|---:|---:|
| **surya-ocr-2** (self-served) | **0.0521** | **0.0391** | [0.0321, 0.0471] | **0.0366** | **2.64** | 0.9666 | 0.0366 |
| mistral-ocr-2512 | 0.0862 | 0.0593 | [0.0377, 0.0731] | 0.0564 | 3.10 | 0.9631 | 0.0494 |
| datalab-balanced-configured | 0.1008 | 0.0630 | [0.0398, 0.0866] | 0.0590 | 4.31 | **0.9703** | 0.0385 |
| tesseract-5 | 0.0816 | 0.0672 | [0.0419, 0.0843] | 0.0607 | 6.88 | 0.9225 | 0.0859 |
| datalab-balanced-apidefault | 0.2070 | 0.1692 | [0.0602, 0.2882] | 0.1375 | 33.58 | 0.9685 | 0.1432 |
| mistral-ocr-4-1 | 0.3393 | 0.2439 | [0.0722, 0.4615] | 0.0425 | 212.99 | 0.9687 | 0.2186 |

The **content** and **sparse** columns are the two strata (1,737 / 428 pages) that the headline
micro-averages together. Splitting them is what makes `mistral-ocr-4-1` legible: at content CER
0.0425 it reads running text better than anything else we ran except surya, and its last place is
one defect on near-empty pages rather than bad OCR. See *mistral-ocr-4-1*, below.

**Read the confidence intervals before reading the order.** They are bootstrapped over six volumes,
so they are wide and they overlap: surya-ocr-2's [0.0321, 0.0471] overlaps mistral-ocr-2512's
[0.0377, 0.0731] and datalab-configured's [0.0398, 0.0866]. The top three are **not** separated at
this sample size. What is clearer is the diplomatic lane, where surya-ocr-2 leads by a wider margin
(0.0521 vs 0.0862) — it reproduces case, ligatures and diacritics more faithfully, not just the
folded reading text.

`pisciumquerelaee00sche` is the hardest volume for *every* engine (surya 0.1341, mistral-2512
0.1317, datalab 0.1479, tesseract 0.1749) — a 1662 Latin text with long-s throughout, where the GT
itself is visibly lossy. That is a corpus property, not a model finding.

## Interleaved with the published board

Our rows against the 14 models in the pinned harness's
[RESULTS.md](https://github.com/finebooks/bhl-ocr-eval/blob/cce90c5cebb450303463d9fd6b7f4a893c209392/RESULTS.md), ranked
together on the same headline. Same ground truth, same page set (`411a92e4a0e4`), same frozen
scorer 3.0 / normalizer `2026-07-20a` — which is the entire reason the pin exists. **Ours in bold.**

| CER read | model | content | sparse | recallµ | loop % |
|---:|---|---:|---:|---:|---:|
| 0.0235 | rednote-hilab/dots.ocr | 0.0220 | 1.79 | 0.9704 | 3.88 |
| 0.0237 | rednote-hilab/dots.mocr | 0.0216 | 2.23 | 0.9836 | 0.88 |
| 0.0305 | ATH-MaaS/OvisOCR2 | 0.0280 | 2.79 | 0.9614 | 2.54 |
| **0.0391** | **surya-ocr-2** (self-served) | **0.0366** | **2.64** | **0.9666** | — |
| 0.0392 | PaddlePaddle/PaddleOCR-VL-1.6 | 0.0367 | 2.69 | 0.9415 | 6.47 |
| 0.0432 | allenai/olmOCR-2-7B-1025-FP8 | 0.0302 | 13.64 | 0.9573 | 0.55 |
| 0.0489 | lightonai/LightOnOCR-2-1B | 0.0351 | 15.79 | 0.9688 | 2.08 |
| 0.0491 | zai-org/GLM-OCR | 0.0236 | 26.93 | 0.9674 | 0.51 |
| 0.0507 | Qwen/Qwen3.5-9B | 0.0314 | 19.98 | 0.9587 | 1.52 |
| 0.0575 | baidu/Qianfan-OCR | 0.0404 | 17.89 | 0.9540 | 1.06 |
| **0.0593** | **mistral-ocr-2512** | **0.0564** | **3.10** | **0.9631** | n/a |
| 0.0604 | baidu/Unlimited-OCR | 0.0451 | 16.19 | 0.9537 | 0.46 |
| 0.0617 | deepseek-ai/DeepSeek-OCR | 0.0590 | 2.96 | 0.9206 | 0.55 |
| 0.0620 | deepseek-ai/DeepSeek-OCR-2 | 0.0600 | 2.09 | 0.9262 | 0.32 |
| **0.0630** | **datalab-balanced-configured** | **0.0590** | **4.31** | **0.9703** | n/a |
| 0.0642 | tesseract-5 *(published)* | 0.0591 | 5.49 | 0.9210 | 0.00 |
| **0.0672** | **tesseract-5** *(ours)* | **0.0607** | **6.88** | **0.9225** | 0.00 |
| **0.1692** | **datalab-balanced-apidefault** | **0.1375** | **33.58** | **0.9685** | n/a |
| 0.1965 | ds4sd/SmolDocling-256M-preview | 0.1946 | 2.18 | 0.8519 | 5.31 |
| **0.2439** | **mistral-ocr-4-1** | **0.0425** | **212.99** | **0.9687** | n/a |

Four things to read before the ordering.

**Our numbers run ~5% pessimistic, and the two tesseract rows measure it.** Same engine, same
pages, same frozen scorer, run twice: the published row reads 0.0642, ours 0.0672 (+0.0030
headline, +0.0016 content). That gap is the calibration offset on every row we produced —
`tesseract-5` is in this repo to measure it, which is why it is a plumbing acceptance test rather
than a result. Applied to surya, 0.0391 becomes ~0.0361 board-equivalent.

**Loop % is not comparable across the two halves.** The board's loop rate comes from a generating
model reporting `finish_reason == "length"`; looped pages are then excluded from that row's other
numbers. Neither Mistral's `/v1/ocr` nor Datalab reports anything equivalent, so their cells read
`n/a` — *unknown*, not zero (see *mistral-ocr-4-1*, below). Our rows are therefore scored on the
whole corpus including their own worst pages, while PaddleOCR-VL-1.6 directly below surya is scored
with 6.47% of its hardest pages removed. That makes our rows conservative relative to the neural
ones, not flattered by them.

**The sparse column is where our engines do well and the mid-board VLMs do not.** surya's 2.64
is beaten only by dots.ocr, DeepSeek-OCR-2 and SmolDocling. Meanwhile the whole 0.043–0.051 cluster
reads content *better* than surya (GLM-OCR 0.0236 vs 0.0366) and falls apart on near-empty pages
(GLM-OCR 26.93). On a collection with plates and blanks — most real collections — that trade is the
finding, and it is invisible in the headline.

**Provenance is not symmetric, so this is a comparison and not a merged board.** RESULTS.md's
*What counts as a score* requires inference the FineBooks team ran themselves under a pinned image
and model revision; a hosted API cannot attest quantization or serving configuration. Our Mistral
and Datalab rows are hosted and could not join that board on those terms. `surya-ocr-2` is the one
row we serve ourselves — and even it is the F16 GGUF through llama.cpp on Apple Silicon Metal,
where upstream serves bfloat16 through vLLM on CUDA. Both are 16-bit weights rather than a lossy
quantization, but the kernels differ and CER is sensitive to that.

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

### surya-ocr-2 — the one row we serve ourselves

Open weights, so this is the only row that can name what served it — which is exactly what
`RESULTS.md`'s *"we ran the inference ourselves"* bar asks for and what none of the three
hosted-API rows can do. Produced locally by `producers/surya_ocr2.py` through llama.cpp on Apple
Silicon; run it with `surya-env/bin/python`, not `uv run`.

Previously **blocked on macOS 13** (torch needs macOS 14+ for Metal; llama.cpp's Metal backend
asserted on model load). The macOS 26 upgrade cleared both, with nothing else changed.

**It is self-served, but it is not the published board's serving configuration.** Upstream runs
surya-ocr-2 under vLLM at `dtype="bfloat16"` on CUDA; this is the F16 GGUF through llama.cpp on
Metal. Both are 16-bit — the GGUF header reports `MOSTLY_F16`, not a lossy 4/8-bit quant — but the
kernels differ, and RESULTS.md is explicit that CER is sensitive to serving configuration. The
caveat ships in the run's `producer-run.json`.

Two traps, both documented in [docs/surya-local.md](docs/surya-local.md): Surya returns **per-block
HTML** and the diplomatic scoring lane does *not* strip markup, so the producer must flatten to
text (it copies the upstream driver's serialization exactly); and Surya's loader downloads the GGUF
with **no revision**, so the producer resolves and pins the SHA itself or the recorded pin is
fiction. Batching is pure throughput — 24 pages/call is ~10x faster than 1 with byte-identical
output.

**Result: CER 0.0391 reading / 0.0521 diplomatic**, 2,165 pages, 0 errors, in 11.4h on an M1 Pro.
Top of our board on both lanes, though the reading-lane CIs overlap the next two rows (see above).

**Its sparse-page behaviour is the best on the board (2.64)** and that is downstream of the layout
skip, not in spite of it. Surya drops `Figure/Image/Diagram/Blank-Page` blocks, so on the corpus's
plates it stays quiet where other engines emit text against a near-empty reference — the failure
mode that put datalab-apidefault at 33.58 and mistral-ocr-4-1 at 212.99.

The same skip is also its one real cost. 23 pages came back empty despite having GT text: plates
whose only text is a short caption, which Surya labels as a single `Picture` and skips. 22 of the 23
fall under the 80-char `sparse_blank` threshold so they never touch the headline; one
(page `9739742`, 97 chars) lands in the content stratum. It shows up in the region evidence as
`caption` 0.69 and `page_footer` 0.69 against `text` 0.97 and `footnote` 0.99. This is upstream
behaviour, not ours — the reference driver skips the same blocks — so the row keeps it and reports
it rather than quietly diverging from the implementation it is meant to match.

## After any change

    uv run pytest                            # this repo's producer tests
    cd harness && uv run pytest              # the pinned harness suite
    cd harness && uv run scoring/scorer.py       # standalone self-test
    cd harness && uv run scoring/normalizers.py  # standalone self-test
