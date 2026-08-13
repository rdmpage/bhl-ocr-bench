# Running surya-ocr-2 locally

**Status: unblocked and running.** Blocked on macOS 13 when first investigated 2026-08-12; the
machine was upgraded to macOS 26.6.1 the same day and both failures went away. This file now
records how the row is produced and what is load-bearing about it.

## Why we wanted it

`surya-ocr-2` is the one model the upstream harness has already committed to boarding —
`RESULTS.md`: *"surya-ocr-2 and PP-OCRv6 join the board in v1.1, once their runs meet the same
bar."* It has open weights, so unlike our three hosted-API rows (both Mistrals and Datalab) a
self-hosted run can pin an exact model revision and satisfy `RESULTS.md`'s *"What counts as a
score"* bar:

> We ran the inference ourselves, for every model on the board — a pinned container image, a pinned
> model revision, a pinned script commit, and the id of the job that produced each score […] a
> hosted inference router does not record which provider served a request, at what quantization, or
> under what serving configuration — and character error rate is sensitive to all three.

That is the gap this closes. It is the only reason to prefer local inference here; cost is not the
motivation.

## What the OS upgrade fixed

Both original blockers were the OS, as suspected. On macOS 26.6.1, with nothing else changed:

| was | now |
|---|---|
| `RuntimeError: The MPS backend is supported on MacOS 14.0+` | `torch.backends.mps.is_available()` is `True` |
| `ggml-metal-context.m:359: GGML_ASSERT(buf_dst) failed` on model load | loads and serves fine |

The third finding stands and still matters: **there is no CPU fallback.** Surya 0.22 recognition
supports only `vllm` (Linux/CUDA) and `llamacpp`; `'torch'` is not a backend. So MPS being
unavailable means the producer cannot run at all rather than running slowly, which is why
`check_environment` treats it as fatal.

Disk is no longer the constraint either: 39 GB free, against 6.0 GB at the time of the block.

## The row

Produced by `producers/surya_ocr2.py`. Run it with the dedicated venv, **not** `uv run` — torch and
surya-ocr live in `surya-env` (Python 3.12; torch has no 3.14 wheels), not the project `.venv`:

```bash
uv venv --python 3.12 surya-env
VIRTUAL_ENV=surya-env uv pip install surya-ocr pandas pyarrow datasets
brew install llama.cpp                      # provides llama-server, which Surya spawns

surya-env/bin/python producers/surya_ocr2.py --limit 12 --out runs/surya-smoke/run.parquet
surya-env/bin/python producers/surya_ocr2.py --out runs/surya-ocr-2/run.parquet
```

### Four things that are load-bearing

1. **The file is not called `surya.py`.** Every producer puts its own directory first on `sys.path`
   to import `common`, so `producers/surya.py` shadows the installed `surya` package and
   `import surya.settings` dies with *"'surya' is not a package"*.

2. **The output must be flattened from HTML to text in the producer.** Surya 0.22 returns per-block
   HTML. `surya-ocr-2` is absent from the harness `normalize_outputs.py` REGISTRY, so nothing
   downstream will clean it up — and the **diplomatic scoring lane does not strip markup** (only
   `_reading` calls `strip_markup`; `_diplomatic` is NFC plus whitespace). Handing over raw HTML
   would score every `<p>` as inserted characters and measure our serialization instead of the
   model. `serialize_page` reproduces the upstream driver's `serialize_pages` exactly: reading
   order, drop `skipped`/empty blocks, `BeautifulSoup(...).get_text(" ", strip=True)`, join with
   newlines.

3. **The weights pin has to be done by hand.** Surya's loader calls `hf_hub_download` with **no**
   `revision` (`inference/backends/llamacpp.py:_download_gguf_files`), so left alone it takes
   whatever `main` points at that day. The producer resolves the SHA itself, fetches both files at
   it, and passes them via `SURYA_GGUF_LOCAL_MODEL_PATH` / `SURYA_GGUF_LOCAL_MMPROJ_PATH` — which
   the backend honours only if **both** are set. These are assigned on the `settings` object, not
   just the environment: `settings` is a pydantic `BaseSettings` singleton that has already read
   the environment by then, so an env var set later is silently ignored.

4. **One process, not a pool.** `RecognitionPredictor` spawns a `llama-server` child, so
   `common.run_pages`' process pool would mean N copies of the model competing for one GPU. The
   producer drives its own batched loop and keeps `common`'s Checkpoint/finalize contract.

### Batching is free

Measured on this machine, each step up is a large win, and every step produced output
**byte-identical** to the smaller grouping over the same pages:

| grouping | speedup | output identical |
|---|---:|---|
| 1 → 6 | 2.45x | yes |
| 6 → 12 | 2.62x | yes |
| 12 → 24 | 1.59x | yes |

Returns flatten by 24, hence the default. Because the concatenated block HTML does not change,
batch size cannot move the score — it is throughput only. `--batch-size 1` is the slow, safe
setting. A batch that raises is retried page by page, so one poison page cannot take 23 healthy
ones down with it or wedge every resume.

Full run: ~9h upper bound for 2,165 pages, checkpointed per page and resumable.

## What this row can and cannot claim

**Can:** self-served, with the GGUF repo revision, the weight precision read out of the GGUF header
at runtime, and the llama.cpp build all recorded in `producer-run.json`.

**Cannot:** claim to reproduce the published board's serving configuration. Upstream
(`uv-scripts/ocr/surya-ocr.py`, listed in the harness's `emit_launch_commands.py` `FOREIGN` set)
serves surya-ocr-2 with **vLLM at `dtype="bfloat16"` on CUDA**; this is the **F16 GGUF through
llama.cpp on Apple Silicon Metal**. Both sides are 16-bit — the GGUF header reports
`general.file_type = 1` (`MOSTLY_F16`), a 632M-parameter model in 1.27 GB, so this is *not* a lossy
4- or 8-bit quantization — but the kernels differ, and RESULTS.md says CER is sensitive to serving
configuration. The caveat is recorded in the run provenance rather than left implicit.

If you want the row on exactly the published footing, run it on Linux with a GPU (an HF Job, as the
board did for all sixteen of its models).

## Layout skipping is the thing to watch

Surya drops blocks whose label is in `SKIP_OCR_LABELS` (`{Figure, Image, Diagram, Blank-Page}`).
That is model behaviour and is left alone, but it is also the single most consequential way a page
can silently come out blank, so the producer records the labels per page in `producer_meta` and
prints a run-level count of blank pages and skipped blocks.

On the 12-page smoke run, 5 pages produced no text and **all 5 have empty ground truth** — the skip
was correct every time.

On the full run, 352 pages produced no text and 1,747 blocks were skipped. Audited against GT:
**23 pages were blanked despite having GT text**, every one of them a plate labelled as a single
`Picture` whose only text is a short caption. 22 of the 23 have under 80 characters of GT, i.e.
they are `sparse_blank` stratum and never reach the headline; the exception is page `9739742` at 97
characters, which does count. No page came back badly truncated (0 pages returned under 35% of GT
length while having GT over 500 characters).

The net of the skip is positive: it is *why* this row has the board's best sparse-page CER (2.64,
against 4.31 for datalab-configured and 33.58 for datalab-apidefault), because it stays silent on
plates instead of emitting text against a near-empty reference. The cost is visible in the region
evidence — `caption` 0.69 and `page_footer` 0.69, against `text` 0.97 and `footnote` 0.99.

Left as-is deliberately: the upstream driver skips the same blocks, so OCR'ing `Picture` regions
would make this row stop matching the implementation it exists to reproduce. It is the same
policy-vs-quality axis DESIGN.md describes for furniture.

## One transient inference failure, and why it cost nothing

The run logged four consecutive `400 ... Failed to initialize samplers: failed to parse grammar`
warnings from llama-server. Four is exactly one item exhausting `max_retries=3` (one attempt plus
three retries) in `inference/backends/openai_client.py`, which swallows the exception and returns
`GenerationResult(raw="", error=True)`.

That does **not** silently blank a page. In `full_page` mode an errored generation makes
`_build_page` return `None`, which triggers Surya's documented per-page fallback to layout plus
block-mode OCR. The page was still read, just through a different prompt path — which is why the
checkpoint records 0 error rows and 0 block-level errors for the whole run.

Worth knowing anyway, because the mechanism is one to distrust in general: a swallowed inference
error is invisible in the producer's own error count. The GT audit above, not the error count, is
what actually rules out silent damage.
