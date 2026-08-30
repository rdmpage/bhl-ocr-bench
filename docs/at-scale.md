# What it would cost to OCR all of BHL

BHL is ~64 million pages. This repo scored 2,165 of them. This file extrapolates: what the hosted
engines would charge, how long they would take, and which of the board's findings actually survives
the jump in scale.

**Prices checked 2026-08-30 and they move.** Every figure below is list price from a vendor page on
that date; at 64M pages you are in "contact us" territory and list is unlikely to be what you pay.
The throughput numbers are ours and are reproducible from `provenance/*.json`.

## The short version

Budget **$50k–$100k**, not the $38k floor and not the $384k ceiling — and spend the first effort on
*not sending 12.65M near-blank pages to a paid API*, because that is where both the waste and the
hallucination live. Do not plan to self-host: `surya-ocr-2` at our measured rate needs 38.5 years.

## Sticker price

| engine / tier | $/1k pages | 64M pages |
|---|---:|---:|
| Google Document AI OCR, volume (>5M/mo) | 0.60 | **$38,400** |
| AWS Textract `DetectDocumentText`, volume tier | 0.60 | **$38,400** |
| Mistral OCR 3 / 25.12-era, Batch API | 1.00 | **$64,000** |
| Google Document AI / Textract, list | 1.50 | **$96,000** |
| Mistral OCR 25.12-era, sync — the tier our `mistral-ocr-2512` row ran on | 2.00 | **$128,000** |
| Mistral OCR 4.1, Batch API | 2.00 | **$128,000** |
| Datalab convert, fast/balanced — the mode we benchmarked | 4.00 | **$256,000** |
| Mistral OCR 4.1, sync | 4.00 | **$256,000** |
| Datalab convert, accurate — not benchmarked here | 6.00 | **$384,000** |

A 10× spread. Two figures need confirming before anyone commits: Datalab's per-mode rate card is
behind a JS-rendered page, so the $4/1k fast/balanced figure comes from secondary sources rather
than the vendor's own table; and Datalab quotes enterprise volume discounts on request.

## Throughput is the binding constraint, not price

From the wall clock recovered in `provenance/*.json` (see the README's *How long each run took*),
extrapolated to 64M pages at the same concurrency:

| row | measured | 64M pages, single stream |
|---|---:|---:|
| `mistral-ocr-4-1` | 10,022 pages/h @ 8 workers | 6,387 h — **0.7 years** |
| `tesseract-5` | 9,115 pages/h @ 7 workers, local CPU | 7,022 h — **0.8 years** |
| `datalab-balanced` | 3,742 pages/h @ 12 workers, 190 rpm cap | 17,105 h — **2.0 years** |
| `surya-ocr-2` | 190 pages/h @ batch 24, M1 Pro | 337,289 h — **38.5 years** |

So this is a batch-API-and-enterprise-contract project, not a script left running. Datalab's free
tier is 10 req/min and **polling requests count against it**, which at our observed ~2.06 polls per
page is roughly 3 requests per page.

`surya-ocr-2` is the row we serve ourselves, and at 38.5 years it is not a candidate at this scale
under any hardware we would plausibly buy — see *Local hardware*, below. Note the irony that Datalab
sells the same model, hosted, for $4/1k.

## The finding that actually matters at scale

19.8% of our corpus is sparse or blank (428 of 2,165) — plates, tissue guards, endpapers. Scaled up
that is **~12.65 million BHL pages**. The board's sparse column, which is easy to skim past at 2,165
pages, becomes the whole story at 64 million.

Characters emitted per near-empty page, against a reference of ~1.4 characters:

| row | chars/page emitted on sparse pages | sparse CER | over-extraction |
|---|---:|---:|---:|
| `surya-ocr-2` | 3.1 | 2.64 | 0.7011 |
| `datalab-balanced-configured` | 3.7 | 4.31 | 0.6469 |
| `mistral-ocr-2512` | 4.9 | 3.10 | 0.7452 |
| `tesseract-5` | 10.8 | 6.88 | 0.9815 |
| `datalab-balanced-apidefault` | 39.1 | 33.58 | 0.9769 |
| **`mistral-ocr-4-1`** | **262.3** | **212.99** | **0.9948** |

`mistral-ocr-4-1` is the best row on the board for *content* — 0.0425 content CER, better than
anything else we ran — and the worst possible choice for BHL. 99.48% of what it emits on a sparse
page is text that is not there. Run it over all of BHL and it produces roughly **3.3 billion
characters of hallucinated text** across 12.65M plates and blanks, at $256k. The 25.12-era model is
both cheaper and 69× better on exactly the pages this collection has most of.

This is the same trade the README flags in *Where our rows stand*: the 0.043–0.051 headline cluster
reads content better than surya and falls apart on near-empty pages. On a natural-history collection
full of plates, the headline CER is the wrong number to shop on.

## The recommendation: classify before you pay

Do not send 64M pages to one engine. Detect blank and plate pages locally first — a cheap classifier
over 64M images costs approximately nothing next to any per-page API rate — and drop them from the
paid stream.

- Saves **$8k–$76k** depending on tier.
- Removes the hallucination exposure entirely, because the pages you drop are precisely the ones
  every engine fails on.
- Leaves ~51M content pages to route to a batch-API tier, where the engines actually agree
  (five of six rows sit in a 0.059–0.067 content band).

Then budget for what the table above leaves out: **egress**. 64M images at ~1 MB each is ~64 TB
moving into whichever cloud. Unless the scans already live there, that can exceed the OCR bill.

## Local hardware: would a newer Mac help?

Our `surya-ocr-2` run took 11.4 h for 2,165 pages on an M1 Pro — 18.94 s/page, batch 24, F16 GGUF
through llama.cpp on Metal.

The batch-scaling measurements in `provenance/surya-ocr-2.json` say this workload is compute-bound,
not bandwidth-bound: 1→6 gave 2.45×, 6→12 gave 2.62×, 12→24 gave 1.59×, so **10.2× cumulative** from
batch 1 to 24, with the fall-off on the last doubling showing a compute ceiling being approached.
Compute ceilings are what Apple's M5 generation attacks — every GPU core carries a Neural
Accelerator exposed through the Metal 4 tensor API, and Apple's own MLX measurements show **3.3–4.0×
on prefill** against M4 but only **19–27% on token generation**, which stays bandwidth-bound. Page
OCR is unusually prefill-heavy (a whole page image becomes vision tokens before any text comes out),
so it sits on the favourable side of that split.

Two caveats before buying anything:

- **A base M5 has less memory bandwidth than the M1 Pro we used** — 153 GB/s against 200 GB/s. Only
  the M5 Pro (307 GB/s) or Max clears it. On a base M5 the decode half would go slightly backwards
  while prefill leapt ahead.
- **llama.cpp has to actually enable the tensor API.** There is a documented case of a llama.cpp
  runtime shipping the upstream tensor-API code but never enabling it on M5 hardware, losing 2–3× of
  prefill. Our run used `b10360 (48d22e295)`. Verify the Metal 4 path is live before attributing any
  speedup to the hardware, or use MLX, where support is first-class.

Estimate: **2–3× on an M5 Pro/Max with a tensor-enabled build**, so 11.4 h → roughly 4–6 h for 2,165
pages. Worth it for iteration speed on this repo. It changes nothing at BHL scale: 38.5 years at 3×
is still 12.8 years.

## Sources

- [Mistral API pricing](https://mistral.ai/pricing/api/) and [Mistral OCR 4](https://mistral.ai/news/ocr-4/)
- [Datalab billing](https://documentation.datalab.to/platform/billing); rate card via [Replicate's writeup](https://replicate.com/blog/datalab-marker-and-ocr-fast-parsing)
- [AWS Textract pricing](https://aws.amazon.com/textract/pricing/)
- [Apple: Exploring LLMs with MLX and the Neural Accelerators in the M5 GPU](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)
- [llama.cpp Metal 4 tensor API not enabled on M5](https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/2040)
