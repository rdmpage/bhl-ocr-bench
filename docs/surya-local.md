# Running surya-ocr-2 locally — parked, blocked on macOS 13

**Status: blocked, not slow.** Investigated 2026-08-12. Revisit after a macOS upgrade.

## Why we wanted it

`surya-ocr-2` is the one model the upstream harness has already committed to boarding —
`RESULTS.md`: *"surya-ocr-2 and PP-OCRv6 join the board in v1.1, once their runs meet the same
bar."* It has open weights, so unlike our three hosted-API rows (both Mistrals and Datalab) a
self-hosted run can pin an exact model revision and would satisfy `RESULTS.md`'s *"What counts as a
score"* bar:

> We ran the inference ourselves, for every model on the board — a pinned container image, a pinned
> model revision, a pinned script commit, and the id of the job that produced each score […] a
> hosted inference router does not record which provider served a request, at what quantization, or
> under what serving configuration — and character error rate is sensitive to all three.

That is the gap this would close. It is the only reason to prefer local inference here; cost is not
the motivation.

## The machine

MacBook Pro, Apple M1 Pro (6P+2E CPU, 14-core GPU), 32 GB RAM, **macOS 13.7.8 Ventura**.
RAM is ample. The GPU is fine. The OS is the problem.

## What actually happens

Three findings that compound into a hard block. None of them is a performance issue.

1. **PyTorch will not use the GPU.**
   ```
   RuntimeError: The MPS backend is supported on MacOS 14.0+.
   ```
   `torch.backends.mps.is_built()` is True, `is_available()` is False. torch 2.11 requires
   macOS 14+; this machine is 13.7.8.

2. **llama.cpp's Metal backend crashes on model load.**
   ```
   ggml-metal-context.m:359: GGML_ASSERT(buf_dst) failed
   ggml_metal_get_tensor_async -> llama_context::decode -> common_init_from_params
   ```
   Reproducible, every attempt, immediately after `load_model` on the surya-2 GGUF.

3. **There is no CPU fallback to retreat to.**
   ```
   ValueError: Unknown inference backend 'torch'. Supported: 'vllm', 'llamacpp'.
   ```
   Surya 0.22 runs recognition only through vLLM (Linux/CUDA) or llama.cpp. Setting
   `LLAMA_ARG_N_GPU_LAYERS=0` / `GGML_METAL=0` / `LLAMA_CPP_NGL=0` does not help: the Metal backend
   is still loaded and still asserts before any offload decision matters.

So there is no configuration of Surya 0.22 that runs on macOS 13 — not a slow one, none.

**Confidence:** high that the OS is the cause. torch states the macOS 14 requirement outright, and
macOS 13 is an unusual target that the current ggml Metal build evidently is not exercised against.
Not certain: the ggml assert could in principle be an unrelated bug in this Homebrew build.

## The actual blocker is disk, not the OS

As of 2026-08-12: **6.0 GB free**. A macOS 14+ upgrade typically wants 20–40 GB free to install. So
the ordering is: free up disk → upgrade macOS → retry Surya. The OS upgrade is free and this Mac
supports it; the disk is what needs solving first.

What this work put on disk, should any of it be worth reclaiming:

| path | size | notes |
|---|---:|---|
| `~/.cache/huggingface/hub/models--datalab-to--surya-ocr-2-gguf` | 1.4 GB | re-downloadable; only needed for this |
| `~/.cache/huggingface/hub/datasets--finebooks--bhl-impact-gt` | 420 MB | the corpus; needed for any re-run, re-downloadable |
| `bhl-ocr-bench/.venv` | 276 MB | `uv sync` rebuilds it |
| `bhl-ocr-bench/runs` | 29 MB | raw OCR + checkpoints; **re-running these costs money**, keep |
| Homebrew `llama.cpp` | — | `brew uninstall llama.cpp`; wanted again after the upgrade |

Reclaiming everything reclaimable here is roughly 2 GB — useful, nowhere near an OS upgrade on its
own.

## Picking this back up

Already installed and working, so the retry is short:

- Homebrew `llama.cpp` (provides `llama-server`, which Surya spawns)
- `surya-ocr` 0.22.1 + torch 2.11 resolve cleanly on **Python 3.12** (not 3.14, the system default
  — torch has no 3.14 wheels)
- `surya-ocr-2` GGUF weights, 1.4 GB, cached

After upgrading macOS:

```bash
uv venv --python 3.12 surya-env
VIRTUAL_ENV=surya-env uv pip install surya-ocr pandas pyarrow
surya-env/bin/python -c "import torch; print(torch.backends.mps.is_available())"   # expect True
```

Then benchmark a handful of content pages before committing to all 2,165. The API in 0.22.1 is:

```python
from surya.recognition import RecognitionPredictor
rec = RecognitionPredictor()
preds = rec([pil_image], full_page=True)          # NOT det_predictor=..., that signature is gone
```

If it runs, wrap it as `producers/surya.py` using the same `producers/common.py` plumbing as every
other engine — checkpoint per page, `__ERR__` sentinel in the text column, canonical schema — and
it drops straight onto the board.

## If you want the row sooner

Run it on Linux with a GPU (an HF Job, as the published board did for all sixteen of its models).
That sidesteps Apple Silicon entirely and is the route that actually matches how the upstream
numbers were produced — which is the whole point of preferring a self-hosted row in the first place.
