"""surya-ocr-2 producer — local inference, the only row on this board we actually served ourselves.

    surya-env/bin/python producers/surya_ocr2.py --limit 12 --out runs/surya-smoke/run.parquet
    surya-env/bin/python producers/surya_ocr2.py --out runs/surya-ocr-2/run.parquet

Run it with `surya-env/bin/python`, not `uv run`: torch and surya-ocr live in a separate Python
3.12 venv (see docs/surya-local.md), not in the project's `.venv`.

NOT `surya.py`. Every producer puts its own directory first on `sys.path` to import `common`, so a
module named `surya.py` here shadows the installed `surya` package and `import surya.settings`
fails with "'surya' is not a package".

WHY THIS ROW EXISTS. Every other row we produce is a hosted API, and `RESULTS.md` is explicit that
a hosted score is weaker evidence: *"a hosted inference router does not record which provider served
a request, at what quantization, or under what serving configuration — and character error rate is
sensitive to all three."* surya-ocr-2 has open weights, so this row can name all three. The
provenance block records the GGUF repo revision, the `general.file_type` read out of the GGUF
header, and the llama.cpp build that served it.

NOT THE SAME SERVING CONFIG AS THE PUBLISHED BOARD. Upstream's `uv-scripts/ocr/surya-ocr.py` serves
the model with vLLM at `dtype="bfloat16"` on a CUDA GPU. This runs the GGUF build through
llama.cpp on Apple Silicon Metal. The weights are 16-bit on both sides — the GGUF header says
`general.file_type = 1` (F16), not a lossy 4- or 8-bit quant — so this is a different 16-bit
kernel, not a quantized approximation of one. That is a small difference, but it IS a difference,
and it is exactly the kind RESULTS.md says CER is sensitive to. Recorded, not hidden.

SERIALIZATION IS COPIED FROM THE UPSTREAM DRIVER, DELIBERATELY. Surya 0.22 returns per-block HTML,
not text. `serialize_page` below reproduces `uv-scripts/ocr/surya-ocr.py`'s `serialize_pages`
exactly — reading order, skip `skipped` and empty blocks, `BeautifulSoup(...).get_text(" ",
strip=True)`, join with newlines. Two independent reasons not to improvise here:

1. **The diplomatic lane does not strip markup.** Only the reading lane runs `strip_markup`
   (`scoring/normalizers.py`); `_diplomatic` is NFC plus whitespace. Handing over raw HTML would
   score every `<p>` and `</p>` as inserted characters and produce a diplomatic CER that measures
   our serialization choice rather than the model.
2. **surya-ocr-2 is not in the harness `normalize_outputs.py` REGISTRY.** Nothing downstream will
   clean this up, so whatever the producer writes is what gets scored.

BATCHED, AND THAT IS FREE. Pages go to the predictor in batches. Measured on this machine (M1 Pro),
each step up is a large win and every step produced output byte-identical to the smaller grouping
over the same pages:

    1 -> 6    2.45x     6 -> 12   2.62x     12 -> 24   1.59x

Returns are flattening by 24, which is why that is the default. Throughput only: because the
concatenated block HTML is unchanged, batch size cannot move the score. Lower it if memory is
tight — `--batch-size 1` is the slow, always-safe setting.

ONE PROCESS, NOT A POOL. `common.run_pages` fans out over a process pool, which is wrong here:
`RecognitionPredictor` spawns a `llama-server` child, so N workers would mean N copies of the model
resident and competing for one GPU. This module drives its own batched loop instead and reuses
`common`'s Checkpoint/finalize contract unchanged — per-page checkpointing still holds, because a
batch's results are written page by page as it lands.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import pathlib
import platform
import struct
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import common  # noqa: E402

MODEL = "surya-ocr-2"
DEFAULT_DATASET = "finebooks/bhl-impact-gt"
GGUF_REPO = "datalab-to/surya-ocr-2-gguf"

# GGUF `general.file_type` enum -> name, for the handful we could plausibly be handed. The point of
# reading this at runtime is that "we ran F16 weights" is a provenance claim, and a claim we read
# out of the file we actually loaded is worth more than one written into a comment by hand.
GGUF_FILE_TYPES = {0: "ALL_F32", 1: "MOSTLY_F16", 2: "MOSTLY_Q4_0", 3: "MOSTLY_Q4_1",
                   7: "MOSTLY_Q8_0", 8: "MOSTLY_Q5_0", 9: "MOSTLY_Q5_1", 10: "MOSTLY_Q2_K",
                   12: "MOSTLY_Q3_K", 14: "MOSTLY_Q4_K", 16: "MOSTLY_Q5_K", 18: "MOSTLY_Q6_K"}


def _harness_pin() -> str | None:
    harness = pathlib.Path(__file__).resolve().parent.parent / "harness"
    try:
        return subprocess.run(["git", "-C", str(harness), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


# ---------------------------------------------------------------------------
# Serialization — see the module docstring; keep byte-compatible with the upstream driver
# ---------------------------------------------------------------------------


def html_to_text(html: str) -> str:
    from bs4 import BeautifulSoup

    return BeautifulSoup(html, "html.parser").get_text(" ", strip=True)


def serialize_page(page_result) -> tuple[str, dict]:
    """Blocks -> one page of plain text, plus the layout facts worth auditing later.

    `skipped` blocks are dropped rather than emitted empty: Surya sets it for the labels in
    `SKIP_OCR_LABELS` ({Figure, Image, Diagram, Blank-Page}), i.e. regions it decided carry no
    text. That decision is part of the model's behaviour and is left alone — but it is also the
    single most consequential thing that can silently blank a page, so the labels and the skip
    count are recorded per page in `producer_meta` and summarised at the end of the run.
    """
    parts, labels, skipped, errored = [], [], 0, 0
    for block in sorted(page_result.blocks, key=lambda b: b.reading_order):
        labels.append(block.label)
        if block.error:
            errored += 1
        if block.skipped or not block.html:
            skipped += 1
            continue
        text = html_to_text(block.html)
        if text:
            parts.append(text)
    meta = {
        "engine": "surya-ocr-2",
        "backend": "llamacpp",
        "blocks": len(page_result.blocks),
        "blocks_skipped": skipped,
        "blocks_errored": errored,
        "labels": labels,
    }
    return "\n".join(parts), meta


# ---------------------------------------------------------------------------
# The batched run loop
# ---------------------------------------------------------------------------


def run_batches(pages, predictor, *, checkpoint: common.Checkpoint, batch_size: int,
                retry_errors: bool = True, progress_every: int = 25):
    """Drive the predictor in batches, checkpointing every page as its batch lands.

    A batch that raises is retried page by page. Batches are a throughput trick, so one page that
    upsets the model must not take eleven healthy pages down with it — without the fallback, a
    single poison page would keep failing its whole batch on every resume and the run could never
    complete.
    """
    done = checkpoint.load()
    if retry_errors:
        stale = [page_id for page_id, record in done.items() if record.get("error")]
        for page_id in stale:
            del done[page_id]
        if stale:
            print(f"  retrying {len(stale)} previously-errored page(s)", flush=True)
    if done:
        print(f"  resuming: {len(done)} page(s) already complete", flush=True)

    completed, errors, started = 0, 0, time.monotonic()
    batch: list[dict] = []

    def flush(batch):
        nonlocal completed, errors
        if not batch:
            return
        from PIL import Image

        images, ids = [], []
        for page in batch:
            with Image.open(io.BytesIO(page["image_bytes"])) as handle:
                images.append(handle.convert("RGB"))
            ids.append(page["PageID"])

        t0 = time.monotonic()
        try:
            results = predictor(images, full_page=True)
            if len(results) != len(images):
                raise ValueError(f"predictor returned {len(results)} results for {len(images)}")
            records = []
            for page_id, result in zip(ids, results):
                text, meta = serialize_page(result)
                records.append({"PageID": page_id, "markdown": text, "error": None, "meta": meta})
        except Exception as exc:  # noqa: BLE001
            if len(images) > 1:
                print(f"  batch of {len(images)} failed ({type(exc).__name__}: {exc}); "
                      f"retrying page by page", flush=True)
                for page in batch:
                    flush([page])
                return
            records = [{"PageID": ids[0], "markdown": None,
                        "error": f"{type(exc).__name__}: {exc}", "meta": {}}]

        elapsed = time.monotonic() - t0
        for record in records:
            if record.get("error"):
                errors += 1
            elif record.get("meta") is not None:
                record["meta"]["seconds"] = round(elapsed / len(records), 2)
                record["meta"]["batch_size"] = len(records)
            checkpoint.append(record)
            completed += 1
            if completed % progress_every == 0:
                rate = completed / max(time.monotonic() - started, 1e-9)
                print(f"  {completed} done ({errors} errors) · {rate * 3600:.0f} pages/h", flush=True)

    for page in pages:
        if page["PageID"] in done:
            continue
        batch.append(page)
        if len(batch) >= batch_size:
            flush(batch)
            batch = []
    flush(batch)

    checkpoint.close()
    print(f"  finished this pass: {completed} page(s), {errors} error(s)", flush=True)
    return completed, errors


# ---------------------------------------------------------------------------
# Environment / provenance
# ---------------------------------------------------------------------------


def gguf_provenance(repo: str, revision: str | None) -> dict:
    """Resolve the GGUF repo to a commit, fetch both files AT that commit, and read the precision.

    Surya's own loader calls `hf_hub_download` with no `revision` (see
    `inference/backends/llamacpp.py:_download_gguf_files`), so left alone it takes whatever `main`
    points at on the day — which is not a pin, and a pinned model revision is most of the reason
    this row is worth producing. Both files are fetched here at an explicit SHA and handed to the
    backend through `SURYA_GGUF_LOCAL_{MODEL,MMPROJ}_PATH`, which it prefers over downloading.

    The precision is read out of the GGUF header rather than assumed: "we served 16-bit weights"
    is a provenance claim, and one read from the file actually loaded beats one written by hand.
    """
    from huggingface_hub import HfApi, hf_hub_download
    from surya.settings import settings

    sha = HfApi().model_info(repo_id=repo, revision=revision).sha
    path = pathlib.Path(hf_hub_download(repo, settings.SURYA_GGUF_MODEL_FILE, revision=sha))
    mmproj = pathlib.Path(hf_hub_download(repo, settings.SURYA_GGUF_MMPROJ_FILE, revision=sha))

    file_type, size_label, arch = None, None, None
    with path.open("rb") as handle:
        if handle.read(4) != b"GGUF":
            raise ValueError(f"{path} is not a GGUF file")
        struct.unpack("<I", handle.read(4))  # format version
        handle.read(8)                       # tensor count
        n_kv = struct.unpack("<Q", handle.read(8))[0]

        def read_string():
            length = struct.unpack("<Q", handle.read(8))[0]
            return handle.read(length).decode("utf-8", "replace")

        scalars = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?",
                   10: "Q", 11: "q", 12: "d"}

        def read_value(kind):
            if kind == 8:
                return read_string()
            if kind == 9:
                element = struct.unpack("<I", handle.read(4))[0]
                count = struct.unpack("<Q", handle.read(8))[0]
                return [read_value(element) for _ in range(count)]
            fmt = scalars[kind]
            return struct.unpack("<" + fmt, handle.read(struct.calcsize("<" + fmt)))[0]

        for _ in range(n_kv):
            key = read_string()
            value = read_value(struct.unpack("<I", handle.read(4))[0])
            if key == "general.file_type":
                file_type = value
            elif key == "general.size_label":
                size_label = value
            elif key == "general.architecture":
                arch = value

    return {
        "repo": repo,
        "revision": sha,
        "file": settings.SURYA_GGUF_MODEL_FILE,
        "mmproj_file": settings.SURYA_GGUF_MMPROJ_FILE,
        "file_type": file_type,
        "file_type_name": GGUF_FILE_TYPES.get(file_type, f"unknown({file_type})"),
        "size_label": size_label,
        "architecture": arch,
        "bytes": path.stat().st_size,
        "local_model_path": str(path),
        "local_mmproj_path": str(mmproj),
    }


def llama_cpp_version() -> str | None:
    from surya.settings import settings

    try:
        proc = subprocess.run([settings.LLAMA_CPP_BINARY, "--version"],
                              capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    line = (proc.stderr or proc.stdout or "").strip().splitlines()
    return line[0].strip() if line else None


def check_environment() -> dict:
    """Fail before a long run rather than during it.

    The historical failure mode on this machine was macOS 13, where torch refused MPS and
    llama.cpp's Metal backend asserted on model load (docs/surya-local.md). Both are OS-version
    problems with no CPU fallback — Surya 0.22 recognition supports only vLLM and llama.cpp — so
    an unavailable MPS device means this will not run at all, not that it will run slowly.
    """
    import shutil

    import torch
    from surya.settings import settings

    if shutil.which(settings.LLAMA_CPP_BINARY) is None:
        raise SystemExit(
            f"{settings.LLAMA_CPP_BINARY} is not on PATH — Surya spawns it to serve the GGUF.\n"
            f"Install it with:  brew install llama.cpp"
        )
    if not torch.backends.mps.is_available():
        raise SystemExit(
            "torch reports MPS unavailable. On Apple Silicon this is normally a macOS version "
            "problem (torch requires macOS 14+), and there is no CPU fallback: Surya 0.22 "
            "recognition supports only the vllm and llamacpp backends. See docs/surya-local.md."
        )
    return {
        "torch": torch.__version__,
        "torch_device": str(settings.TORCH_DEVICE_MODEL),
        "llama_cpp": llama_cpp_version(),
        "llama_cpp_ngl": settings.LLAMA_CPP_NGL,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--revision", default=None, help="pin the dataset revision (resolved to a SHA)")
    ap.add_argument("--split", default="train")
    ap.add_argument("--id-column", default="PageID")
    ap.add_argument("--image-column", default="image")
    ap.add_argument("--model-revision", default=None,
                    help=f"pin the {GGUF_REPO} revision; defaults to whatever main resolves to now")
    ap.add_argument("--out", default=f"runs/{MODEL}/run.parquet")
    ap.add_argument("--checkpoint", default=None, help="default: <out dir>/checkpoint.jsonl")
    ap.add_argument("--batch-size", type=int, default=24,
                    help="pages per predictor call. Throughput only — output measured "
                         "byte-identical across batch sizes; 24 is ~10x faster than 1 on an M1 "
                         "Pro and is where the returns flatten. Lower it if memory is tight.")
    ap.add_argument("--limit", type=int, default=None,
                    help="smoke-test against N pages. NOT scoreable: the page set will not match "
                         "the benchmark, and score_dataset fails closed on an incomplete set.")
    ap.add_argument("--no-retry-errors", dest="retry_errors", action="store_false", default=True)
    args = ap.parse_args()

    out_path = pathlib.Path(args.out)
    checkpoint_path = pathlib.Path(args.checkpoint) if args.checkpoint \
        else out_path.parent / "checkpoint.jsonl"

    environment = check_environment()
    weights = gguf_provenance(GGUF_REPO, args.model_revision)
    revision = common.resolve_revision(args.dataset, args.revision)

    # Point the backend at the exact files resolved above. It only honours these when BOTH are
    # set; otherwise it re-downloads from `main` and the recorded revision would be a fiction.
    # Assigned on the settings OBJECT, not just the environment: `settings` is a pydantic
    # BaseSettings singleton that read the environment when `surya.settings` was first imported,
    # which has already happened by here, so an env var set now would be silently ignored. The
    # env vars are set too, for any subprocess that builds its own Settings.
    from surya.settings import settings

    settings.SURYA_GGUF_LOCAL_MODEL_PATH = weights["local_model_path"]
    settings.SURYA_GGUF_LOCAL_MMPROJ_PATH = weights["local_mmproj_path"]
    os.environ["SURYA_GGUF_LOCAL_MODEL_PATH"] = weights["local_model_path"]
    os.environ["SURYA_GGUF_LOCAL_MMPROJ_PATH"] = weights["local_mmproj_path"]

    print(f"{MODEL} | {weights['repo']}@{weights['revision'][:12]} "
          f"{weights['file_type_name']} {weights['size_label']} | "
          f"{environment['llama_cpp']} | torch {environment['torch']} "
          f"on {environment['torch_device']}", flush=True)

    index = common.load_index(args.dataset, revision=revision, split=args.split,
                              image_column=args.image_column, id_column=args.id_column)
    if args.limit:
        index = index[:args.limit]
        print(f"WARNING: --limit {args.limit} — smoke test only, not a scoreable run", flush=True)
    wanted = {row["PageID"] for row in index}

    already = len(common.Checkpoint(checkpoint_path).load())
    remaining = max(len(index) - already, 0)
    print(f"dataset {args.dataset} @ {revision or 'local'} | {len(index)} pages | "
          f"{already} already cached | batch {args.batch_size}", flush=True)
    if remaining:
        # ~15 s/page at batch 24 on an M1 Pro, measured on text-heavy pages; the corpus also holds
        # blank versos and plates that run much faster, so this is an upper bound. Either way the
        # run is hours, which is worth saying before it starts rather than after.
        print(f"estimate: up to ~{remaining * 15 / 3600:.1f}h for {remaining} remaining pages "
              f"(~15 s/page at batch 24, M1 Pro). Resumable — Ctrl-C costs at most one batch.",
              flush=True)

    def streamed():
        for page in common.load_pages(args.dataset, revision=revision, split=args.split,
                                      image_column=args.image_column, id_column=args.id_column):
            if page["PageID"] in wanted:
                yield page

    from surya.recognition import RecognitionPredictor

    started = time.monotonic()
    predictor = RecognitionPredictor()
    checkpoint = common.Checkpoint(checkpoint_path)
    run_batches(streamed(), predictor, checkpoint=checkpoint, batch_size=args.batch_size,
                retry_errors=args.retry_errors)
    common.finalize(checkpoint, [row["PageID"] for row in index], model=MODEL, out_path=out_path)

    # A page whose every block was skipped reads as "the model saw no text here". Sometimes that is
    # correct (the corpus has blank versos and full-page plates with empty GT), but it is also what
    # a layout misfire looks like, so surface the count rather than leaving it to be discovered
    # from the score.
    records = common.Checkpoint(checkpoint_path).load()
    blank = sum(1 for r in records.values() if not r.get("error") and not (r.get("markdown") or ""))
    total_skipped = sum((r.get("meta") or {}).get("blocks_skipped", 0) for r in records.values())
    print(f"layout: {blank} page(s) produced no text; {total_skipped} block(s) skipped as "
          f"Figure/Image/Diagram/Blank-Page across the run")

    meta_path = out_path.parent / "producer-run.json"
    meta_path.write_text(json.dumps({
        "prompt": None,
        "note": "model-native OCR mode (RecognitionPredictor, full_page=True)",
        "producer": "bhl-ocr-bench producers/surya_ocr2.py",
        "model": MODEL,
        "weights": weights,
        "surya_version": "0.22.1",
        "inference_backend": "llamacpp",
        "environment": environment,
        "batch_size": args.batch_size,
        "batch_note": "batching is a throughput setting only. Measured 1->6 2.45x, 6->12 2.62x, "
                      "12->24 1.59x, and at every step the concatenated block HTML was "
                      "byte-identical to the smaller grouping over the same pages, so batch size "
                      "cannot move the score",
        "serving_caveat": "Self-hosted, so the weights and the serving stack are both pinned above "
                          "— but this is NOT the configuration the published board used. Upstream "
                          "(uv-scripts/ocr/surya-ocr.py) serves surya-ocr-2 with vLLM at "
                          "dtype=bfloat16 on CUDA; this is the F16 GGUF through llama.cpp on Apple "
                          "Silicon Metal. Both are 16-bit weights, not a lossy quantization, but "
                          "the kernels differ and RESULTS.md notes CER is sensitive to serving "
                          "configuration.",
        "dataset": args.dataset,
        "dataset_resolved_revision": revision,
        "postproc_version": "0",
        "postprocessing": "block HTML -> plain text, reproducing the upstream driver's "
                          "serialize_pages exactly: reading order, drop skipped/empty blocks, "
                          "BeautifulSoup.get_text(' ', strip=True), join with newlines. Required "
                          "rather than cosmetic: surya-ocr-2 is absent from the harness "
                          "normalize_outputs REGISTRY, and the diplomatic scoring lane does not "
                          "strip markup, so raw HTML would be scored as inserted characters.",
        "pages_without_text": blank,
        "blocks_skipped": total_skipped,
        "platform": f"{platform.system()} {platform.release()} {platform.machine()}",
        "wall_clock_s": round(time.monotonic() - started, 1),
        "harness_pin": _harness_pin(),
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"run provenance -> {meta_path}")


if __name__ == "__main__":
    main()
