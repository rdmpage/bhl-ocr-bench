"""Shared plumbing for every OCR producer adapter.

A producer does exactly one job: read the pinned benchmark dataset, send each page's image to
one engine, and write a parquet with the canonical schema below. Everything that is NOT
engine-specific lives here so that each adapter is only the engine call.

The canonical schema is identical across adapters:

    PageID         int64    join key, matches the GT page set exactly
    markdown       string   the engine's text for that page (never null)
    model          string   the board row this output belongs to
    error          string   null on success; a diagnostic message on failure
    producer_meta  string   JSON blob for engine-specific fields (lang, version, latency, ...)

`producer_meta` exists so engine-specific detail (Tesseract's per-volume language pack, Mistral's
response id) never widens the schema. Adapters stay comparable; their extras stay auditable.

Two contracts here are load-bearing and easy to get wrong:

1. **Errors must be encoded in the TEXT column, not the `error` column.** `score_dataset.py`
   detects a failed page by testing the OCR string for a `__ERR__` / `[... error ...]` sentinel
   (see its `_is_error`); it never reads a sibling `error` column. A producer that writes `""` on
   failure and stashes the reason elsewhere gets that page scored as "the model read nothing",
   which silently *lowers* the model's apparent error rate on the pages it actually crashed on.
   So `finalize` writes the sentinel into `markdown` and keeps `error` as a human-facing duplicate.

2. **Checkpoint per page, resume per page.** Scoring fails closed: one errored or missing page
   makes the whole run ineligible, and these runs are long (and, for hosted engines, paid). The
   checkpoint is an append-only JSONL written and fsynced after every page, so a crash costs one
   page rather than the run. Resume reads it back and skips what is already done; a later record
   for the same PageID supersedes an earlier one, which is what makes retrying failed pages work.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait

# score_dataset.py's `_is_error` treats a leading "__ERR__" as a producer error. Keep this byte
# for byte in sync with runners/normalize_outputs.py:ERROR_PREFIX in the pinned harness.
ERROR_PREFIX = "__ERR__"

CANONICAL_COLUMNS = ("PageID", "markdown", "model", "error", "producer_meta")


def error_text(message: str) -> str:
    """Render a failure as the sentinel string the harness will recognise as a producer error."""
    return f"{ERROR_PREFIX}{message}"


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------


class Checkpoint:
    """Append-only per-page JSONL, fsynced on every write.

    One JSON object per line: {"PageID": int, "markdown": str|null, "error": str|null,
    "meta": {...}}. Append-only rather than rewrite-in-place so an interrupted write can only
    ever truncate the final line, which `load` discards.
    """

    def __init__(self, path):
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = None

    def load(self) -> dict[int, dict]:
        """Read completed pages. Later records win; a torn trailing line is dropped."""
        done: dict[int, dict] = {}
        if not self.path.exists():
            return done
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    # Only the last line can be torn (append-only + fsync); anything else means
                    # the file was corrupted by something other than a crash mid-append.
                    print(f"  checkpoint: discarding unparseable line in {self.path}", flush=True)
                    continue
                if "PageID" in record:
                    done[record["PageID"]] = record
        return done

    def append(self, record: dict) -> None:
        if self._handle is None:
            self._handle = self.path.open("a", encoding="utf-8")
        self._handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._handle.flush()
        os.fsync(self._handle.fileno())

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


# ---------------------------------------------------------------------------
# Page source
# ---------------------------------------------------------------------------


def load_index(dataset, *, revision=None, split="train", image_column="image",
               id_column="PageID", extra_columns=()):
    """Load the page list WITHOUT images — ids plus whatever else routing needs.

    Kept separate from `load_pages` so an adapter can plan the whole run (page set, per-volume
    language, cost estimate) and validate its routing before a single image is decoded. Dropping
    the image column makes this cheap even on an image-heavy corpus.
    """
    from datasets import load_dataset

    ds = load_dataset(dataset, revision=revision, split=split)
    for column in (id_column, *extra_columns):
        if column not in ds.column_names:
            raise ValueError(f"dataset {dataset!r} has no column {column!r}")
    keep = {id_column, *extra_columns}
    ds = ds.remove_columns([c for c in ds.column_names if c not in keep])
    return [{"PageID": int(row[id_column]), **{c: row[c] for c in extra_columns}} for row in ds]


def _image_bytes(cell, *, page_id):
    """Get the encoded bytes out of an undecoded `datasets` image cell.

    With `decode=False` the cell is `{"bytes": ..., "path": ...}`, and which of the two is
    populated depends on how the dataset stores images: bytes are inline for a parquet-embedded
    dataset, but a folder-based dataset (which `finebooks/bhl-impact-gt` is) hands over
    `bytes=None` and a path into the local Hub cache. Handle both — assuming bytes cost us a run.
    """
    if isinstance(cell, (bytes, bytearray)):
        return bytes(cell)
    if isinstance(cell, dict):
        if cell.get("bytes"):
            return cell["bytes"]
        path = cell.get("path")
        if path:
            data = pathlib.Path(path).read_bytes()
            if data:
                return data
    raise ValueError(f"page {page_id} has an empty image cell")


def load_pages(dataset, *, revision=None, split="train", image_column="image",
               id_column="PageID", extra_columns=()):
    """Stream (PageID, image bytes, extras) from the pinned benchmark dataset.

    Images are read with `decode=False`, so each row carries the ORIGINAL encoded bytes rather
    than a decoded PIL object. That keeps memory flat while streaming and makes the bytes cheap
    to hand to a worker process or POST to an API — decoding, if the engine needs it, is the
    adapter's business.
    """
    from datasets import Image, load_dataset

    ds = load_dataset(dataset, revision=revision, split=split)
    if image_column not in ds.column_names:
        raise ValueError(f"dataset {dataset!r} has no image column {image_column!r}")
    if id_column not in ds.column_names:
        raise ValueError(f"dataset {dataset!r} has no id column {id_column!r}")
    for column in extra_columns:
        if column not in ds.column_names:
            raise ValueError(f"dataset {dataset!r} has no column {column!r}")
    ds = ds.cast_column(image_column, Image(decode=False))
    keep = [image_column, id_column, *extra_columns]
    ds = ds.remove_columns([c for c in ds.column_names if c not in keep])
    for row in ds:
        payload = _image_bytes(row[image_column], page_id=row[id_column])
        yield {
            "PageID": int(row[id_column]),
            "image_bytes": payload,
            **{c: row[c] for c in extra_columns},
        }


def resolve_revision(dataset, requested=None):
    """Pin a Hub dataset to an immutable commit before a long run reads a single page."""
    if pathlib.Path(dataset).exists():
        return None
    from huggingface_hub import HfApi

    sha = HfApi().dataset_info(repo_id=str(dataset), revision=requested).sha
    if not isinstance(sha, str) or len(sha) != 40:
        raise ValueError(f"could not resolve an immutable revision for {dataset!r}")
    return sha


# ---------------------------------------------------------------------------
# Driving the run
# ---------------------------------------------------------------------------


def run_pages(pages, work, *, checkpoint: Checkpoint, workers=1, executor="process",
              initializer=None, initargs=(), retry_errors=True, progress_every=100):
    """Run `work` over every page, checkpointing each result as it lands.

    `work` is a module-level callable (it has to be picklable for the process pool) taking the
    page dict and returning `(text, meta)`; raising marks the page a producer error.

    `executor="process"` for CPU-bound local engines, `"thread"` for network-bound APIs.
    Results are consumed as they complete, so a slow page never blocks the checkpoint. In-flight
    work is bounded at `2 * workers` so streaming a 2,165-page corpus stays memory-flat.
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

    pending = (page for page in pages if page["PageID"] not in done)
    pool_cls = ProcessPoolExecutor if executor == "process" else ThreadPoolExecutor
    kwargs = {"max_workers": workers}
    if initializer is not None:
        kwargs |= {"initializer": initializer, "initargs": initargs}

    completed, errors, started = 0, 0, time.monotonic()
    with pool_cls(**kwargs) as pool:
        in_flight: dict = {}

        def submit_next():
            page = next(pending, None)
            if page is None:
                return False
            # The image bytes are not needed after submission; keep only the id for reporting.
            in_flight[pool.submit(work, page)] = page["PageID"]
            return True

        for _ in range(max(1, 2 * workers)):
            if not submit_next():
                break

        while in_flight:
            finished, _ = wait(list(in_flight), return_when=FIRST_COMPLETED)
            for future in finished:
                page_id = in_flight.pop(future)
                try:
                    text, meta = future.result()
                    if not isinstance(text, str):
                        raise TypeError(f"work returned {type(text).__name__}, expected str")
                    record = {"PageID": page_id, "markdown": text, "error": None, "meta": meta}
                except Exception as exc:  # noqa: BLE001 - any failure is a durable error row
                    errors += 1
                    record = {"PageID": page_id, "markdown": None,
                              "error": f"{type(exc).__name__}: {exc}", "meta": {}}
                checkpoint.append(record)
                completed += 1
                if completed % progress_every == 0:
                    rate = completed / max(time.monotonic() - started, 1e-9)
                    print(f"  {completed} done ({errors} errors) · {rate:.1f} pages/s", flush=True)
                submit_next()

    checkpoint.close()
    print(f"  finished this pass: {completed} page(s), {errors} error(s)", flush=True)
    return completed, errors


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def finalize(checkpoint: Checkpoint, page_ids, *, model, out_path):
    """Write the canonical parquet, in benchmark page order, one row per page.

    Refuses to write a run that is missing pages: an incomplete table is not scoreable, and
    finding that out here beats finding it out from the scorer after a paid run.
    """
    import pandas as pd

    done = checkpoint.load()
    page_ids = [int(p) for p in page_ids]
    missing = [p for p in page_ids if p not in done]
    if missing:
        raise SystemExit(
            f"{len(missing)} of {len(page_ids)} pages are missing from the checkpoint "
            f"(examples: {missing[:5]}). Re-run the producer to resume — scoring fails closed "
            f"on an incomplete page set, so this run is not scoreable yet."
        )

    rows, failed = [], 0
    for page_id in page_ids:
        record = done[page_id]
        error = record.get("error")
        if error:
            failed += 1
            # The sentinel goes in the TEXT column: that is the only channel the harness reads.
            text = error_text(error)
        else:
            text = record["markdown"]
        rows.append({
            "PageID": page_id,
            "markdown": text,
            "model": model,
            "error": error,
            "producer_meta": json.dumps(record.get("meta") or {}, ensure_ascii=False,
                                        sort_keys=True),
        })

    frame = pd.DataFrame(rows, columns=list(CANONICAL_COLUMNS))
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out_path, index=False)

    print(f"\n{len(frame)} pages -> {out_path}")
    if failed:
        print(
            f"WARNING: {failed} page(s) are producer errors and carry the {ERROR_PREFIX} sentinel.\n"
            f"         Scoring fails closed on producer errors: this run will be INELIGIBLE.\n"
            f"         Re-run to retry them before scoring.",
            file=sys.stderr, flush=True,
        )
    return frame
