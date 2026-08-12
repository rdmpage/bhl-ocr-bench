"""Datalab producer — hosted `/api/v1/convert` (Marker / Surya / Chandra).

    export DATALAB_API_KEY=...            # or write it to ~/.config/datalab/api_key
    uv run producers/datalab.py --limit 5 --out runs/datalab-smoke/run.parquet
    uv run producers/datalab.py --mode balanced --out runs/datalab-balanced/run.parquet

Bespoke again — `multipart/form-data` in, structured JSON out — so `run_openai.py` cannot drive
it, for the same reason Mistral could not. Two things differ from the Mistral adapter and both
shape the code:

1. **It is asynchronous.** The POST returns `{request_id, request_check_url}` and the result is
   fetched by polling that URL. A page is therefore not one request but a submit plus an
   unbounded-ish poll loop, and "still processing" has to be distinguished from "failed" — the
   API signals a soft failure with HTTP 200 and `success: false`, which is exactly the shape that
   gets mistaken for a result. `--poll-timeout` bounds the wait so a wedged request becomes a
   durable error rather than hanging the run.

2. **The rate limits are low and tiered.** Free tier is 10 requests/minute with 5 concurrent;
   Team is 200/minute with 400 concurrent. `--workers` defaults to 4 to sit inside the free tier's
   concurrency, and there is a client-side request-per-minute throttle, because on this API a 429
   is easy to earn and the polling traffic counts too. Raise both deliberately if your account
   allows it — a run that trips the limit repeatedly is slower than one that never does.

The corpus is WebP and `/convert` accepts WebP directly, so pages are uploaded as-is with no
re-encoding — nothing is resampled between the ground truth's pixels and the engine's input.

Provenance note: the response carries a `versions` object identifying the model builds that served
the request. That is recorded per page, because the whole reason a hosted row is weaker evidence
than a self-hosted one is that you usually cannot say what actually served you (see RESULTS.md,
"What counts as a score"). Where the API does tell us, we keep it.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import pathlib
import subprocess
import sys
import threading
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import common  # noqa: E402

MODEL = "datalab"
DEFAULT_DATASET = "finebooks/bhl-impact-gt"
API_URL = "https://www.datalab.to/api/v1/convert"
DEFAULT_KEY_FILE = pathlib.Path.home() / ".config" / "datalab" / "api_key"

_SETTINGS: dict = {}


class RateLimiter:
    """Client-side requests-per-minute cap, shared across worker threads.

    The API's own 429 is recoverable, but on a low free-tier limit it is cheaper to not earn it:
    every 429 costs a retry slot AND the backoff wall-clock, and polling requests count toward the
    same budget as submissions.
    """

    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self._times: collections.deque = collections.deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        if self.per_minute <= 0:
            return
        while True:
            with self._lock:
                now = time.monotonic()
                while self._times and now - self._times[0] >= 60.0:
                    self._times.popleft()
                if len(self._times) < self.per_minute:
                    self._times.append(now)
                    return
                wait = 60.0 - (now - self._times[0]) + 0.01
            time.sleep(max(wait, 0.01))


def _harness_pin() -> str | None:
    harness = pathlib.Path(__file__).resolve().parent.parent / "harness"
    try:
        return subprocess.run(["git", "-C", str(harness), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _read_api_key(explicit: pathlib.Path | None) -> str:
    key = os.environ.get("DATALAB_API_KEY", "").strip()
    if key:
        return key
    for candidate in (explicit, DEFAULT_KEY_FILE):
        if candidate and pathlib.Path(candidate).is_file():
            key = pathlib.Path(candidate).read_text(encoding="utf-8").strip()
            if key:
                return key
    raise SystemExit(
        "No Datalab API key found. Either:\n"
        "  export DATALAB_API_KEY=...\n"
        f"or write the key to {DEFAULT_KEY_FILE} (mkdir -p its parent first), "
        "or pass --api-key-file."
    )


# The API validates the upload's declared MIME type, not just the filename: posting WebP bytes as
# `application/octet-stream` is rejected with "Invalid file type" even though WebP is supported.
# Sniff the format and declare both the extension and the type it actually is.
_FORMATS = (
    (lambda d: d[:4] == b"RIFF" and d[8:12] == b"WEBP", ".webp", "image/webp"),
    (lambda d: d[:8] == b"\x89PNG\r\n\x1a\n", ".png", "image/png"),
    (lambda d: d[:3] == b"\xff\xd8\xff", ".jpg", "image/jpeg"),
    (lambda d: d[:4] in (b"II*\x00", b"MM\x00*"), ".tiff", "image/tiff"),
    (lambda d: d[:6] in (b"GIF87a", b"GIF89a"), ".gif", "image/gif"),
)


def _upload_type(data: bytes) -> tuple[str, str]:
    """Return (filename suffix, MIME type) for the uploaded bytes."""
    for matches, suffix, mime in _FORMATS:
        if matches(data):
            return suffix, mime
    raise ValueError(f"unrecognised image format (magic {data[:8]!r})")


def _suffix(data: bytes) -> str:
    return _upload_type(data)[0]


def build_form(mode: str, *, image_captions: bool, keep_furniture: bool) -> dict:
    """The multipart form fields, including two options that are not in the public docs.

    `disable_image_captions` stops the engine writing an AI-generated *description* of each
    figure into the markdown as alt text. Those descriptions are a product feature, not a
    transcription of ink on the page, and on this corpus they were 4.5% of all output and worth
    ~0.055 CER — so for a transcription benchmark they are noise, and captions default to OFF here.

    `keep_pageheader_in_output` / `keep_pagefooter_in_output` (nested under `additional_config`)
    control whether running heads and page numbers — the GT's FURNITURE layer — are emitted at
    all. Per DESIGN.md this is *policy, not quality*: the body-only headline neither rewards nor
    penalises it in principle, though in practice emitted furniture does land as zero-reference
    insertions against a body target. Keeping it is the default here so this row sits on the same
    side of that axis as the verbatim Mistral rows rather than being compared across policies.
    """
    additional = {
        "keep_pageheader_in_output": bool(keep_furniture),
        "keep_pagefooter_in_output": bool(keep_furniture),
    }
    return {
        "mode": mode,
        "output_format": "markdown",
        "paginate": "false",
        "disable_image_captions": "false" if image_captions else "true",
        "additional_config": json.dumps(additional),
    }


def _init_worker(api_key: str, form: dict, timeout: float, max_attempts: int, url: str,
                 poll_timeout: float, poll_interval: float, limiter: RateLimiter) -> None:
    import requests

    session = requests.Session()
    session.headers.update({"X-API-Key": api_key})
    _SETTINGS.update(session=session, form=form, mode=form.get("mode"), timeout=timeout,
                     max_attempts=max_attempts, url=url, poll_timeout=poll_timeout,
                     poll_interval=poll_interval, limiter=limiter)


def _request(method: str, url: str, **kwargs):
    """One rate-limited HTTP call with retry on 429/5xx/transport errors."""
    import requests

    session, limiter = _SETTINGS["session"], _SETTINGS["limiter"]
    max_attempts, timeout = _SETTINGS["max_attempts"], _SETTINGS["timeout"]
    last = None
    for attempt in range(1, max_attempts + 1):
        limiter.acquire()
        try:
            response = session.request(method, url, timeout=timeout, **kwargs)
            if response.status_code == 200:
                return response
            if response.status_code != 429 and response.status_code < 500:
                raise ValueError(f"HTTP {response.status_code}: {response.text[:300]}")
            last = f"HTTP {response.status_code}: {response.text[:200]}"
            delay = float(response.headers.get("Retry-After") or 0) or min(2 ** attempt, 30)
        except (requests.Timeout, requests.ConnectionError) as exc:
            last = f"{type(exc).__name__}: {exc}"
            delay = min(2 ** attempt, 30)
        if attempt == max_attempts:
            break
        time.sleep(delay)
    raise RuntimeError(f"giving up after {max_attempts} attempts; last failure: {last}")


def _extract_markdown(payload: dict) -> str:
    """Pull the markdown out of a completed result, failing closed on an unexpected shape."""
    for key in ("markdown", "text", "output"):
        value = payload.get(key)
        if isinstance(value, str):
            return value
    raise ValueError(
        f"completed result has no markdown/text field; keys={sorted(payload)[:12]}"
    )


def ocr_page(page: dict) -> tuple[str, dict]:
    """Submit one page, poll until the result is ready, return its markdown."""
    data = page["image_bytes"]
    suffix, mime = _upload_type(data)
    files = {"file": (f"page{suffix}", data, mime)}
    form = dict(_SETTINGS["form"])

    submit = _request("POST", _SETTINGS["url"], files=files, data=form).json()
    if not submit.get("success", True):
        raise RuntimeError(f"submit rejected: {submit.get('error')!r}")
    check_url = submit.get("request_check_url")
    if not check_url:
        raise ValueError(f"submit response has no request_check_url: {json.dumps(submit)[:300]}")

    deadline = time.monotonic() + _SETTINGS["poll_timeout"]
    polls = 0
    while True:
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"result not ready after {_SETTINGS['poll_timeout']:.0f}s "
                f"({polls} polls); request_id={submit.get('request_id')}"
            )
        time.sleep(_SETTINGS["poll_interval"])
        polls += 1
        body = _request("GET", check_url).json()
        status = str(body.get("status", "")).lower()
        if status in ("processing", "pending", "queued", "running", ""):
            if body.get("success") is False:
                # A soft failure: HTTP 200 carrying success=false. This is the shape most likely
                # to be mistaken for a result, so it is raised rather than returned.
                raise RuntimeError(f"request failed: {body.get('error')!r}")
            if status:
                continue
        if body.get("success") is False:
            raise RuntimeError(f"request failed: {body.get('error')!r}")
        if status in ("complete", "completed", "success", "done") or "markdown" in body:
            text = _extract_markdown(body)
            return text, {
                "engine": "datalab",
                "mode": _SETTINGS["mode"],
                "request_id": submit.get("request_id"),
                "versions": body.get("versions") or submit.get("versions"),
                "page_count": body.get("page_count"),
                "polls": polls,
            }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument("--id-column", default="PageID")
    ap.add_argument("--image-column", default="image")
    ap.add_argument("--mode", default="balanced", choices=("fast", "balanced", "accurate"))
    ap.add_argument("--image-captions", action="store_true", default=False,
                    help="let the engine write AI-generated figure descriptions into the markdown. "
                         "OFF by default: they are not transcription, and on this corpus they were "
                         "4.5%% of output and worth ~0.055 CER.")
    ap.add_argument("--drop-furniture", dest="keep_furniture", action="store_false", default=True,
                    help="let the engine omit running heads and page numbers. Kept by default so "
                         "this row sits on the same furniture policy as the verbatim rows.")
    ap.add_argument("--model-label", default=None,
                    help="board row name; defaults to datalab-<mode>")
    ap.add_argument("--url", default=API_URL)
    ap.add_argument("--api-key-file", type=pathlib.Path, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--workers", type=int, default=4,
                    help="concurrent pages; free tier allows 5 concurrent requests")
    ap.add_argument("--rpm", type=int, default=9,
                    help="client-side requests/minute cap across all threads; free tier is 10, "
                         "and POLLS COUNT TOO. 0 disables the throttle.")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--poll-timeout", type=float, default=600.0)
    ap.add_argument("--poll-interval", type=float, default=2.0)
    ap.add_argument("--max-attempts", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None,
                    help="smoke-test against N pages. NOT scoreable — the page set will not match.")
    ap.add_argument("--no-retry-errors", dest="retry_errors", action="store_false", default=True)
    args = ap.parse_args()

    # The config is part of the row's identity: a captions-on run and a captions-off run are not
    # the same measurement, and silently sharing a label would let one overwrite the other's
    # checkpoint and produce a table mixing both.
    suffixes = "".join(("-captions" if args.image_captions else "",
                        "" if args.keep_furniture else "-nofurniture"))
    label = args.model_label or f"datalab-{args.mode}{suffixes}"
    form = build_form(args.mode, image_captions=args.image_captions,
                      keep_furniture=args.keep_furniture)
    out_path = pathlib.Path(args.out or f"runs/{label}/run.parquet")
    checkpoint_path = pathlib.Path(args.checkpoint) if args.checkpoint \
        else out_path.parent / "checkpoint.jsonl"

    api_key = _read_api_key(args.api_key_file)
    revision = common.resolve_revision(args.dataset, args.revision)

    index = common.load_index(args.dataset, revision=revision, split=args.split,
                              image_column=args.image_column, id_column=args.id_column)
    if args.limit:
        index = index[:args.limit]
        print(f"WARNING: --limit {args.limit} — smoke test only, not a scoreable run", flush=True)
    wanted = {row["PageID"] for row in index}

    already = len(common.Checkpoint(checkpoint_path).load())
    remaining = len(index) - already
    if args.rpm > 0 and remaining > 0:
        # Every page needs 1 submit + at least 1 poll, so the floor is ~2 requests per page.
        print(f"throttle: {args.rpm} req/min · ~{remaining * 2 / max(args.rpm, 1) / 60:.1f}h "
              f"lower bound for {remaining} remaining pages (submits + one poll each)", flush=True)
    print(f"datalab mode={args.mode} label={label} | dataset {args.dataset} @ {revision} | "
          f"{len(index)} pages | {already} already cached | {args.workers} workers", flush=True)

    def streamed():
        for page in common.load_pages(args.dataset, revision=revision, split=args.split,
                                      image_column=args.image_column, id_column=args.id_column):
            if page["PageID"] in wanted:
                yield page

    started = time.monotonic()
    limiter = RateLimiter(args.rpm)
    checkpoint = common.Checkpoint(checkpoint_path)
    common.run_pages(
        streamed(), ocr_page, checkpoint=checkpoint, workers=args.workers, executor="thread",
        initializer=_init_worker,
        initargs=(api_key, form, args.timeout, args.max_attempts, args.url,
                  args.poll_timeout, args.poll_interval, limiter),
        retry_errors=args.retry_errors, progress_every=25,
    )
    common.finalize(checkpoint, [row["PageID"] for row in index], model=label, out_path=out_path)

    # Roll the per-page `versions` blobs up so the provenance says what actually served the run.
    served = collections.Counter()
    for record in common.Checkpoint(checkpoint_path).load().values():
        versions = (record.get("meta") or {}).get("versions")
        if versions:
            served[json.dumps(versions, sort_keys=True)] += 1

    meta_path = out_path.parent / "producer-run.json"
    meta_path.write_text(json.dumps({
        "prompt": None,
        "note": "model-native OCR mode",
        "producer": "bhl-ocr-bench producers/datalab.py",
        "model": label,
        "endpoint": args.url,
        "endpoint_note": "bespoke multipart /api/v1/convert with asynchronous polling; NOT "
                         "OpenAI-chat-compatible, so the harness run_openai.py cannot drive it",
        "request_settings": {**form, "workers": args.workers, "rpm_cap": args.rpm,
                             "timeout_s": args.timeout, "poll_timeout_s": args.poll_timeout,
                             "max_attempts": args.max_attempts},
        "request_settings_note": "disable_image_captions and additional_config."
                                 "keep_page{header,footer}_in_output are undocumented form fields; "
                                 "captions are figure descriptions rather than transcription, and "
                                 "the furniture flags set this row's policy to match the verbatim "
                                 "rows (see DESIGN.md, 'The furniture trade-off')",
        "served_versions": {json.loads(k): v for k, v in served.items()} if served else None,
        "provenance_caveat": "Hosted API row. RESULTS.md 'What counts as a score' requires "
                             "self-run inference under a pinned image and model revision; a hosted "
                             "endpoint cannot attest serving configuration or quantization. Any "
                             "`served_versions` above is what the API volunteered, not a pin.",
        "dataset": args.dataset,
        "dataset_resolved_revision": revision,
        "postproc_version": "0",
        "postprocessing": "none — raw markdown as returned",
        "wall_clock_s": round(time.monotonic() - started, 1),
        "harness_pin": _harness_pin(),
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"run provenance -> {meta_path}")


if __name__ == "__main__":
    main()
