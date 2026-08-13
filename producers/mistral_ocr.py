"""Mistral OCR producer — the first hosted, paid engine.

    export MISTRAL_API_KEY=...
    uv run producers/mistral_ocr.py --limit 5 --out runs/mistral-smoke/run.parquet   # cents
    uv run producers/mistral_ocr.py --out runs/mistral-ocr/run.parquet               # the real run

Mistral's OCR endpoint is **not** OpenAI-chat-compatible, so the harness's `run_openai.py` cannot
drive it. `/v1/ocr` is a bespoke endpoint: you POST a document (here a base64 data URI of one page
image) and get back structured JSON with a `pages[]` array, each carrying `markdown`. There is no
prompt surface at all, which is exactly why the run provenance says
`{"prompt": null, "note": "model-native OCR mode"}` — that is the literal truth about this engine,
not a placeholder.

Three things this adapter is careful about, all of them because the run costs money and scoring
fails closed on a single bad page:

1. **Retries happen in-call, not just on resume.** A 429 or a 5xx is transient; letting it fall
   through to a durable error row would mean re-running the page later at full price, and a run
   with one unretried error is ineligible regardless of how good the other 2,164 pages are.
   Retry-After is honoured when the server sends it.
2. **An empty read is a result, not an error.** 428 of these pages are sparse or blank. A blank
   page legitimately OCRs to "", and the harness accepts "" as a deliberate empty output. Raising
   on empty (as several of the harness's own model registries do) would turn the most interesting
   pages in the corpus into producer errors and sink the whole run.
3. **The model id is verified, not assumed.** `--model` defaults to the `mistral-ocr-latest` alias;
   the producer resolves it against `/v1/models` at startup and records the concrete version the
   API reports, so the board row says which model actually ran rather than which alias was typed.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import pathlib
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import common  # noqa: E402

MODEL = "mistral-ocr"
DEFAULT_DATASET = "finebooks/bhl-impact-gt"
API_BASE = "https://api.mistral.ai/v1"

# Set once per worker thread by the initializer; threads share the process, so a module global is
# the simplest way to hand the session and settings to `ocr_page` without pickling anything.
_SETTINGS: dict = {}


def _harness_pin() -> str | None:
    harness = pathlib.Path(__file__).resolve().parent.parent / "harness"
    try:
        return subprocess.run(["git", "-C", str(harness), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


DEFAULT_KEY_FILE = pathlib.Path.home() / ".config" / "mistral" / "api_key"


def _read_api_key(explicit: pathlib.Path | None) -> str:
    """Take the key from $MISTRAL_API_KEY, else a key file. Never log or persist it.

    The key file exists because a key exported in one interactive shell is not visible to a
    separately-launched run; a file is the reliable channel. It is read, stripped, and used —
    it never reaches the checkpoint, the provenance, or stdout.
    """
    key = os.environ.get("MISTRAL_API_KEY", "").strip()
    if key:
        return key
    for candidate in (explicit, DEFAULT_KEY_FILE):
        if candidate and pathlib.Path(candidate).is_file():
            key = pathlib.Path(candidate).read_text(encoding="utf-8").strip()
            if key:
                return key
    raise SystemExit(
        "No Mistral API key found. Either:\n"
        "  export MISTRAL_API_KEY=...\n"
        f"or write the key to {DEFAULT_KEY_FILE} (mkdir -p its parent first), "
        "or pass --api-key-file."
    )


def _mime(data: bytes) -> str:
    """Sniff the image type from magic bytes — the corpus is webp, but do not assume it."""
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:4] == b"II*\x00" or data[:4] == b"MM\x00*":
        return "image/tiff"
    raise ValueError(f"unrecognised image format (magic {data[:8]!r})")


def _init_worker(api_key: str, model: str, timeout: float, max_attempts: int,
                 base_url: str) -> None:
    import requests

    session = requests.Session()
    session.headers.update({
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    })
    _SETTINGS.update(session=session, model=model, timeout=timeout,
                     max_attempts=max_attempts, base_url=base_url)


def ocr_page(page: dict) -> tuple[str, dict]:
    """OCR one page through /v1/ocr, retrying transient failures in-call."""
    import requests

    session = _SETTINGS["session"]
    model = _SETTINGS["model"]
    timeout = _SETTINGS["timeout"]
    max_attempts = _SETTINGS["max_attempts"]
    url = f"{_SETTINGS['base_url']}/ocr"

    data = page["image_bytes"]
    payload = {
        "model": model,
        "document": {
            "type": "image_url",
            "image_url": f"data:{_mime(data)};base64,{base64.b64encode(data).decode('ascii')}",
        },
        # We score text, never the cropped illustrations; asking for them would multiply the
        # response size for nothing.
        "include_image_base64": False,
    }

    last = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = session.post(url, json=payload, timeout=timeout)
            if response.status_code == 200:
                body = response.json()
                pages = body.get("pages")
                if not isinstance(pages, list) or not pages:
                    raise ValueError(f"response has no pages[]: {json.dumps(body)[:300]}")
                chunks = []
                for item in pages:
                    text = item.get("markdown")
                    if text is None:
                        raise ValueError(f"page object has no markdown key: {json.dumps(item)[:200]}")
                    if not isinstance(text, str):
                        raise ValueError(f"markdown is {type(text).__name__}, expected str")
                    chunks.append(text)
                # A single image is one page; join defensively in case the API ever splits.
                usage = body.get("usage_info") or {}
                return "\n\n".join(chunks), {
                    "engine": "mistral-ocr",
                    "model": body.get("model", model),
                    "api_pages": len(pages),
                    "pages_processed": usage.get("pages_processed"),
                    "attempts": attempt,
                }

            # 429 and 5xx are transient; 4xx other than 429 is a request problem that will not
            # get better by repeating it, so fail fast rather than burning the rate limit.
            if response.status_code != 429 and response.status_code < 500:
                raise ValueError(
                    f"HTTP {response.status_code}: {response.text[:300]}"
                )
            last = f"HTTP {response.status_code}: {response.text[:200]}"
            delay = float(response.headers.get("Retry-After") or 0) or min(2 ** attempt, 30)
        except (requests.Timeout, requests.ConnectionError) as exc:
            last = f"{type(exc).__name__}: {exc}"
            delay = min(2 ** attempt, 30)

        if attempt == max_attempts:
            break
        time.sleep(delay)

    raise RuntimeError(f"giving up after {max_attempts} attempts; last failure: {last}")


def resolve_model(api_key: str, requested: str, base_url: str) -> tuple[str, list[str]]:
    """Check the requested model exists before spending anything, and list the OCR models."""
    import requests

    response = requests.get(f"{base_url}/models", timeout=30,
                            headers={"Authorization": f"Bearer {api_key}"})
    if response.status_code != 200:
        raise SystemExit(f"cannot list models (HTTP {response.status_code}): {response.text[:300]}")
    ids = [m.get("id") for m in response.json().get("data", []) if m.get("id")]
    ocr_models = sorted(i for i in ids if "ocr" in i.lower())
    if requested not in ids:
        raise SystemExit(
            f"model {requested!r} is not available to this account.\n"
            f"OCR models visible here: {ocr_models or '(none)'}\n"
            f"Pass --model with one of those."
        )
    return requested, ocr_models


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument("--id-column", default="PageID")
    ap.add_argument("--image-column", default="image")
    ap.add_argument("--model", default="mistral-ocr-latest",
                    help="verified against /v1/models before the run starts")
    ap.add_argument("--model-label", default=MODEL,
                    help="the name this row carries on the board")
    ap.add_argument("--base-url", default=API_BASE)
    ap.add_argument("--api-key-file", type=pathlib.Path, default=None,
                    help=f"read the key from a file instead of $MISTRAL_API_KEY "
                         f"(default fallback: {DEFAULT_KEY_FILE})")
    ap.add_argument("--out", default="runs/mistral-ocr/run.parquet")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--workers", type=int, default=8,
                    help="concurrent requests; raise cautiously, 429s cost wall-clock in backoff")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--max-attempts", type=int, default=5,
                    help="in-call retries per page before it becomes a durable error row")
    ap.add_argument("--limit", type=int, default=None,
                    help="smoke-test against N pages. NOT scoreable — the page set will not match.")
    ap.add_argument("--no-retry-errors", dest="retry_errors", action="store_false", default=True)
    args = ap.parse_args()

    api_key = _read_api_key(args.api_key_file)

    out_path = pathlib.Path(args.out)
    checkpoint_path = pathlib.Path(args.checkpoint) if args.checkpoint \
        else out_path.parent / "checkpoint.jsonl"

    model, ocr_models = resolve_model(api_key, args.model, args.base_url)
    revision = common.resolve_revision(args.dataset, args.revision)

    index = common.load_index(args.dataset, revision=revision, split=args.split,
                              image_column=args.image_column, id_column=args.id_column)
    if args.limit:
        index = index[:args.limit]
        print(f"WARNING: --limit {args.limit} — smoke test only, not a scoreable run", flush=True)
    wanted = {row["PageID"] for row in index}

    already = len(common.Checkpoint(checkpoint_path).load())
    print(f"model {model} (OCR models available: {ocr_models}) | dataset {args.dataset} @ "
          f"{revision} | {len(index)} pages | {already} already cached | {args.workers} workers",
          flush=True)

    def streamed():
        for page in common.load_pages(args.dataset, revision=revision, split=args.split,
                                      image_column=args.image_column, id_column=args.id_column):
            if page["PageID"] in wanted:
                yield page

    clock = common.WallClock(out_path.parent / "passes.jsonl")
    checkpoint = common.Checkpoint(checkpoint_path)
    with clock.pass_():
        common.run_pages(
            streamed(), ocr_page, checkpoint=checkpoint, workers=args.workers, executor="thread",
            initializer=_init_worker,
            initargs=(api_key, model, args.timeout, args.max_attempts, args.base_url),
            retry_errors=args.retry_errors,
        )
    common.finalize(checkpoint, [row["PageID"] for row in index], model=args.model_label,
                    out_path=out_path)

    meta_path = out_path.parent / "producer-run.json"
    meta_path.write_text(json.dumps({
        "prompt": None,
        "note": "model-native OCR mode",
        "producer": "bhl-ocr-bench producers/mistral_ocr.py",
        "model": args.model_label,
        "api_model_requested": args.model,
        "endpoint": f"{args.base_url}/ocr",
        "endpoint_note": "bespoke /v1/ocr returning structured JSON; NOT OpenAI-chat-compatible, "
                         "so the harness run_openai.py cannot drive it",
        "request_settings": {"include_image_base64": False, "timeout_s": args.timeout,
                             "max_attempts": args.max_attempts, "workers": args.workers},
        "dataset": args.dataset,
        "dataset_resolved_revision": revision,
        "postproc_version": "0",
        "postprocessing": "none — raw pages[].markdown as returned",
        "wall_clock_s": clock.total_s,
        "wall_clock_passes": clock.load(),
        "wall_clock_note": "cumulative across every pass of this run, not just the last "
                           "invocation; null means one pass predates per-pass timing and the "
                           "total is not recoverable",
        "harness_pin": _harness_pin(),
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"run provenance -> {meta_path}")


if __name__ == "__main__":
    main()
