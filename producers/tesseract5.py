"""tesseract-5 producer — the classical, CPU, deterministic baseline.

This is the acceptance test for the whole pipeline, not a result we need. The published board's
`tesseract-5` row reads CER 0.0642; if this adapter plus the local benchmark build plus the pinned
scorer do not land near that, the plumbing is wrong and no paid API run should follow.

    uv run producers/tesseract5.py --out runs/tesseract-5/run.parquet

THE LANGUAGE MAP IS THE RESULT. Tesseract scores ~0.19 CER on this corpus with the wrong language
pack and ~0.064 with the right per-volume ones — the difference between last place and mid-table.
The map below is reproduced from the pinned harness's `drivers/tesseract-port.py`, which derived it
from the ground truth itself rather than from catalogue metadata (an earlier pass trusted the
catalogue and ran Cyrillic over a German/French volume). It is keyed on `BarCode`, which is what
`prep_sample.py` copies into the benchmark's `volume` column.

Because a missing language pack silently triples the error rate into a number that still looks
plausible, `--strict-langs` (default on) aborts rather than falling back to English.

No post-processing is applied. `tesseract-5` is deliberately absent from the harness's
`normalize_outputs.py` REGISTRY, so the published row was scored on raw `image_to_string` output;
cleaning it up here would move the number away from the row we are trying to reproduce.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import pathlib
import platform
import shutil
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import common  # noqa: E402

MODEL = "tesseract-5"
DEFAULT_DATASET = "finebooks/bhl-impact-gt"

# Per-book language packs, keyed on BarCode. `trudy` is genuinely mixed German/French, hence '+'.
TESS_LANGS = {
    "birdsofgreatbrit02butl": "eng",
    "conchologiaiconi05reev": "eng",
    "daschitinskelett00prel": "deu",
    "histoirenaturell10cuvi": "fra",
    "pisciumquerelaee00sche": "lat",
    "trudyrusskagoent161881russ": "deu+fra",
}


def _harness_pin() -> str | None:
    """Record which harness commit scored this run — the submodule SHA, read from git."""
    harness = pathlib.Path(__file__).resolve().parent.parent / "harness"
    try:
        return subprocess.run(["git", "-C", str(harness), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _worker_init(omp_threads: int) -> None:
    """Pin Tesseract's OpenMP thread count inside each worker.

    We parallelise across pages, so letting each page fan out over every core just oversubscribes
    the machine. Pinning it to 1 also removes the only nondeterminism in the engine's scheduling,
    which matters for a baseline whose whole job is to be reproducible.
    """
    os.environ["OMP_THREAD_LIMIT"] = str(omp_threads)


def ocr_page(page: dict) -> tuple[str, dict]:
    """OCR one page. Module-level so the process pool can pickle it."""
    import pytesseract
    from PIL import Image

    lang = page["lang"]
    with Image.open(io.BytesIO(page["image_bytes"])) as image:
        text = pytesseract.image_to_string(image.convert("RGB"), lang=lang)
    return text, {"lang": lang, "engine": "tesseract"}


def check_environment(wanted_langs: set[str], *, strict: bool) -> tuple[str, dict]:
    """Verify the binary and every language pack BEFORE a two-hour run starts."""
    if shutil.which("tesseract") is None:
        raise SystemExit(
            "tesseract is not on PATH. Install it with:  brew install tesseract tesseract-lang"
        )
    import pytesseract

    version = str(pytesseract.get_tesseract_version())
    have = set(pytesseract.get_languages(config=""))
    missing = sorted(wanted_langs - have)
    if missing:
        message = (
            f"missing tesseract language pack(s): {missing} (have {len(have)} installed).\n"
            f"Install them with:  brew install tesseract-lang"
        )
        if strict:
            raise SystemExit(
                message + "\nFalling back to 'eng' would roughly triple this model's error rate, "
                "so this aborts rather than producing a plausible wrong number. "
                "Pass --loose-langs to override deliberately."
            )
        print(f"WARNING: {message}\n  falling back to 'eng' for those volumes", flush=True)
    return version, {"available": len(have), "missing": missing}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", default=DEFAULT_DATASET, help="benchmark dataset (images live here)")
    ap.add_argument("--revision", default=None, help="pin the dataset revision (resolved to a SHA)")
    ap.add_argument("--split", default="train")
    ap.add_argument("--id-column", default="PageID")
    ap.add_argument("--image-column", default="image")
    ap.add_argument("--volume-column", default="BarCode",
                    help="column carrying the volume identity the language map is keyed on")
    ap.add_argument("--out", default="runs/tesseract-5/run.parquet")
    ap.add_argument("--checkpoint", default=None, help="default: <out dir>/checkpoint.jsonl")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--omp-threads", type=int, default=1,
                    help="OMP_THREAD_LIMIT inside each worker; 1 keeps the run deterministic")
    ap.add_argument("--limit", type=int, default=None,
                    help="smoke-test against N pages. NOT scoreable: the page set will not match "
                         "the benchmark, and score_dataset fails closed on an incomplete set.")
    ap.add_argument("--loose-langs", dest="strict", action="store_false", default=True,
                    help="fall back to 'eng' when a pack is missing instead of aborting")
    ap.add_argument("--no-retry-errors", dest="retry_errors", action="store_false", default=True,
                    help="keep previously-errored pages as-is instead of retrying them on resume")
    args = ap.parse_args()

    out_path = pathlib.Path(args.out)
    checkpoint_path = pathlib.Path(args.checkpoint) if args.checkpoint \
        else out_path.parent / "checkpoint.jsonl"

    version, lang_info = check_environment(
        {pack for langs in TESS_LANGS.values() for pack in langs.split("+")}, strict=args.strict,
    )
    revision = common.resolve_revision(args.dataset, args.revision)
    print(f"tesseract {version} | dataset {args.dataset} @ {revision or 'local'} | "
          f"{args.workers} workers | OMP_THREAD_LIMIT={args.omp_threads}", flush=True)

    # Plan the run on the image-free index first: resolve every page's language pack and fail on
    # an unmapped volume BEFORE decoding anything. Images are then streamed one at a time, so a
    # 2,165-page corpus never sits in memory at once.
    index = common.load_index(
        args.dataset, revision=revision, split=args.split, image_column=args.image_column,
        id_column=args.id_column, extra_columns=(args.volume_column,),
    )
    if args.limit:
        index = index[:args.limit]
        print(f"WARNING: --limit {args.limit} — smoke test only, not a scoreable run", flush=True)

    unknown = sorted({row[args.volume_column] for row in index} - set(TESS_LANGS))
    if unknown:
        raise SystemExit(
            f"no language mapping for volume(s) {unknown}. The language map is the result here "
            f"(see this file's docstring) — add them to TESS_LANGS deliberately rather than "
            f"letting them default to English."
        )
    lang_by_page = {row["PageID"]: TESS_LANGS[row[args.volume_column]] for row in index}
    wanted = {row["PageID"] for row in index}

    counts: dict[str, int] = {}
    for lang in lang_by_page.values():
        counts[lang] = counts.get(lang, 0) + 1
    print(f"{len(index)} pages | languages: {counts}", flush=True)

    def streamed():
        for page in common.load_pages(
            args.dataset, revision=revision, split=args.split, image_column=args.image_column,
            id_column=args.id_column,
        ):
            if page["PageID"] not in wanted:
                continue
            yield {**page, "lang": lang_by_page[page["PageID"]]}

    checkpoint = common.Checkpoint(checkpoint_path)
    common.run_pages(
        streamed(), ocr_page, checkpoint=checkpoint, workers=args.workers, executor="process",
        initializer=_worker_init, initargs=(args.omp_threads,), retry_errors=args.retry_errors,
    )
    common.finalize(checkpoint, [row["PageID"] for row in index], model=MODEL, out_path=out_path)

    # Emit the run-provenance object the scorer wants, rather than hand-maintaining a file that
    # can drift from the run that actually happened. `prompt: null` is the literal, correct answer
    # for an OCR engine with no prompt surface.
    meta_path = out_path.parent / "producer-run.json"
    meta_path.write_text(json.dumps({
        "prompt": None,
        "note": "model-native OCR mode",
        "producer": "bhl-ocr-bench producers/tesseract5.py",
        "model": MODEL,
        "tesseract_version": version,
        "language_packs": lang_info,
        "language_map": TESS_LANGS,
        "dataset": args.dataset,
        "dataset_resolved_revision": revision,
        "omp_thread_limit": args.omp_threads,
        "workers": args.workers,
        "platform": f"{platform.system()} {platform.machine()} (homebrew tesseract, CPU)",
        "postproc_version": "0",
        "postprocessing": "none — tesseract-5 is absent from the harness normalize_outputs "
                          "REGISTRY, so the published row was scored on raw image_to_string output",
        "harness_pin": _harness_pin(),
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"run provenance -> {meta_path}")


if __name__ == "__main__":
    main()
