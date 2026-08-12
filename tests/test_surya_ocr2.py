"""Tests for the surya-ocr-2 adapter.

Two things here are worth more than the rest and most of this file exists to pin them down:

1. **HTML must not reach the scorer.** Surya returns per-block HTML, `surya-ocr-2` is not in the
   harness `normalize_outputs.py` REGISTRY, and the diplomatic scoring lane does not strip markup.
   So if the producer stops flattening, nothing downstream catches it and the row silently scores
   its own serialization instead of the model.

2. **A bad page must not poison its batch.** Batching is a throughput trick; if one page raises and
   takes 23 healthy pages with it, those pages fail again on every resume and the run can never
   complete.

Nothing here imports surya or torch: the producer defers those to inside the functions that need
them, so the pure logic is testable from the project venv.
"""
from __future__ import annotations

import io
import json
import types

import pytest
from PIL import Image

import surya_ocr2 as S


def block(html, *, order, label="Text", skipped=False, error=False):
    return types.SimpleNamespace(html=html, reading_order=order, label=label,
                                 skipped=skipped, error=error)


def page(*blocks):
    return types.SimpleNamespace(blocks=list(blocks), image_bbox=[0.0, 0.0, 100.0, 100.0])


def png_bytes():
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), "white").save(buffer, format="PNG")
    return buffer.getvalue()


# --------------------------------------------------------------------------- serialization


def test_html_to_text_strips_tags_and_unescapes_entities():
    """The whole point of the flattening step. `&amp;` must come back as `&`, or the model is
    charged for characters it read correctly."""
    assert S.html_to_text("<p>Piscium <b>Querel&#230;</b> &amp; Vindici&#230;.</p>") == \
        "Piscium Querelæ & Vindiciæ."


def test_html_to_text_leaves_no_angle_brackets():
    text = S.html_to_text("<h1>BULIMUS.</h1><p><i>Testa ovata</i>, vel oblonga</p>")
    assert "<" not in text and ">" not in text
    assert "Testa ovata" in text and "BULIMUS." in text


def test_serialize_page_emits_reading_order_not_list_order():
    """Blocks arrive in arbitrary order; reading_order is the document order."""
    text, _ = S.serialize_page(page(block("<p>third</p>", order=2),
                                    block("<p>first</p>", order=0),
                                    block("<p>second</p>", order=1)))
    assert text == "first\nsecond\nthird"


def test_serialize_page_drops_skipped_blocks():
    """Surya marks Figure/Image/Diagram/Blank-Page as skipped — regions it decided carry no text."""
    text, meta = S.serialize_page(page(
        block("<p>caption</p>", order=1, label="Caption"),
        block("", order=0, label="Picture", skipped=True),
    ))
    assert text == "caption"
    assert meta["blocks"] == 2
    assert meta["blocks_skipped"] == 1
    assert meta["labels"] == ["Picture", "Caption"]


def test_serialize_page_drops_empty_blocks_without_counting_them_as_text():
    text, meta = S.serialize_page(page(block("", order=0), block("<p>real</p>", order=1)))
    assert text == "real"
    assert meta["blocks_skipped"] == 1


def test_serialize_page_of_an_all_picture_page_is_empty_not_whitespace():
    """A blank page must be exactly "" — `finalize` and the blank-page count both test falsiness,
    and a stray newline would read as "the model produced text here"."""
    text, meta = S.serialize_page(page(block("", order=0, label="Picture", skipped=True)))
    assert text == ""
    assert meta["blocks_skipped"] == 1


def test_serialize_page_records_block_errors():
    _, meta = S.serialize_page(page(block("<p>ok</p>", order=0),
                                    block("<p>bad</p>", order=1, error=True)))
    assert meta["blocks_errored"] == 1


def test_serialize_page_meta_is_json_serializable():
    """`finalize` json.dumps this into producer_meta; a non-serializable value fails the whole run
    at the very end, after hours of inference."""
    _, meta = S.serialize_page(page(block("<p>x</p>", order=0)))
    assert json.loads(json.dumps(meta))["engine"] == "surya-ocr-2"


# --------------------------------------------------------------------------- the batched loop


class FakePredictor:
    """Records the batch sizes it was called with; optionally raises on one page's image."""

    def __init__(self, poison=None):
        self.calls = []
        self.poison = poison

    def __call__(self, images, full_page=True):
        self.calls.append(len(images))
        if self.poison is not None and len(images) == self.poison:
            raise RuntimeError("llama.cpp fell over")
        return [page(block(f"<p>page {i}</p>", order=0)) for i in range(len(images))]


def pages(n):
    data = png_bytes()
    return [{"PageID": 1000 + i, "image_bytes": data} for i in range(n)]


def test_run_batches_groups_pages_and_checkpoints_every_one(tmp_path):
    checkpoint = S.common.Checkpoint(tmp_path / "checkpoint.jsonl")
    predictor = FakePredictor()
    completed, errors = S.run_batches(pages(7), predictor, checkpoint=checkpoint, batch_size=3)

    assert (completed, errors) == (7, 0)
    assert predictor.calls == [3, 3, 1]          # trailing partial batch is flushed
    done = S.common.Checkpoint(tmp_path / "checkpoint.jsonl").load()
    assert len(done) == 7
    assert all(record["error"] is None for record in done.values())


def test_a_failing_batch_falls_back_to_one_page_at_a_time(tmp_path):
    """The load-bearing one: without this, a single poison page fails its whole batch on every
    resume and the run can never reach a complete page set."""
    checkpoint = S.common.Checkpoint(tmp_path / "checkpoint.jsonl")
    predictor = FakePredictor(poison=4)          # any 4-image call raises; singles succeed
    completed, errors = S.run_batches(pages(4), predictor, checkpoint=checkpoint, batch_size=4)

    assert (completed, errors) == (4, 0)
    assert predictor.calls == [4, 1, 1, 1, 1]    # failed batch, then each page alone
    done = S.common.Checkpoint(tmp_path / "checkpoint.jsonl").load()
    assert len(done) == 4 and all(r["error"] is None for r in done.values())


def test_a_page_that_fails_alone_becomes_a_durable_error_row(tmp_path):
    class AlwaysFails:
        def __call__(self, images, full_page=True):
            raise RuntimeError("boom")

    checkpoint = S.common.Checkpoint(tmp_path / "checkpoint.jsonl")
    completed, errors = S.run_batches(pages(2), AlwaysFails(), checkpoint=checkpoint, batch_size=2)

    assert (completed, errors) == (2, 2)
    done = S.common.Checkpoint(tmp_path / "checkpoint.jsonl").load()
    assert all("boom" in record["error"] for record in done.values())
    assert all(record["markdown"] is None for record in done.values())


def test_error_rows_carry_the_sentinel_into_the_text_column(tmp_path):
    """`score_dataset` detects a failed page only by the text column's sentinel, never by a
    sibling error column — see common.py. This is the end-to-end proof for this producer."""
    class AlwaysFails:
        def __call__(self, images, full_page=True):
            raise RuntimeError("boom")

    checkpoint = S.common.Checkpoint(tmp_path / "checkpoint.jsonl")
    S.run_batches(pages(1), AlwaysFails(), checkpoint=checkpoint, batch_size=1)
    frame = S.common.finalize(S.common.Checkpoint(tmp_path / "checkpoint.jsonl"), [1000],
                              model="surya-ocr-2", out_path=tmp_path / "run.parquet")
    assert frame.loc[0, "markdown"].startswith(S.common.ERROR_PREFIX)


def test_resume_skips_completed_pages_and_retries_errored_ones(tmp_path):
    path = tmp_path / "checkpoint.jsonl"
    seed = S.common.Checkpoint(path)
    seed.append({"PageID": 1000, "markdown": "already done", "error": None, "meta": {}})
    seed.append({"PageID": 1001, "markdown": None, "error": "RuntimeError: boom", "meta": {}})
    seed.close()

    predictor = FakePredictor()
    completed, _ = S.run_batches(pages(3), predictor, checkpoint=S.common.Checkpoint(path),
                                 batch_size=8)

    assert completed == 2                        # 1000 skipped; 1001 retried, 1002 fresh
    done = S.common.Checkpoint(path).load()
    assert done[1000]["markdown"] == "already done"
    assert done[1001]["error"] is None           # the retry superseded the error record


def test_no_retry_errors_leaves_the_error_row_alone(tmp_path):
    path = tmp_path / "checkpoint.jsonl"
    seed = S.common.Checkpoint(path)
    seed.append({"PageID": 1000, "markdown": None, "error": "RuntimeError: boom", "meta": {}})
    seed.close()

    completed, _ = S.run_batches(pages(1), FakePredictor(), checkpoint=S.common.Checkpoint(path),
                                 batch_size=4, retry_errors=False)
    assert completed == 0
    assert S.common.Checkpoint(path).load()[1000]["error"] == "RuntimeError: boom"


def test_a_short_result_list_is_an_error_not_a_silent_misalignment(tmp_path):
    """Zipping N images against fewer results would attach page A's text to page B's id — a
    corrupt run that still scores. It must fail instead."""
    class DropsOne:
        def __call__(self, images, full_page=True):
            return [page(block("<p>x</p>", order=0))] * (len(images) - 1)

    checkpoint = S.common.Checkpoint(tmp_path / "checkpoint.jsonl")
    completed, errors = S.run_batches(pages(2), DropsOne(), checkpoint=checkpoint, batch_size=2)
    assert (completed, errors) == (2, 2)


# --------------------------------------------------------------------------- provenance


@pytest.mark.parametrize("code,name", [(1, "MOSTLY_F16"), (7, "MOSTLY_Q8_0"), (14, "MOSTLY_Q4_K")])
def test_gguf_file_type_names_are_known(code, name):
    """The precision claim in the provenance is read from the GGUF header, so the enum has to be
    right — reporting a Q4 build as F16 would overstate what this row is."""
    assert S.GGUF_FILE_TYPES[code] == name


def test_unknown_gguf_file_type_is_reported_not_guessed():
    assert S.GGUF_FILE_TYPES.get(999, "unknown(999)") == "unknown(999)"
