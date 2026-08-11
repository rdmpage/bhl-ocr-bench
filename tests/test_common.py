"""Tests for the producer plumbing.

These cover the two contracts that silently corrupt a run rather than crashing it: the error
sentinel landing in the text column, and checkpoint/resume being page-exact.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

import common


# --------------------------------------------------------------------------- checkpoint


def test_checkpoint_roundtrip(tmp_path):
    cp = common.Checkpoint(tmp_path / "cp.jsonl")
    cp.append({"PageID": 1, "markdown": "one", "error": None, "meta": {}})
    cp.append({"PageID": 2, "markdown": "two", "error": None, "meta": {"lang": "deu"}})
    cp.close()
    done = common.Checkpoint(tmp_path / "cp.jsonl").load()
    assert set(done) == {1, 2}
    assert done[2]["meta"]["lang"] == "deu"


def test_checkpoint_later_record_supersedes_earlier(tmp_path):
    """This is what makes retrying a failed page work on resume."""
    cp = common.Checkpoint(tmp_path / "cp.jsonl")
    cp.append({"PageID": 7, "markdown": None, "error": "Timeout", "meta": {}})
    cp.append({"PageID": 7, "markdown": "recovered", "error": None, "meta": {}})
    cp.close()
    done = common.Checkpoint(tmp_path / "cp.jsonl").load()
    assert done[7]["markdown"] == "recovered"
    assert done[7]["error"] is None


def test_checkpoint_survives_a_torn_final_line(tmp_path):
    path = tmp_path / "cp.jsonl"
    cp = common.Checkpoint(path)
    cp.append({"PageID": 1, "markdown": "kept", "error": None, "meta": {}})
    cp.close()
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"PageID": 2, "markdo')  # killed mid-append
    done = common.Checkpoint(path).load()
    assert set(done) == {1}


def test_checkpoint_missing_file_is_empty(tmp_path):
    assert common.Checkpoint(tmp_path / "nope.jsonl").load() == {}


# --------------------------------------------------------------------------- run_pages


def _echo(page):
    return f"text for {page['PageID']}", {"ok": True}


def _boom(page):
    if page["PageID"] == 2:
        raise RuntimeError("engine exploded")
    return f"text for {page['PageID']}", {}


def test_run_pages_skips_completed_pages(tmp_path):
    cp = common.Checkpoint(tmp_path / "cp.jsonl")
    cp.append({"PageID": 1, "markdown": "already", "error": None, "meta": {}})
    cp.close()
    pages = [{"PageID": 1}, {"PageID": 2}, {"PageID": 3}]
    completed, errors = common.run_pages(pages, _echo, checkpoint=common.Checkpoint(cp.path),
                                         workers=2, executor="thread")
    assert (completed, errors) == (2, 0)
    done = common.Checkpoint(cp.path).load()
    assert done[1]["markdown"] == "already"  # untouched
    assert done[3]["markdown"] == "text for 3"


def test_run_pages_records_failures_without_aborting(tmp_path):
    cp = common.Checkpoint(tmp_path / "cp.jsonl")
    pages = [{"PageID": n} for n in (1, 2, 3)]
    completed, errors = common.run_pages(pages, _boom, checkpoint=cp, workers=2,
                                         executor="thread")
    assert (completed, errors) == (3, 1)
    done = common.Checkpoint(cp.path).load()
    assert "engine exploded" in done[2]["error"]
    assert done[1]["error"] is None and done[3]["error"] is None


def test_run_pages_retries_errored_pages_on_resume(tmp_path):
    cp = common.Checkpoint(tmp_path / "cp.jsonl")
    cp.append({"PageID": 1, "markdown": None, "error": "Timeout", "meta": {}})
    cp.close()
    completed, errors = common.run_pages([{"PageID": 1}], _echo,
                                         checkpoint=common.Checkpoint(cp.path),
                                         workers=1, executor="thread")
    assert (completed, errors) == (1, 0)
    assert common.Checkpoint(cp.path).load()[1]["markdown"] == "text for 1"


def test_run_pages_can_keep_errors_when_retry_disabled(tmp_path):
    cp = common.Checkpoint(tmp_path / "cp.jsonl")
    cp.append({"PageID": 1, "markdown": None, "error": "Timeout", "meta": {}})
    cp.close()
    completed, _ = common.run_pages([{"PageID": 1}], _echo, checkpoint=common.Checkpoint(cp.path),
                                    workers=1, executor="thread", retry_errors=False)
    assert completed == 0


def test_run_pages_rejects_non_string_output(tmp_path):
    """A producer returning None must become a durable error, never a null OCR cell:
    score_dataset fails closed on non-string cells, but only after the run is over."""
    cp = common.Checkpoint(tmp_path / "cp.jsonl")
    completed, errors = common.run_pages([{"PageID": 1}], lambda page: (None, {}),
                                         checkpoint=cp, workers=1, executor="thread")
    assert (completed, errors) == (1, 1)
    assert "TypeError" in common.Checkpoint(cp.path).load()[1]["error"]


# --------------------------------------------------------------------------- image cells


def test_image_bytes_accepts_inline_bytes():
    assert common._image_bytes({"bytes": b"PNG", "path": None}, page_id=1) == b"PNG"
    assert common._image_bytes(b"PNG", page_id=1) == b"PNG"


def test_image_bytes_reads_a_path_backed_cell(tmp_path):
    """A folder-based dataset (which finebooks/bhl-impact-gt is) hands over bytes=None and a
    path into the Hub cache. Assuming bytes here cost a full run."""
    image = tmp_path / "p.webp"
    image.write_bytes(b"WEBPDATA")
    assert common._image_bytes({"bytes": None, "path": str(image)}, page_id=1) == b"WEBPDATA"


def test_image_bytes_rejects_a_genuinely_empty_cell():
    with pytest.raises(ValueError, match="empty image cell"):
        common._image_bytes({"bytes": None, "path": None}, page_id=42)


# --------------------------------------------------------------------------- finalize


def test_finalize_writes_canonical_schema_in_page_order(tmp_path):
    cp = common.Checkpoint(tmp_path / "cp.jsonl")
    for page_id in (3, 1, 2):
        cp.append({"PageID": page_id, "markdown": f"t{page_id}", "error": None, "meta": {"i": page_id}})
    cp.close()
    out = tmp_path / "run.parquet"
    frame = common.finalize(common.Checkpoint(cp.path), [1, 2, 3], model="eng-1", out_path=out)

    assert list(frame.columns) == list(common.CANONICAL_COLUMNS)
    assert list(frame.PageID) == [1, 2, 3]
    assert list(frame.markdown) == ["t1", "t2", "t3"]
    assert set(frame.model) == {"eng-1"}
    assert json.loads(frame.producer_meta[0]) == {"i": 1}
    assert list(pd.read_parquet(out).PageID) == [1, 2, 3]


def test_finalize_puts_the_error_sentinel_in_the_text_column(tmp_path):
    """The harness detects a failed page from the OCR STRING, never from a sibling column.
    Writing "" here would get the page scored as 'the engine read nothing'."""
    cp = common.Checkpoint(tmp_path / "cp.jsonl")
    cp.append({"PageID": 1, "markdown": None, "error": "HTTPError: 500", "meta": {}})
    cp.close()
    frame = common.finalize(common.Checkpoint(cp.path), [1], model="eng-1",
                            out_path=tmp_path / "run.parquet")
    assert frame.markdown[0].startswith(common.ERROR_PREFIX)
    assert "HTTPError: 500" in frame.markdown[0]
    assert frame.error[0] == "HTTPError: 500"


def test_finalize_error_sentinel_matches_the_pinned_harness_detector(tmp_path):
    """Guard against the sentinel drifting away from score_dataset.py's `_is_error`."""
    import pathlib
    import re

    source = (pathlib.Path(__file__).resolve().parent.parent / "harness" / "runners"
              / "score_dataset.py").read_text(encoding="utf-8")
    assert re.search(r'startswith\("__ERR__"\)', source), \
        "the pinned harness no longer detects the __ERR__ prefix — re-check common.ERROR_PREFIX"
    assert common.ERROR_PREFIX == "__ERR__"


def test_finalize_refuses_an_incomplete_run(tmp_path):
    cp = common.Checkpoint(tmp_path / "cp.jsonl")
    cp.append({"PageID": 1, "markdown": "t1", "error": None, "meta": {}})
    cp.close()
    with pytest.raises(SystemExit, match="missing from the checkpoint"):
        common.finalize(common.Checkpoint(cp.path), [1, 2], model="eng-1",
                        out_path=tmp_path / "run.parquet")


def test_finalize_keeps_a_deliberate_empty_string(tmp_path):
    """An empty read on a blank page is a real result, not an error — and the harness accepts
    "" as a valid deliberate empty output."""
    cp = common.Checkpoint(tmp_path / "cp.jsonl")
    cp.append({"PageID": 1, "markdown": "", "error": None, "meta": {}})
    cp.close()
    frame = common.finalize(common.Checkpoint(cp.path), [1], model="eng-1",
                            out_path=tmp_path / "run.parquet")
    assert frame.markdown[0] == ""
    assert frame.error[0] is None
