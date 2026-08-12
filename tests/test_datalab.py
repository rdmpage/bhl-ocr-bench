"""Tests for the Datalab adapter.

The async submit-then-poll shape has a failure mode the Mistral adapter does not: a soft failure
arrives as HTTP 200 carrying `success: false`, which is exactly the shape most likely to be
mistaken for a result and scored as "the engine read nothing". Several tests exist only to pin
that down.
"""
from __future__ import annotations

import json

import pytest
import requests
import responses

import datalab as D

URL = "https://www.datalab.to/api/v1/convert"
CHECK = "https://www.datalab.to/api/v1/convert/abc123"
WEBP = b"RIFF\x00\x00\x00\x00WEBP" + b"\x00" * 32


@pytest.fixture(autouse=True)
def _worker(monkeypatch):
    monkeypatch.setattr(D.time, "sleep", lambda _s: None)
    D._SETTINGS.clear()
    form = D.build_form("balanced", image_captions=False, keep_furniture=True)
    D._SETTINGS.update(session=requests.Session(), form=form, mode="balanced", timeout=30,
                       max_attempts=4, url=URL, poll_timeout=600.0, poll_interval=0.0,
                       limiter=D.RateLimiter(0))
    yield
    D._SETTINGS.clear()


# --------------------------------------------------------------------------- form fields


def test_form_disables_image_captions_by_default():
    """Figure descriptions are a product feature, not transcription: on this corpus they were
    4.5% of output and worth ~0.055 CER."""
    form = D.build_form("balanced", image_captions=False, keep_furniture=True)
    assert form["disable_image_captions"] == "true"
    assert form["output_format"] == "markdown"


def test_form_can_re_enable_captions():
    form = D.build_form("balanced", image_captions=True, keep_furniture=True)
    assert form["disable_image_captions"] == "false"


def test_furniture_flags_are_nested_json_under_additional_config():
    """The API takes these inside a JSON-encoded `additional_config` field, not as top-level
    form keys — passing them flat silently does nothing."""
    kept = json.loads(D.build_form("fast", image_captions=False,
                                   keep_furniture=True)["additional_config"])
    assert kept == {"keep_pageheader_in_output": True, "keep_pagefooter_in_output": True}
    dropped = json.loads(D.build_form("fast", image_captions=False,
                                      keep_furniture=False)["additional_config"])
    assert dropped == {"keep_pageheader_in_output": False, "keep_pagefooter_in_output": False}


@responses.activate
def test_every_form_field_reaches_the_wire():
    responses.add(responses.POST, URL, json=_submitted(), status=200)
    responses.add(responses.GET, CHECK, json={"status": "complete", "markdown": "x"}, status=200)
    D.ocr_page({"PageID": 1, "image_bytes": WEBP})
    raw = responses.calls[0].request.body
    raw = raw if isinstance(raw, bytes) else raw.encode()
    assert b"disable_image_captions" in raw and b"true" in raw
    assert b"keep_pageheader_in_output" in raw
    assert b"additional_config" in raw


def _submitted():
    return {"success": True, "error": None, "request_id": "abc123", "request_check_url": CHECK}


# --------------------------------------------------------------------------- upload shape


def test_upload_type_matches_the_bytes():
    assert D._upload_type(WEBP) == (".webp", "image/webp")
    assert D._upload_type(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16) == (".png", "image/png")
    with pytest.raises(ValueError, match="unrecognised image format"):
        D._upload_type(b"nope")


@responses.activate
def test_declares_the_real_mime_type_not_octet_stream():
    """The API validates the declared MIME type: posting WebP as application/octet-stream is
    rejected with 'Invalid file type' even though WebP is supported."""
    responses.add(responses.POST, URL, json=_submitted(), status=200)
    responses.add(responses.GET, CHECK, json={"status": "complete", "markdown": "x"}, status=200)
    D.ocr_page({"PageID": 1, "image_bytes": WEBP})
    raw = responses.calls[0].request.body
    raw = raw if isinstance(raw, bytes) else raw.encode()
    assert b"image/webp" in raw
    assert b"application/octet-stream" not in raw


@responses.activate
def test_uploads_webp_unconverted_with_the_requested_mode():
    """The corpus is WebP and /convert accepts it, so nothing is re-encoded between the ground
    truth's pixels and the engine's input."""
    responses.add(responses.POST, URL, json=_submitted(), status=200)
    responses.add(responses.GET, CHECK, json={"status": "complete", "markdown": "text"}, status=200)
    D.ocr_page({"PageID": 1, "image_bytes": WEBP})
    body = responses.calls[0].request.body
    raw = body if isinstance(body, bytes) else body.encode()
    assert b"page.webp" in raw
    assert b"balanced" in raw
    assert b"markdown" in raw


# --------------------------------------------------------------------------- polling


@responses.activate
def test_polls_until_complete():
    responses.add(responses.POST, URL, json=_submitted(), status=200)
    responses.add(responses.GET, CHECK, json={"status": "processing"}, status=200)
    responses.add(responses.GET, CHECK, json={"status": "processing"}, status=200)
    responses.add(responses.GET, CHECK, status=200, json={
        "status": "complete", "markdown": "# done", "versions": {"chandra": "2.0"},
    })
    text, meta = D.ocr_page({"PageID": 1, "image_bytes": WEBP})
    assert text == "# done"
    assert meta["polls"] == 3
    # Hosted rows are weak evidence precisely because you cannot usually say what served them;
    # keep it whenever the API volunteers it.
    assert meta["versions"] == {"chandra": "2.0"}


@responses.activate
def test_empty_markdown_is_a_result_not_an_error():
    responses.add(responses.POST, URL, json=_submitted(), status=200)
    responses.add(responses.GET, CHECK, json={"status": "complete", "markdown": ""}, status=200)
    text, _ = D.ocr_page({"PageID": 1, "image_bytes": WEBP})
    assert text == ""


@responses.activate
def test_soft_failure_with_http_200_is_an_error():
    """success:false arrives as HTTP 200. Returning it would score the page as a blank read."""
    responses.add(responses.POST, URL, json=_submitted(), status=200)
    responses.add(responses.GET, CHECK, status=200,
                  json={"success": False, "error": "page limit exceeded"})
    with pytest.raises(RuntimeError, match="page limit exceeded"):
        D.ocr_page({"PageID": 1, "image_bytes": WEBP})


@responses.activate
def test_submit_rejected_up_front_is_an_error():
    responses.add(responses.POST, URL, status=200,
                  json={"success": False, "error": "out of credits"})
    with pytest.raises(RuntimeError, match="out of credits"):
        D.ocr_page({"PageID": 1, "image_bytes": WEBP})


@responses.activate
def test_missing_check_url_is_an_error():
    responses.add(responses.POST, URL, json={"success": True, "request_id": "x"}, status=200)
    with pytest.raises(ValueError, match="no request_check_url"):
        D.ocr_page({"PageID": 1, "image_bytes": WEBP})


@responses.activate
def test_completed_result_without_text_fails_closed():
    responses.add(responses.POST, URL, json=_submitted(), status=200)
    responses.add(responses.GET, CHECK, json={"status": "complete", "pages": 1}, status=200)
    with pytest.raises(ValueError, match="no markdown/text field"):
        D.ocr_page({"PageID": 1, "image_bytes": WEBP})


@responses.activate
def test_poll_timeout_becomes_a_durable_error(monkeypatch):
    """A wedged request must not hang the run — it becomes an error row the resume can retry."""
    responses.add(responses.POST, URL, json=_submitted(), status=200)
    responses.add(responses.GET, CHECK, json={"status": "processing"}, status=200)
    D._SETTINGS["poll_timeout"] = -1.0  # already expired
    with pytest.raises(TimeoutError, match="result not ready"):
        D.ocr_page({"PageID": 1, "image_bytes": WEBP})


# --------------------------------------------------------------------------- retries


@responses.activate
def test_retries_a_429_on_submit():
    responses.add(responses.POST, URL, json={"detail": "rate limited"}, status=429)
    responses.add(responses.POST, URL, json=_submitted(), status=200)
    responses.add(responses.GET, CHECK, json={"status": "complete", "markdown": "ok"}, status=200)
    text, _ = D.ocr_page({"PageID": 1, "image_bytes": WEBP})
    assert text == "ok"


@responses.activate
def test_does_not_retry_a_client_error():
    responses.add(responses.POST, URL, json={"detail": "bad key"}, status=401)
    with pytest.raises(ValueError, match="HTTP 401"):
        D.ocr_page({"PageID": 1, "image_bytes": WEBP})
    assert len(responses.calls) == 1


# --------------------------------------------------------------------------- rate limiter


def test_rate_limiter_allows_the_budget_then_blocks(monkeypatch):
    slept = []
    monkeypatch.setattr(D.time, "sleep", lambda s: slept.append(s))
    clock = {"t": 1000.0}
    monkeypatch.setattr(D.time, "monotonic", lambda: clock["t"])

    limiter = D.RateLimiter(3)
    for _ in range(3):
        limiter.acquire()
    assert slept == []          # budget available, no waiting

    # Fourth call inside the same minute must wait, then succeed once the window rolls.
    def advancing_sleep(seconds):
        slept.append(seconds)
        clock["t"] += 61.0
    monkeypatch.setattr(D.time, "sleep", advancing_sleep)
    limiter.acquire()
    assert len(slept) == 1 and slept[0] > 0


def test_rate_limiter_disabled_never_waits(monkeypatch):
    monkeypatch.setattr(D.time, "sleep", lambda _s: pytest.fail("should not sleep"))
    limiter = D.RateLimiter(0)
    for _ in range(50):
        limiter.acquire()


# --------------------------------------------------------------------------- api key


def test_api_key_prefers_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("DATALAB_API_KEY", " env-key ")
    assert D._read_api_key(tmp_path / "unused") == "env-key"


def test_api_key_falls_back_to_a_file(monkeypatch, tmp_path):
    monkeypatch.delenv("DATALAB_API_KEY", raising=False)
    f = tmp_path / "api_key"
    f.write_text("file-key\n")
    assert D._read_api_key(f) == "file-key"


def test_api_key_never_reaches_provenance():
    source = pytest.importorskip("pathlib").Path(D.__file__).read_text(encoding="utf-8")
    provenance_block = source.split("meta_path.write_text(", 1)[1]
    assert "api_key" not in provenance_block
