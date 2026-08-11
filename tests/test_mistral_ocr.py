"""Tests for the Mistral OCR adapter.

This engine costs money per page and scoring fails closed, so the retry and empty-page paths are
validated against a mocked API before a real run rather than discovered during one.
"""
from __future__ import annotations

import json

import pytest
import requests
import responses

import mistral_ocr as M

URL = "https://api.mistral.ai/v1/ocr"
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
WEBP = b"RIFF\x00\x00\x00\x00WEBP" + b"\x00" * 32


@pytest.fixture(autouse=True)
def _worker(monkeypatch):
    """Install a real session plus a no-op sleep so backoff tests do not actually wait."""
    monkeypatch.setattr(M.time, "sleep", lambda _s: None)
    M._SETTINGS.clear()
    M._SETTINGS.update(session=requests.Session(), model="mistral-ocr-latest", timeout=30,
                       max_attempts=4, base_url="https://api.mistral.ai/v1")
    yield
    M._SETTINGS.clear()


def _ok(markdown, **extra):
    return {"pages": [{"index": 0, "markdown": markdown}], "model": "mistral-ocr-2506",
            "usage_info": {"pages_processed": 1}, **extra}


# --------------------------------------------------------------------------- mime sniffing


def test_mime_sniffs_the_corpus_format():
    assert M._mime(WEBP) == "image/webp"
    assert M._mime(PNG) == "image/png"
    assert M._mime(b"\xff\xd8\xff" + b"\x00" * 16) == "image/jpeg"


def test_mime_rejects_an_unknown_format():
    with pytest.raises(ValueError, match="unrecognised image format"):
        M._mime(b"not an image at all")


# --------------------------------------------------------------------------- happy paths


@responses.activate
def test_returns_markdown_and_records_the_concrete_model():
    responses.add(responses.POST, URL, json=_ok("# A page\n\ntext"), status=200)
    text, meta = M.ocr_page({"PageID": 1, "image_bytes": WEBP})
    assert text == "# A page\n\ntext"
    # The board row should say which model actually ran, not which alias was typed.
    assert meta["model"] == "mistral-ocr-2506"
    assert meta["attempts"] == 1


@responses.activate
def test_sends_a_data_uri_with_the_sniffed_mime_and_no_image_payload():
    responses.add(responses.POST, URL, json=_ok("x"), status=200)
    M.ocr_page({"PageID": 1, "image_bytes": WEBP})
    body = json.loads(responses.calls[0].request.body)
    assert body["document"]["image_url"].startswith("data:image/webp;base64,")
    assert body["include_image_base64"] is False
    assert body["model"] == "mistral-ocr-latest"


@responses.activate
def test_an_empty_read_is_a_result_not_an_error():
    """428 pages are sparse/blank. Raising on empty would turn the most interesting pages in
    the corpus into producer errors and make the whole run ineligible."""
    responses.add(responses.POST, URL, json=_ok(""), status=200)
    text, _ = M.ocr_page({"PageID": 1, "image_bytes": WEBP})
    assert text == ""


@responses.activate
def test_multiple_returned_pages_are_joined():
    responses.add(responses.POST, URL, status=200, json={
        "pages": [{"markdown": "one"}, {"markdown": "two"}], "model": "m",
    })
    text, meta = M.ocr_page({"PageID": 1, "image_bytes": WEBP})
    assert text == "one\n\ntwo"
    assert meta["api_pages"] == 2


# --------------------------------------------------------------------------- retries


@responses.activate
def test_retries_a_429_then_succeeds():
    responses.add(responses.POST, URL, json={"message": "rate limited"}, status=429,
                  headers={"Retry-After": "1"})
    responses.add(responses.POST, URL, json=_ok("recovered"), status=200)
    text, meta = M.ocr_page({"PageID": 1, "image_bytes": WEBP})
    assert text == "recovered"
    assert meta["attempts"] == 2


@responses.activate
def test_retries_a_500_then_succeeds():
    responses.add(responses.POST, URL, body="upstream boom", status=503)
    responses.add(responses.POST, URL, json=_ok("recovered"), status=200)
    text, _ = M.ocr_page({"PageID": 1, "image_bytes": WEBP})
    assert text == "recovered"


@responses.activate
def test_gives_up_after_max_attempts_and_reports_the_last_failure():
    for _ in range(6):
        responses.add(responses.POST, URL, body="still down", status=502)
    with pytest.raises(RuntimeError, match="giving up after 4 attempts"):
        M.ocr_page({"PageID": 1, "image_bytes": WEBP})
    assert len(responses.calls) == 4


@responses.activate
def test_does_not_retry_a_client_error():
    """A 401/400 will not get better by repeating it — burning the rate limit on it just
    slows every other page down."""
    responses.add(responses.POST, URL, json={"message": "unauthorized"}, status=401)
    with pytest.raises(ValueError, match="HTTP 401"):
        M.ocr_page({"PageID": 1, "image_bytes": WEBP})
    assert len(responses.calls) == 1


@responses.activate
def test_retries_a_timeout():
    responses.add(responses.POST, URL, body=requests.Timeout("too slow"))
    responses.add(responses.POST, URL, json=_ok("recovered"), status=200)
    text, _ = M.ocr_page({"PageID": 1, "image_bytes": WEBP})
    assert text == "recovered"


# --------------------------------------------------------------------------- malformed responses


@responses.activate
def test_rejects_a_response_with_no_pages():
    responses.add(responses.POST, URL, json={"model": "m"}, status=200)
    with pytest.raises(ValueError, match="no pages"):
        M.ocr_page({"PageID": 1, "image_bytes": WEBP})


@responses.activate
def test_rejects_a_page_object_without_markdown():
    responses.add(responses.POST, URL, json={"pages": [{"index": 0}]}, status=200)
    with pytest.raises(ValueError, match="no markdown key"):
        M.ocr_page({"PageID": 1, "image_bytes": WEBP})


@responses.activate
def test_rejects_non_string_markdown():
    responses.add(responses.POST, URL, json={"pages": [{"markdown": 42}]}, status=200)
    with pytest.raises(ValueError, match="expected str"):
        M.ocr_page({"PageID": 1, "image_bytes": WEBP})


# --------------------------------------------------------------------------- model resolution


# --------------------------------------------------------------------------- api key


def test_api_key_prefers_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("MISTRAL_API_KEY", "  from-env  ")
    assert M._read_api_key(tmp_path / "unused") == "from-env"


def test_api_key_falls_back_to_a_file(monkeypatch, tmp_path):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    key_file = tmp_path / "api_key"
    key_file.write_text("from-file\n")
    assert M._read_api_key(key_file) == "from-file"


def test_api_key_missing_explains_both_routes(monkeypatch, tmp_path):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    monkeypatch.setattr(M, "DEFAULT_KEY_FILE", tmp_path / "absent")
    with pytest.raises(SystemExit, match="MISTRAL_API_KEY"):
        M._read_api_key(None)


def test_api_key_never_reaches_provenance():
    """Guard: the provenance file ships in git, so nothing key-shaped may be written into it."""
    source = (M.pathlib.Path(M.__file__)).read_text(encoding="utf-8")
    provenance_block = source.split("meta_path.write_text(", 1)[1]
    assert "api_key" not in provenance_block


@responses.activate
def test_resolve_model_accepts_a_listed_model():
    responses.add(responses.GET, "https://api.mistral.ai/v1/models", status=200, json={
        "data": [{"id": "mistral-ocr-latest"}, {"id": "mistral-large"}],
    })
    model, ocr_models = M.resolve_model("k", "mistral-ocr-latest", "https://api.mistral.ai/v1")
    assert model == "mistral-ocr-latest"
    assert ocr_models == ["mistral-ocr-latest"]


@responses.activate
def test_resolve_model_fails_closed_and_lists_the_alternatives():
    """Fail before spending anything, and say what IS available — guessing a model name is
    how you discover at page 900 that you ran the wrong one."""
    responses.add(responses.GET, "https://api.mistral.ai/v1/models", status=200, json={
        "data": [{"id": "mistral-ocr-2506"}, {"id": "mistral-small"}],
    })
    with pytest.raises(SystemExit, match="mistral-ocr-2506"):
        M.resolve_model("k", "mistral-ocr-4", "https://api.mistral.ai/v1")
