"""Tests for the path scrubber that runs on every bake.

It edits generated artifacts in place, so the risk is not that it fails — it is that it quietly
rewrites more than the path prefixes, or that a second bake undoes the first.
"""
from __future__ import annotations

import json

import scrub_local_paths as S


def test_the_repo_root_becomes_relative():
    assert S.scrub(f"{S.ROOT}/runs/tesseract-5/run.parquet") == "runs/tesseract-5/run.parquet"


def test_the_home_directory_becomes_a_tilde():
    assert S.scrub(f"{S.HOME}/.cache/huggingface/hub/x.gguf") == "~/.cache/huggingface/hub/x.gguf"


def test_the_repo_root_wins_over_the_home_directory():
    """The repo lives under the home directory, so the other order would rewrite the repo root to
    `~/Development/...` and never match it again."""
    assert "Development" not in S.scrub(f"{S.ROOT}/benchmark/gt")


def test_it_is_idempotent():
    """It runs on every bake, so a second pass must be a no-op."""
    once = S.scrub(f"{S.ROOT}/benchmark/gt {S.HOME}/.cache/x")
    assert S.scrub(once) == once


def test_it_keeps_the_weights_revision():
    """The HF snapshot hash is the weights revision — the reason the path is recorded at all."""
    sha = "6a3a4c30e5e74446d4f8b6afd05b2f2da970f470"
    assert sha in S.scrub(f"{S.HOME}/.cache/huggingface/hub/models--x/snapshots/{sha}/surya-2.gguf")


def test_it_changes_nothing_but_the_paths(tmp_path):
    """A JSON round-trip would reformat bytes unrelated to the paths; text substitution must not."""
    blob = {"cer_reading_micro": 0.0672, "ci": [0.0419, 0.0843], "unicode": "café — ligature ﬁ",
            "ocr_source": f"{S.ROOT}/runs/tesseract-5/run.parquet"}
    original = json.dumps(blob, indent=2, sort_keys=True) + "\n"
    scrubbed = S.scrub(original)

    assert json.loads(scrubbed)["ocr_source"] == "runs/tesseract-5/run.parquet"
    # Every other line survives byte-for-byte, including the non-ASCII one.
    before = [ln for ln in original.splitlines() if "ocr_source" not in ln]
    assert [ln for ln in scrubbed.splitlines() if "ocr_source" not in ln] == before
