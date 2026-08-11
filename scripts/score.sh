#!/usr/bin/env bash
# Score a producer run through the PINNED harness.
#
#   scripts/score.sh runs/tesseract-5
#
# Three flags here are not optional, and getting any of them wrong produces a plausible number
# rather than an error:
#
#   --gt benchmark/gt   score_dataset.py defaults to `davanstrien/bhl-eval-impact-sample`, which
#                       has been removed from the Hub (401). Worse, pointing it at the SOURCE
#                       dataset `finebooks/bhl-impact-gt` succeeds: score_gt_row reads body_text /
#                       furniture_text with .get(), so a source-shaped GT silently degrades the
#                       board's body-only headline into full-text CER. benchmark/gt is the local
#                       rebuild that carries those columns (see benchmark/build_benchmark.py).
#
#   --model-col model   our parquet names the row per page; without this every row is scored
#                       under the literal model name "model".
#
#   --out scorecards/   the default writes into harness/data/, i.e. inside the read-only pinned
#                       submodule. Keep our artifacts in our own repo.
set -euo pipefail

RUN_DIR="${1:?usage: scripts/score.sh <run-dir>   (e.g. runs/tesseract-5)}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NAME="$(basename "$RUN_DIR")"

OCR="$ROOT/$RUN_DIR/run.parquet"
PROV="$ROOT/$RUN_DIR/producer-run.json"
GT="$ROOT/benchmark/gt"
OUT="$ROOT/scorecards/$NAME.parquet"

for required in "$OCR" "$PROV"; do
  [ -f "$required" ] || { echo "missing $required — has the producer finished?" >&2; exit 1; }
done
[ -d "$GT" ] || { echo "missing $GT — run: uv run benchmark/build_benchmark.py" >&2; exit 1; }

mkdir -p "$ROOT/scorecards"
cd "$ROOT/harness"
exec uv run runners/score_dataset.py \
  --ocr "$OCR" \
  --ocr-col markdown \
  --key-col PageID \
  --model-col model \
  --gt "$GT" \
  --run-provenance-file "$PROV" \
  --out "$OUT"
