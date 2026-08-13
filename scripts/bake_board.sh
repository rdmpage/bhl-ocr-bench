#!/usr/bin/env bash
# Rebuild every scorecard, the tracked provenance snapshot, and the board — in that order.
#
#   scripts/bake_board.sh
#
# No inference runs. Every row is re-scored from its cached `runs/<dir>/run.parquet` through the
# pinned harness, so this is a pure function of (cached raw output, run provenance, pinned scorer).
# Re-running it must not move a number; if one moves, something that was supposed to be frozen
# is not.
#
# Three things here are load-bearing:
#
#   The board row name is NOT the run directory name, and not always the run parquet's `model`
#   column either. `datalab-balanced-apidefault` is scored from `runs/datalab-balanced/`, whose
#   parquet says `datalab-balanced` — the distinguishing label lives only in the scorer
#   invocation. That is why each row below carries its run dir and its model flag explicitly
#   rather than being globbed: a glob would silently rename that row and split its history.
#
#   Scorecards are re-scored BEFORE the board is baked, because each scorecard embeds the run
#   provenance it was scored against and the board reads it back out from there. Baking without
#   re-scoring republishes whatever provenance was current at scoring time — which is how the
#   board came to claim `mistral-ocr-2512` ran 2,165 pages in 5.5 s.
#
#   The provenance snapshot in `provenance/` is the tracked evidence; `runs/` is gitignored. It is
#   copied from the run dirs here rather than by hand, because it had drifted from them.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GT="$ROOT/benchmark/gt"

# board row | run dir | scorer model selector
ROWS=(
  "tesseract-5|tesseract-5|--model-col model"
  "mistral-ocr-4-1|mistral-ocr-4-1|--model-col model"
  "mistral-ocr-2512|mistral-ocr-2512|--model-col model"
  "datalab-balanced-apidefault|datalab-balanced|--model datalab-balanced-apidefault"
  "datalab-balanced-configured|datalab-balanced-configured|--model-col model"
  "surya-ocr-2|surya-ocr-2|--model-col model"
)

[ -d "$GT" ] || { echo "missing $GT — run: uv run benchmark/build_benchmark.py" >&2; exit 1; }
mkdir -p "$ROOT/scorecards" "$ROOT/provenance" "$ROOT/boards"

for row in "${ROWS[@]}"; do
  IFS='|' read -r name dir selector <<< "$row"
  OCR="$ROOT/runs/$dir/run.parquet"
  PROV="$ROOT/runs/$dir/producer-run.json"
  for required in "$OCR" "$PROV"; do
    [ -f "$required" ] || { echo "missing $required — has the producer finished?" >&2; exit 1; }
  done

  echo "==> scoring $name (from runs/$dir)"
  # shellcheck disable=SC2086 # $selector is a deliberate two-token flag pair
  (cd "$ROOT/harness" && uv run runners/score_dataset.py \
    --ocr "$OCR" --ocr-col markdown --key-col PageID $selector \
    --gt "$GT" --run-provenance-file "$PROV" \
    --out "$ROOT/scorecards/$name.parquet")

  cp "$PROV" "$ROOT/provenance/$name.json"
done

echo "==> baking board"
(cd "$ROOT/harness" && uv run runners/leaderboard.py "$ROOT/scorecards/*.parquet")
# leaderboard.py hardcodes its output inside the pinned submodule. data/* is gitignored there, so
# this leaves the submodule clean, but the artifact belongs in our repo.
cp "$ROOT/harness/data/leaderboard.json" "$ROOT/boards/leaderboard.json"
echo "board -> boards/leaderboard.json"

git -C "$ROOT/harness" diff --quiet || { echo "ERROR: pinned harness is dirty" >&2; exit 1; }
