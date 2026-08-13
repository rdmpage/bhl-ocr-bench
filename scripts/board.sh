#!/usr/bin/env bash
# The one entry point: score rows through the PINNED harness, then rebuild the board.
#
#   scripts/board.sh                 # every row: re-score all, refresh provenance/, bake
#   scripts/board.sh surya-ocr-2     # just that row (then bake, which is cheap)
#   scripts/board.sh --list          # the registered rows and where each is scored from
#
# No inference runs here — every row is scored from its cached `runs/<dir>/run.parquet`, so this
# is a pure function of (cached raw output, run provenance, pinned scorer). Re-running it must not
# move a number; if one moves, something that was supposed to be frozen is not.
#
# ROWS below is the registry. A run that is not in it cannot be scored, which is deliberate: a
# `--limit N` smoke run is not scoreable anyway (the page set will not match and the scorer fails
# closed), so every scoreable run is a board row and ought to be declared as one.
#
# Six things here are load-bearing. The first three are scorer flags that produce a plausible
# number rather than an error when you get them wrong; the last three are sequencing.
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
#
#   The board row name is NOT the run directory name, and not always the run parquet's `model`
#   column either. `datalab-balanced-apidefault` is scored from `runs/datalab-balanced/`, whose
#   parquet says `datalab-balanced` — the distinguishing label lives only in the scorer
#   invocation, which is why that row overrides the selector below. Globbing the run dirs instead
#   would silently rename that row and split its history.
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

if [ "${1:-}" = "--list" ]; then
  printf '%-30s %s\n' "ROW" "SCORED FROM"
  for row in "${ROWS[@]}"; do
    IFS='|' read -r name dir _ <<< "$row"
    printf '%-30s runs/%s\n' "$name" "$dir"
  done
  exit 0
fi

# Resolve the requested rows up front, so a typo fails before anything is scored rather than
# after, and never by quietly scoring nothing.
selected=()
if [ "$#" -eq 0 ]; then
  selected=("${ROWS[@]}")
else
  for want in "$@"; do
    found=""
    for row in "${ROWS[@]}"; do
      [ "${row%%|*}" = "$want" ] && { selected+=("$row"); found=1; break; }
    done
    [ -n "$found" ] || {
      echo "unknown row '$want' — registered rows:" >&2
      for row in "${ROWS[@]}"; do echo "  ${row%%|*}" >&2; done
      exit 1
    }
  done
fi

[ -d "$GT" ] || { echo "missing $GT — run: uv run benchmark/build_benchmark.py" >&2; exit 1; }
mkdir -p "$ROOT/scorecards" "$ROOT/provenance" "$ROOT/boards"

for row in "${selected[@]}"; do
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
