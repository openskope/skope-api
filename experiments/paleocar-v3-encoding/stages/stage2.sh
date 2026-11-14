#!/usr/bin/env bash
# Stage 2: the six candidates stage 1 picked, over all 1898 years, in the simulation of the server.
# Two batches of three, so the volume holds one batch at a time. The host keeps every variant.
set -euo pipefail
cd "$(dirname "$0")/.."
export WORK=${WORK:?Set WORK; see README.md}/stage2
export RESULTS=$PWD/results/stage2
mkdir -p "$RESULTS"
exec > >(tee "results/stage2.log") 2>&1
BATCHES=(
  "b512-band-c25-uint16 b128-tile-c100-uint16 b64-tile-c100-uint16"
  "b128-tile-c250-uint16 b64-tile-c250-uint16 b64-tile-c25-uint16"
)
for batch in "${BATCHES[@]}"; do
  echo "=== $batch: build $(date)"
  # shellcheck disable=SC2086  # $batch is a list of names
  ./run.sh build --full-range --no-baseline --blocks 64,128,512 --dtypes uint16 --only $batch 2>&1 \
    | grep -v -e '^Warning' -e 'no valid pixels'
  echo "=== $batch: load $(date)"
  # shellcheck disable=SC2086
  ./run.sh c-load $batch
  echo "=== $batch: measure $(date)"
  # shellcheck disable=SC2086
  ./run.sh C --rounds 3 --only $batch 2>&1 | grep -v -e 'Warning' -e 'warnings.warn'
done
./run.sh c-clean
./run.sh analyze --select > results/report-stage2.md
echo "=== done $(date)"
