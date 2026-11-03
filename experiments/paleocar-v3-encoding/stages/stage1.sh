#!/usr/bin/env bash
# Stage 1: every variant in the staging simulation, in batches by block size.
set -euo pipefail
cd "$(dirname "$0")/.."
export WORK=${WORK:?Set WORK; see README.md}/stage1
export RESULTS=$PWD/results/stage1
for b in 128 64 256 512; do
  echo "=== batch b$b: build $(date)"
  ./run.sh build --blocks $b 2>&1 | grep -v -e '^Warning' -e 'no valid pixels'
  if [[ $b == 128 ]]; then echo "=== parity"; ./run.sh parity; fi
  echo "=== batch b$b: load $(date)"
  ./run.sh c-load
  echo "=== batch b$b: measure $(date)"
  ./run.sh C --rounds 3 2>&1 | grep -v -e 'Warning' -e 'warnings.warn'
  ./run.sh drop "b$b-*"
done
./run.sh c-clean
./run.sh analyze > results/report-stage1.md
echo "=== done $(date)"
