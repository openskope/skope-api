#!/usr/bin/env bash
# After stage 2: the selected layout in UInt16 and UInt32, measured together over all 1898 years.
# The UInt16 variant from stage 2 is reused; only the UInt32 one is built.
# Usage: stages/stage2-uint32.sh b128-tile-c100   (the selected variant's name without its data type)
set -euo pipefail
cd "$(dirname "$0")/.."
layout=${1:?usage: $0 <layout, e.g. b128-tile-c100>}
block=${layout#b}; block=${block%%-*}
export WORK=${WORK:?Set WORK; see README.md}/stage2
export RESULTS=$PWD/results/stage2-uint32
mkdir -p "$RESULTS"
exec > >(tee "results/stage2-uint32.log") 2>&1
echo "=== $layout: build $(date)"
./run.sh build --full-range --no-baseline --blocks "$block" --dtypes uint16,uint32 --only "$layout-*" 2>&1 \
  | grep -v -e '^Warning' -e 'no valid pixels'
echo "=== $layout: load $(date)"
./run.sh c-load "$layout-uint16" "$layout-uint32"
echo "=== $layout: measure $(date)"
./run.sh C --rounds 3 --only "$layout-*" 2>&1 | grep -v -e 'Warning' -e 'warnings.warn'
./run.sh c-clean
./run.sh analyze > results/report-stage2-uint32.md
echo "=== done $(date)"
