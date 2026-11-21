#!/usr/bin/env bash
# Build and benchmark the encoding variants. See README.md for the order of steps.
#
#   ./run.sh build [build_variants.py args]   build variants (release build image)
#   ./run.sh setup-a                          local virtualenv for setting A
#   ./run.sh A [bench args]                   native, no Docker, warm (comparison only)
#   ./run.sh B [bench args]                   Docker on a bind mount, warm (comparison only)
#   ./run.sh c-load | C [bench args] | c-clean   simulation of the server on a Docker named volume
#   ./run.sh parity                           check the extraction mirror against the API's function
#   ./run.sh drop GLOB                        delete built variants matching GLOB (never the baseline)
#   ./run.sh analyze [--select]               report every results file; --select applies the selection rule
#
# Extra bench args are passed to `bench.py run`, e.g. --rounds 5 --only 'b128-*' --mode cold.
#
# WORK (required except for analyze): a directory with room for the variants, outside the repository.
# SOURCE (required for build): a local copy of the ppt_annual source GeoTIFF.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
API_REPO=$(cd "$HERE/../.." && pwd)
case ${1:-} in
  analyze | '') ;;
  *) : "${WORK:?Set WORK to a directory for the variants (see README.md)}" ;;
esac
WORK=${WORK:-}
VARIANTS=${VARIANTS:-$WORK/variants}
RESULTS=${RESULTS:-$HERE/results}
VENV=${VENV:-$WORK/venv}
VOLUME=${VOLUME:-skope-exp-encoding}

INGEST_IMAGE=openskope/skope-api-ingest:latest
API_IMAGE=openskope/skope-api:dev
TITILER_IMAGE=openskope/skope-api-titiler:latest

# Staging host skope-01: 8 vCPU, 16 GB; GDAL's default cache is 5% of RAM per process.
STAGING_LIMITS=(--cpus 8 -e GDAL_CACHEMAX=819)

mkdir -p "$RESULTS"

bench_in() {  # image variants-mount setting [docker args...] -- [bench args...]
  local image=$1 mount=$2 setting=$3 ops=$4; shift 4
  local docker_args=()
  while [[ $# -gt 0 && $1 != -- ]]; do docker_args+=("$1"); shift; done
  [[ ${1:-} == -- ]] && shift
  docker run --rm --entrypoint python "${docker_args[@]}" \
    -v "$mount":/variants:ro -v "$HERE":/exp:ro -v "$RESULTS":/results \
    "$image" /exp/bench.py run --variants /variants --setting "$setting" --ops "$ops" \
    --out "/results/$setting-$ops.jsonl" "$@"
}

case ${1:-} in
  build)
    shift
    : "${SOURCE:?Set SOURCE to a local copy of the ppt_annual source GeoTIFF (see README.md)}"
    mkdir -p "$WORK"
    docker run --rm --entrypoint python \
      -v "$SOURCE":/source.tif:ro -v "$WORK":/work -v "$HERE":/exp:ro \
      -v "$API_REPO/timeseries/ingest/src":/src:ro -e PYTHONPATH=/src \
      "$INGEST_IMAGE" -u /exp/build_variants.py /source.tif /work/variants --scratch /work/scratch "$@"
    ;;
  setup-a)
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install --quiet "rio-tiler==9.0.6" "rasterio==1.5.0" numpy
    ;;
  A)
    shift
    "$VENV/bin/python" "$HERE/bench.py" run --variants "$VARIANTS" --setting A --ops extract views \
      --mode warm --out "$RESULTS/A.jsonl" "$@"
    ;;
  B)
    shift
    bench_in "$API_IMAGE" "$VARIANTS" B extract --cpus 8 -- --mode warm "$@"
    bench_in "$TITILER_IMAGE" "$VARIANTS" B views --cpus 8 -- --mode warm "$@"
    ;;
  c-load)  # copy variants into the volume, replacing what's there: ./run.sh c-load [names...]
    shift
    docker volume create "$VOLUME" >/dev/null
    docker run --rm -v "$VARIANTS":/from:ro -v "$VOLUME":/to alpine sh -c \
      'rm -rf /to/* && cd /from && if [ $# -gt 0 ]; then cp -a "$@" /to/; else cp -a . /to/; fi' sh "$@"
    docker run --rm -v "$VOLUME":/to alpine du -sh /to
    ;;
  C)
    shift
    bench_in "$API_IMAGE" "$VOLUME" C extract "${STAGING_LIMITS[@]}" -- --mode cold "$@"
    bench_in "$TITILER_IMAGE" "$VOLUME" C views "${STAGING_LIMITS[@]}" -- --mode cold "$@"
    ;;
  c-clean)
    docker volume rm "$VOLUME"
    ;;
  drop)  # drop built variants matching a glob, keeping the baseline: ./run.sh drop 'b64-*'
    shift
    for dir in "$VARIANTS"/$1; do
      [[ $dir == */b512-band-c100-uint32 ]] || rm -rf "$dir"
    done
    ;;
  parity)
    docker run --rm --entrypoint python -w /code -v "$VARIANTS":/variants:ro -v "$HERE":/exp:ro \
      "$API_IMAGE" /exp/bench.py parity --variants /variants
    ;;
  analyze)
    shift
    python3 "$HERE/analyze.py" "$@" "$RESULTS"/*.jsonl
    ;;
  *)
    sed -n '2,17p' "$0"
    exit 2
    ;;
esac
