# PaleoCAR v3 encoding experiment

The harness for EXP-002, EXP-003 and EXP-004 (dataset release spec, Section 18). It runs in two stages:

1. **Stage 1:** every layout over a 500-year window, to choose the stage 2 candidates from the raw measurements.
2. **Stage 2:** the candidates over all 1898 years, scored by the selection rule in the report. Then the selected
   layout in UInt16 and UInt32.

## Files

| File | What it does |
| --- | --- |
| `build_variants.py` | Builds the variants with the release build's COG options. |
| `bench.py` | Measures map views and extraction, one fresh process per measurement. |
| `analyze.py` | Reports the measurements; with `--select`, applies the selection rule. |
| `areas.geojson` | The four extraction areas. |
| `run.sh` | Wraps the Docker invocations for each setting. |
| `stages/*.sh` | The commands for each stage. |

Runs write their measurements, logs and reports to `results/` (override with `RESULTS=`), which is not committed.
The findings are in the [experiment report](../../docs/experiments/paleocar-v3-encoding.md).

## What is measured

- **Map views:** every tile of a view, at MapLibre's tileSize 128, served by 2 processes like TiTiler's 2 workers,
  over 12-step slider sequences. Tiles open a one-band `vrt://` view of the file, as the API asks TiTiler for them.
  - `fit` and `native`: the whole region in a 1280 × 800 map, at zoom 7 (the initial view) and zoom 8 (the first
    full-resolution zoom).
  - `area-<name>`: the visualize map (about 640 × 550 px) framed on each extraction area, as skopeui's
    `fitToGeoJson` frames a study area.
- **Extraction:** a mirror of the API's read loop (10 threads, 1M-cell band chunking) over each area, for 10, 100 and
  500 years and every timestep.
- **For every measurement:** time, peak memory, block-device reads and sectors read (cold reads only), and a hash of
  the results, to check that every layout gives identical outputs.

## Settings

| | Where | Cache | Role |
| --- | --- | --- | --- |
| C | Linux containers (API image for extraction, TiTiler image for tiles), named volume, `--cpus 8`, `GDAL_CACHEMAX=819` | cold | the simulation of the server; decides |
| B | Same images, variants on a bind mount from the host | warm | comparison only |
| A | Native, no Docker: a virtualenv with TiTiler's rio-tiler and rasterio | warm | comparison only |

Cold reads need Linux (`posix_fadvise`), which settings B and C get from Docker.

## Setup

- **Docker** with at least 8 CPUs and enough memory for a 1.5 GB extraction (the experiment ran with 7.75 GiB), and
  the repository's images:
  `make build` builds the API and TiTiler images, `make release` the ingest image.
- **`SOURCE`:** a local copy of the `ppt_annual` source GeoTIFF, the URI that
  `timeseries/ingest/datasets/paleocar_v3/source-manifest.yml` lists for it (about 1,900 bands).
- **`WORK`:** a directory outside the repository for variants, scratch files and the setting-A virtualenv. The stage
  scripts use `$WORK/stage1` and `$WORK/stage2`. Stage 2 keeps its seven variants, about 30 GB, and setting C copies
  each batch into a Docker volume, so allow about twice that.

```bash
export SOURCE=/path/to/ppt_annual/prediction_scaled.tif
export WORK=/path/to/exp-encoding
```

Nothing here is imported by the API or the release build (EXP-001).

## Running it

Stop the dev stack first (`make down`), so it doesn't compete for memory and CPU, and keep the machine from sleeping:
stage 1 takes about 12 hours, stage 2 about 6.

### Check the extraction mirror

```bash
./run.sh build --blocks 128 --interleaves TILE --chunks 100 --dtypes uint32 --no-baseline
./run.sh parity
```

This builds one variant, then compares the mirrored extraction loop with the API's own function.

### Stage 1

```bash
stages/stage1.sh     # every layout, in batches by block size; writes results/report-stage1.md
```

### Stage 2

```bash
stages/stage2.sh                     # the six candidates; writes results/report-stage2.md
stages/stage2-uint32.sh <layout>     # the selected layout, e.g. b128-tile-c100
```

### Settings A and B

Optional, to compare local setups with the simulation. Build the layouts first, then:

```bash
./run.sh setup-a
./run.sh A --rounds 3 --only <names...>
./run.sh B --rounds 3 --only <names...>
```

### Report options

`analyze.py --select` applies the rule. `--max-view-loss` (default 0.75 s per step), `--max-size-ratio` (1.5) and
`--max-files` (5,000) set its limits.
