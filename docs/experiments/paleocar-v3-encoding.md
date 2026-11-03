# PaleoCAR v3 COG encoding experiment

- Experiments: EXP-002, EXP-003 and EXP-004 of the
  [Dataset Release Specification v1](../specs/dataset-release-v1.md)
- Run: stage 1 on 2026-10-08/09; stage 2 and the data type check on 2026-11-02/03
- Harness: [`experiments/paleocar-v3-encoding/`](../../experiments/paleocar-v3-encoding/)
- Result: **`BLOCKSIZE=128`, `INTERLEAVE=TILE`, 250 timesteps per file, `UInt16`**

## Question

Which COG layout should PaleoCAR v3 releases use? The layout is the combination of block size, interleave, number of
timesteps per file (chunk size) and data type. It decides how fast the two main read paths are:

- **Map views:** TiTiler serves map tiles, each reading one band (one year) over a wide window.
- **Extraction:** the API's time-series extraction reads every band of a file over a small window (the study area).

They pull in opposite directions: extraction favours small blocks, map views favour fewer blocks. Extraction is the
slower operation and the one that holds server capacity (one job per API worker), so the selection rule ranks by
extraction and only checks that map views stay within a limit.

## Scope

The conclusions apply to releases served **from local disk** on a host like the one below. This is how SKOPE serves
releases for the foreseeable future.

If releases are ever served from cloud object storage (for example S3), run a new experiment under those conditions
before reusing this layout: request latency, header size, the number of files, header caching between slider steps,
and per-request cost barely register on local disk but could change the best layout there.

Also outside this experiment: categorical variables (`paleocar_v2/maize_farming_niche`) and static rasters such as
SRTM.

## Target environment and its simulation

The target host is DRAC Arbutus `skope-01`: 8 vCPU, 16 GB RAM and a Cinder volume, shared by the API (4 gunicorn
workers, each running at most one extraction at a time), TiTiler (2 workers), Redis and the UI.

There is no access to that host for benchmarking, so the experiment ran on a laptop in a simulation of it (setting C
in the harness):

- Linux containers: the API image for extraction, the TiTiler image for tiles, the ingest image for builds.
- Variants on a Docker named volume, containers limited to `--cpus 8`, `GDAL_CACHEMAX=819`.
- **Cold reads:** the page cache is dropped (`posix_fadvise(DONTNEED)`) before every measurement.
- A fresh process for every measurement, for per-operation peak memory and no shared GDAL state.
- Tiles served by 2 processes, as TiTiler's 2 workers.
- Extraction through a mirror of the API's read loop: 10 threads and the API's 1M-cell band chunking. The mirror is
  checked against the API's own function (`run.sh parity`).
- Block-device reads and sectors (`/proc/diskstats`) and process I/O (`/proc/self/io`) recorded with every
  measurement.
- The dev stack stopped during measurement.

| Component | Version |
| --- | --- |
| Laptop | Apple M2, 8 cores, 24 GB; macOS 26.7.1; Docker 29.9.2 with 8 CPUs and 7.75 GiB |
| Builds (ingest image) | GDAL 3.12.0 |
| Extraction (API image) | Python 3.13.18, rasterio 1.5.1, GDAL 3.12.4 |
| Tiles (TiTiler image) | Python 3.14.4, rasterio 1.5.0, GDAL 3.12.1, rio-tiler 9.0.6 |

A local disk is faster than a network volume, so absolute times on the real host will be higher. The comparison
between layouts is what carries over. The bytes-read figures show which layouts would lose more on slower storage.

## Method

### Common to all runs

- **Variable:** `paleocar_v3/ppt_annual`, the release's source file, on the full 1560 × 1440 grid.
- **Build options:** the release build's own COG options (ZSTD level 9, predictor 2, average overviews, statistics,
  `BIGTIFF=IF_SAFER`, `SPARSE_OK=FALSE`), varying only block size, interleave, chunk size and data type.
- **Data types:** `UInt32` (as the source) and `UInt16` with nodata 65535. Every valid `ppt_annual` value fits in
  `UInt16`.
- **Extraction areas:** a point, and small (20 × 20 cells), medium (100 × 100) and large (1013 × 542) polygons.
- **Map views:** the time to serve every tile of a view, at MapLibre's tile size of 128, over a 12-step slider
  sequence (12 adjacent years). Two kinds of view:
  - **framed on each area,** as skopeui frames a study area on its visualize page (a map of about 640 × 550 px, 30 px
    padding, map zoom at most 10): 30 to 36 tiles;
  - **the whole region** in a 1280 × 800 map, at the initial view (zoom 7, 30 tiles) and at the first full-resolution
    zoom (zoom 8, 70 tiles).
- **Tiles** read a one-band view of the file (`vrt://…?bands=<band>`), as the API asks TiTiler for them. Without it,
  GDAL's warper decodes every band of the file for each tile.
- **Empty years:** 0417–0589 hold no data. Windows avoid them; full-range extractions read them, as the API does.
- **Noise:** each result is the median of 3 rounds. Its standard error comes from the spread of the runs (median
  absolute deviation). Two layouts tie when their difference is within 2 standard errors of it.
- **Identical outputs:** every layout must give the same results. Each extraction is hashed from its per-timestep
  mean and median, rounded to 3 decimals; each map view from the rendered pixels and nodata mask of every tile in the
  sequence.

### Stage 1: choosing candidates

All 48 combinations of block size 64, 128, 256 or 512, interleave `BAND` or `TILE`, chunk size 25, 100 or 250, and
`UInt16` or `UInt32`, over a 500-year window (1401–1900) that every chunk size divides into whole files. The current
release's layout, `b512-band-c100-uint32`, is one of them and was measured in every batch as a control. Stage 1
measured whole-region map views and extractions of 10, 100 and 500 years.

The window holds a chunk-250 layout in only 2 files, too few to keep 10 extraction threads busy, which holds those
layouts back on the large area. Stage 1 therefore only picked candidates, one per role, from the raw `UInt16`
measurements:

| Role | Candidate | Stage 1 evidence |
| --- | --- | --- |
| Fastest map views | `b512-band-c25-uint16` | 1.22 s per step, averaged over the two whole-region views |
| Fastest extraction | `b128-tile-c100-uint16` | 2.42 s summed over the four areas |
| Fastest extraction, less memory | `b64-tile-c100-uint16` | 2.40 s, with 790 MB peak on the large area against 839 MB |
| Leading extractors at chunk 250 | `b128-tile-c250-uint16`, `b64-tile-c250-uint16` | fastest on the point, small and medium areas |
| Lowest memory | `b64-tile-c25-uint16` | 579 MB peak on the large area |

Other findings from stage 1:

- `UInt16` was about 11% smaller than `UInt32`, and equal or faster in every operation.
- 512-pixel blocks gave the fastest whole-region map views and the slowest extraction at every chunk size.
- At equal chunk size, `TILE` used the same or less memory than `BAND`. Memory grew mainly with chunk size.

### Stage 2: deciding

The 6 candidates were built over all 1898 years (0103–2000) and measured in 3 rounds: map views framed on each area,
whole-region map views, and extractions of 10, 100, 500 and all years on each area.

The selection rule was fixed before the run:

1. Rank by the sum of the median full-range extraction times over the four areas.
2. Break ties by faster map views framed on the areas, then the smaller release, then fewer files.
3. Select the first layout whose map views framed on each area are at most **0.75 s per slider step** slower than
   the fastest candidate's on the same area.
4. The selected layout must have at most 5,000 files for a 12-variable release and a release size at most 1.5 times
   the smallest candidate's, and must be readable by the deployed TiTiler and API.

Whole-region views, shorter extractions, memory, reads and bytes are reported, not scored.

## Results

### Extraction over the full range

Median seconds ± standard error, in the selection order:

| Layout | Total | Point | Small | Medium | Large |
| --- | --- | --- | --- | --- | --- |
| **b128-tile-c250-uint16** | **6.03 ± 0.14** | 0.16 | 0.16 | 0.39 | 5.32 |
| b64-tile-c250-uint16 | 6.64 ± 0.23 | 0.14 | 0.15 | 0.39 | 5.95 |
| b128-tile-c100-uint16 | 7.85 ± 0.10 | 0.19 | 0.19 | 0.45 | 7.02 |
| b64-tile-c25-uint16 | 8.33 ± 0.63 | 0.24 | 0.27 | 0.47 | 7.35 |
| b64-tile-c100-uint16 | 8.12 ± 0.04 | 0.17 | 0.18 | 0.46 | 7.31 |
| b512-band-c25-uint16 | 13.02 ± 0.15 | 1.02 | 0.81 | 0.88 | 10.30 |

- `b128-tile-c250-uint16` leads `b64-tile-c250-uint16` by 0.61 s, outside the tie band of 0.54 s.
- `b64-tile-c25-uint16` and `b64-tile-c100-uint16` tie; the tie goes to `b64-tile-c25-uint16`'s slightly faster map
  views.
- 512-pixel blocks are about twice as slow overall, and take about 1 s even for a point.

### Map views

Median seconds per slider step:

| Layout | Point | Small | Medium | Large | Whole region, z7 | Whole region, z8 |
| --- | --- | --- | --- | --- | --- | --- |
| b128-tile-c250-uint16 | 0.38 | 0.38 | 0.36 | 0.49 | 1.60 | 0.92 |
| b64-tile-c250-uint16 | 0.38 | 0.37 | 0.38 | 0.50 | 1.90 | 0.97 |
| b128-tile-c100-uint16 | 0.30 | 0.30 | 0.30 | 0.43 | 1.52 | 0.81 |
| b64-tile-c25-uint16 | 0.30 | 0.29 | 0.30 | 0.43 | 1.80 | 0.87 |
| b64-tile-c100-uint16 | 0.30 | 0.30 | 0.30 | 0.44 | 1.75 | 0.81 |
| b512-band-c25-uint16 | 0.30 | 0.30 | 0.30 | 0.45 | 1.10 | 0.84 |

On views framed on the areas, every candidate is within 0.08 s per step of the fastest, so the 0.75 s limit didn't
bind. The layouts differ most on the whole-region view at zoom 7, where 512-pixel blocks are fastest (1.10 s against
1.60 s for the selected layout).

### Memory, reads and release

| Layout | Peak, large area | 4 × large | Disk reads, large | Read, large | Release (12 variables) | Files |
| --- | --- | --- | --- | --- | --- | --- |
| b128-tile-c250-uint16 | 1333 MB | 5.3 GB | 2,923 | 1,313 MB | 45.4 GB | 96 |
| b64-tile-c250-uint16 | 1402 MB | 5.6 GB | 7,189 | 1,386 MB | 47.7 GB | 96 |
| b128-tile-c100-uint16 | 1503 MB | 6.0 GB | 5,473 | 1,489 MB | 45.4 GB | 228 |
| b64-tile-c25-uint16 | 566 MB | 2.3 GB | 16,818 | 2,011 MB | 47.7 GB | 912 |
| b64-tile-c100-uint16 | 1466 MB | 5.9 GB | 13,246 | 1,501 MB | 47.7 GB | 228 |
| b512-band-c25-uint16 | 1146 MB | 4.6 GB | 1,313 | 2,586 MB | 43.4 GB | 912 |

- **4 × large** is four large-area extractions at once, one per API worker, computed from the single peak. Every
  candidate fits in the host's 16 GB next to the other services. Concurrent extractions were not measured.
- The selected layout reads half the bytes of 512-pixel blocks on the large area, so it should lose less than they
  would on a slower network volume.
- Every candidate is within the size and file limits.

### Selection

**`b128-tile-c250-uint16`**: the fastest full-range extraction, with a worst map-view loss of 0.08 s per step against
the fastest candidate, on the large area.

### UInt16 against UInt32 (EXP-004)

The selected layout was built in both data types and measured together:

| | UInt16 ÷ UInt32 |
| --- | --- |
| Release size | 0.89 (45.4 against 51.3 GB) |
| Full-range extraction | 0.92 (6.31 against 6.88 s) |
| Map views framed on the areas | 0.96 |
| Peak memory, large area | 0.96 (1333 against 1386 MB) |

Decoded results were identical in both data types. The `UInt16` run (6.31 ± 0.16 s) agrees with stage 2's 6.03 s
within noise. Per EXP-004, a variable may narrow to `UInt16` only after its preflight scan proves every valid value
representable.

### Identical outputs

Every extraction and every map view gave a single hash across all layouts and both data types, in both stages. The
check compares layouts with each other: the `UInt32` run is the comparison with the source data type, and stage 1,
over its 500-year window, the comparison with the current release's layout. It shows that the choice of layout
doesn't change what users get; it doesn't validate the data itself.

## Extraction time depends on range length

On the large area, the ranking depends on how many years are extracted:

| Years | b128-tile-c250 | b128-tile-c100 | b64-tile-c25 |
| --- | --- | --- | --- |
| 10 | 0.15 s | 0.17 s | 0.16 s |
| 100 | 1.24 s | 1.23 s | 0.36 s |
| 500 | 3.01 s | 1.38 s | 1.11 s |
| All 1898 | 5.32 s | 7.02 s | 7.35 s |

Extraction runs in parallel across files: a 100-year range is 1 file at chunk 250 or 100, but 4 files at chunk 25.
Over the full range, smaller chunks pay instead for opening more files.

The selection scores the full range because that is what skopeui extracts: its first extraction always covers the
dataset's full range. A shorter range reaches the extraction endpoint only when a user changes the range on the
visualize page, which today starts a new extraction. `/timeseries/analyze` can already cut a range out of a stored
extraction without reading raster files, and the visualize page should use it. With that change, the choice holds
for every range a user selects; without it, ranges of a few hundred years are up to 3.5 times slower than with
chunk 25.

## Build messages

Every full-range build prints `ERROR 1: VRTDataset>` 173 times, once per empty year. GDAL can't compute statistics
for a band with no valid pixels and records `STATISTICS_VALID_PERCENT=0` for it instead. The message is harmless;
release builds of PaleoCAR v3 will show it too.

## Deviations from the specification

The experiment differs from what EXP-002 to EXP-004, COG-012, AT-012, AT-027 and AT-031 asked for, as written when
the experiment ran, in the following ways.

- **Benchmark variable.** The specification names `paleocar_v2/gdd_may_sept` (`Int16`). The experiment used
  `paleocar_v3/ppt_annual` (`UInt32`), because PaleoCAR v3 is the dataset being released and its `UInt32` source is
  the one the `UInt16` question applies to.
- **Order of the factors.** EXP-002 picks chunk sizes first (1, 25, 100 and 250 in `BAND` with 512-pixel blocks), and
  EXP-003 then builds the best two chunk sizes in 8 layouts (`BAND` or `TILE`, block size 256 or 512). The experiment
  measured all 48 combinations of the three chunk sizes, the four block sizes 64 to 512, both interleaves and both
  data types at once, then rebuilt 6 candidates over the full range. The factors interact: 512-pixel blocks were the
  slowest for extraction at every chunk size, so choosing chunk sizes in that layout would have judged them in the
  worst one for extraction. Block sizes below 256 were added because an earlier round showed full-series extraction
  7 to 12 times faster with 128-pixel blocks than with 512 (warm reads). Chunk size 1 was left out: it means
  22,776 files for a 12-variable release.
- **Full range only for the candidates.** EXP-002 builds every variant over the full range. Stage 1 used a
  500-year window instead, because a full-range build takes 10 to 25 minutes per variant; stage 2 rebuilt the
  candidates over the full range, and selection uses only those measurements.
- **Selection priority.** EXP-003 prioritizes one-timestep map-tile latency, with guardrails on extraction and other
  measures. The experiment ranks by extraction and limits map views to 0.75 s per step slower than the fastest
  candidate, a rule set before stage 2. Once tiles read a one-band view of the file, map views framed on a study
  area differed by at most 0.08 s per step across the candidates, while extraction differed by a factor of two; and
  extraction is the operation that holds server capacity.
- **Warm reads.** Not measured. Releases are larger than the host's memory (about 45 GB against 16 GB), so cold
  reads are the representative case.
- **50-step slider sequences.** Not measured. A 12-step sequence gives the per-step cost, but stays within one file.
  A longer sequence would also show the cost of opening the next file, which matters most for 25-timestep files
  and least for the selected layout, which changes file every 250 steps.
- **Concurrency.** Tiles were served by 2 processes, as TiTiler's 2 workers. Extractions were measured one at a
  time; 4 at once, one per API worker, were estimated from the single-extraction peak memory (5.3 GB for the
  selected layout, well within 16 GB), and their time under contention for CPU and disk was not measured.
- **Range requests.** The specification asks for range-request bytes and counts. Releases are served from local
  disk, so the experiment reports block-device reads and bytes read instead.
- **Categorical variable.** EXP-002 requires testing one (`paleocar_v2/maize_farming_niche`). Not tested: PaleoCAR v3
  has only continuous variables. A categorical dataset needs its own check before it adopts this layout. The
  optional `lbda_v2` file-open control was not used.
- **Data type (EXP-004, AT-031).** Measured on `ppt_annual` only, and for estimates, not uncertainty products.
  Extraction results (per-timestep mean and median) and tile pixels were compared; the COGs' embedded band
  statistics were not. As EXP-004 is written, this measurement supports narrowing `ppt_annual`; another variable
  needs its own measurement.
- **Additions.** A point area was measured alongside the three polygons, and map views were also framed on each
  area as skopeui frames a study area.

## Reproducing

From [`experiments/paleocar-v3-encoding/`](../../experiments/paleocar-v3-encoding/), with the dev stack stopped and
`SOURCE` and `WORK` set as the harness [README](../../experiments/paleocar-v3-encoding/README.md) describes:

```bash
./run.sh build --blocks 128 --interleaves TILE --chunks 100 --dtypes uint32 --no-baseline
./run.sh parity                           # the extraction mirror against the API's function
stages/stage1.sh                          # stage 1, about 12 hours
stages/stage2.sh                          # stage 2, about 6 hours
stages/stage2-uint32.sh b128-tile-c250    # the data type check, about 30 minutes
```

Each run writes its measurements and a generated report to `results/`, which is not committed.
