# cog_stac_pipeline

Converts a multi-band GeoTiff dataset into a set of Cloud-Optimized GeoTIFFs (COGs), a STAC catalog, and a lookup dictionary that downstream applications can use to fetch data for a specific variable and timestep.

## What it does

A typical paleoclimate dataset stores years or centuries of data as one large multi-band GeoTiff per variable (e.g. `ppt.tif` with one band per year). This pipeline:

1. **Slices** each GeoTiff into smaller COG files (up to `max_bands_per_slice` bands each), making them efficient to serve over HTTP range requests.
2. **Builds a STAC catalog** — a standard JSON tree that lists every file, its spatial extent, its time range, and raster statistics.
3. **Builds a lookup dictionary** — a flat JSON map from `variable → ISO timestep → {file path, band index}`, so apps can answer "give me `ppt` at year `0850`" without parsing STAC.
4. **Validates and patches `metadata.yml`** — the shared metadata file that describes all SKOPE datasets. If required fields (CRS, transform, min/max, time resolution) are missing, they are extracted from the data and written back. If they are present but don't match the data, the pipeline raises an error.

## Outputs

Given `dataset_name = "paleocar_v3"` and `input_dir = "data.nosync"`, the pipeline writes:

```
data.nosync/
└── paleocar_v3/
    ├── cogs/
    │   └── <var>/
    │       ├── <var>_1.tif   ← COG slice (bands 1–100)
    │       ├── <var>_2.tif   ← COG slice (bands 101–200)
    │       └── ...
    ├── stac/
    │   ├── catalog.json
    │   └── <var>/
    │       ├── collection.json
    │       └── <var>_1/
    │           └── <var>_1.json   ← STAC Item
    │           ...
    └── lookup.json
```

**`lookup.json` example:**
```json
{
  "ppt": {
    "0103": { "file": "paleocar_v3/cogs/ppt/ppt_1.tif", "bidx": 1 },
    "0104": { "file": "paleocar_v3/cogs/ppt/ppt_1.tif", "bidx": 2 },
    ...
  }
}
```

The file paths in `lookup.json` are relative to `input_dir` (i.e. `data.nosync/`).

## S3 support ⚠️ WIP — not fully validated

`input_dir` and the derived output paths accept `s3://bucket/prefix` in addition to local paths. Set them in `main.py` as you would any other path:

```python
input_dir = "s3://my-bucket/datasets/paleocar_v3/input"
```

COG reads and writes go through GDAL's `/vsis3/` virtual filesystem (converted transparently by `fs_utils`). The STAC catalog and `lookup.json` are written via boto3.

**Known uncertainties before relying on this in production:**
- `pystac.utils.make_relative_href` behavior with `s3://` URIs has not been tested — if it misbehaves, asset hrefs in all STAC items will be wrong.
- `_S3StacIO` uses `write_text_method` / `read_text_method` hooks that require pystac ≥ 1.4 and the `stac_io` kwarg on `catalog.save`. Verify with `python -c "import pystac; print(pystac.__version__)"`.
- GDAL and boto3 use **separate credential chains**. An IAM role or `~/.aws/credentials` file satisfies boto3 but not necessarily GDAL — set `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` env vars (or `gdal.SetConfigOption`) to cover both.

A dry run against a small single-variable, two-band GeoTiff on S3 is the recommended way to surface any of these issues before a full dataset run.

`metadata.yml` always lives on the local filesystem — it is a shared config file, not a dataset output.

## Prerequisites

- GDAL ≥ 3.12 (uses `gdal.Run` pipeline commands)
- Python packages managed by `uv` from `pyproject.toml`: `pystac`, `rio_stac`, `python-dateutil`, `pyyaml`
- `boto3` — only required when using S3 paths

## Configuration

Configuration is read from environment variables by `PipelineConfig.from_env()`:

| Environment variable | Description |
|---|---|
| `INPUT_DIR` | Directory containing the source `.tif` files |
| `INPUT_MANIFEST_PATH` | Optional local YAML manifest mapping variable IDs to local or `s3://` TIFF URIs; when set, it replaces `INPUT_DIR` discovery |
| `OUTPUT_DIR` | Directory where `cogs/`, `stac/`, and `lookup.json` are written. Defaults to `<INPUT_DIR>/<DATASET_NAME>` |
| `DATASET_NAME` | Dataset ID — must match an entry in `metadata.yml` |
| `METADATA_FILE_PATH` | Path to the shared YAML metadata file |
| `TRUNC_TO_UINT16` | Cast values to UInt16 and set nodata=65535 (reduces file size for integer datasets) |
| `PREFLIGHT_ONLY` | Validate manifests and band-derived temporal coverage without creating output |
| `MAX_BANDS_PER_SLICE` | Maximum number of time bands per COG slice |
| `DATASET_START_DATETIME` | ISO datetime of the first band in the source GeoTiff |
| `DATASET_TIME_DELTA` | JSON object with `dateutil.relativedelta` keys (e.g. `{"years": 1}`) |

An input manifest is a YAML mapping with one explicit source TIFF per variable:

```yaml
dataset_id: paleocar_v3
variables:
  - id: ppt_annual
    uri: s3://skope/paleocar_v3/ppt_annual/prediction_scaled.tif
  - id: gdd_cotton_annual
    uri: /srv/datasets-import/paleocar_v3/gdd_cotton_annual/cube.tif
```

The URI may be a local path or an `s3://` object. The ID, rather than the source
filename, becomes the lookup key, COG directory name, and STAC collection ID.
This permits nested S3 sources and local production files to participate in one
coherent dataset build.

## Running

From the `skope-api/timeseries/ingest/` directory:

```bash
uv run cog-stac-pipeline
```

From the repository root, build and run the containerized pipeline:

```bash
docker compose --profile ingest build ingest
docker compose --profile ingest run --rm ingest
```

or:

```bash
make ingest
```

COG files are skipped if they already exist on disk, so the pipeline is safe to re-run after interruption.

## metadata.yml

The pipeline reads from and optionally writes to a shared `metadata.yml` (default: in the working directory, but configurable via `metadata_file_path`). Each dataset entry must have an `id` matching `dataset_name` and a `variables` list with an entry for each variable found in `input_dir`.

The pipeline will **add** missing fields (`crs`, `transform`, `timespan.period.gte`, `timespan.period.lte`, `timespan.resolution`, `variables[*].min`, `variables[*].max`) and **raise an error** if any existing field conflicts with what it finds in the data. Temporal endpoint validation uses each source raster's band count and runs before output directories are created.

When an input manifest is used, its `dataset_id` must match `DATASET_NAME` and
its variable IDs must exactly equal the IDs in the selected metadata dataset.
The pipeline checks this contract before opening any source rasters.

**Minimal required structure before first run:**
```yaml
- id: paleocar_v3
  description: "..."
  variables:
    - id: ppt
    - id: gdd
```

## Module overview

| File | Responsibility |
|---|---|
| `main.py` | Configuration, directory setup, per-variable orchestration, saving outputs |
| `cog_builder.py` | GDAL operations: band selection → temp GeoTiff → COG conversion |
| `stac_builder.py` | STAC item/collection/catalog creation, lookup dict population, `process_variable` orchestrator |
| `metadata.py` | YAML loading, and field-level validation/patching for timespan, CRS, transform, and min/max |
| `datetime_utils.py` | ISO key formatting, date range generation, `relativedelta` helpers |
| `fs_utils.py` | Filesystem abstraction: path detection, VSI conversion, directory creation, file listing, and text writes for both local and S3 paths |
