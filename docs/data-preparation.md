# Dataset Preparation Runbook

This runbook prepares the dataset tree mounted at `/data` by the API and
TiTiler. Development uses `./cog-input`; staging and production use
`/srv/datasets`. Preparing data and deploying the API are separate operations:
build and validate data outside the live dataset directory, then promote it.

The canonical processor is the containerized COG/STAC pipeline in
`timeseries/ingest`. The older scripts in
[`openskope/skope-datasets`](https://github.com/openskope/skope-datasets) are
useful provenance, but target the retired `/projects/skope/datasets` layout and
do not generate the `lookup.json` required by the current API.

## Required dataset contract

Each published dataset has this shape beneath the mounted storage root:

```text
<storage-root>/
└── <dataset_id>/
    ├── lookup.json
    ├── cogs/
    │   └── <variable_id>/
    │       └── <variable_id>_<slice>.tif
    └── stac/
        └── catalog.json
```

`lookup.json` maps each variable and ordered ISO timestep to a COG path and
one-based band index. Its paths are relative to `<storage-root>`, for example:

```json
{
  "ppt_annual": {
    "0103": {
      "file": "paleocar_v3/cogs/ppt_annual/ppt_annual_1.tif",
      "bidx": 1
    }
  }
}
```

The dataset ID and variable IDs must exactly match the registry used by the
target environment. Keep these files aligned:

- `timeseries/ingest/metadata.yml`, used to validate pipeline inputs
- `timeseries/metadata.yml`, mounted as the development API registry
- `deploy/metadata/staging.yml` and `deploy/metadata/prod.yml`, built into the
  deployed API images

Do not promote output when an input filename, lookup key, or registry variable
ID differs.

## Prepare PaleoCAR v3 source files

The public source is the [`skope` S3 website](http://skope.s3-website-us-west-2.amazonaws.com/).
The bucket contains twelve PaleoCAR v3 variable groups. Each group contains
`prediction.tif`, `prediction_scaled.tif`, `pi_deviation.tif`, and
`pi_deviation_scaled.tif`; individual files are roughly 1.2–2.9 GB.

The checked-in manifest at
`timeseries/ingest/manifests/paleocar_v3.yml` selects the
`prediction_scaled.tif` product from all twelve variable groups:

| Pipeline variable ID | Public S3 object |
| --- | --- |
| `gdd_cotton_annual` | `paleocar_v3/gdd_cotton_annual/prediction_scaled.tif` |
| `gdd_cotton_maysept` | `paleocar_v3/gdd_cotton_maysept/prediction_scaled.tif` |
| `gdd_cotton_wateryear` | `paleocar_v3/gdd_cotton_wateryear/prediction_scaled.tif` |
| `gdd_maize_annual` | `paleocar_v3/gdd_maize_annual/prediction_scaled.tif` |
| `gdd_maize_maysept` | `paleocar_v3/gdd_maize_maysept/prediction_scaled.tif` |
| `gdd_maize_wateryear` | `paleocar_v3/gdd_maize_wateryear/prediction_scaled.tif` |
| `gdd_wheat_annual` | `paleocar_v3/gdd_wheat_annual/prediction_scaled.tif` |
| `gdd_wheat_maysept` | `paleocar_v3/gdd_wheat_maysept/prediction_scaled.tif` |
| `gdd_wheat_wateryear` | `paleocar_v3/gdd_wheat_wateryear/prediction_scaled.tif` |
| `ppt_annual` | `paleocar_v3/ppt_annual/prediction_scaled.tif` |
| `ppt_maysept` | `paleocar_v3/ppt_maysept/prediction_scaled.tif` |
| `ppt_wateryear` | `paleocar_v3/ppt_wateryear/prediction_scaled.tif` |

This extends the nine groups requested in
[`openskope/planning#40`](https://github.com/openskope/planning/issues/40) with
the three additional groups present in the bucket. The manifest is the
operational source map; `timeseries/ingest/metadata.yml` is the descriptive
dataset record. Their dataset ID and complete variable-ID set must match
exactly. The pipeline verifies that relationship before it opens any raster.

The default ingest profile streams these public objects directly through GDAL,
so downloading them first is not required:

```bash
make ingest
```

To transform existing production TIFFs instead, copy the manifest and replace
each `uri` with the corresponding local file path. Preserve the same dataset
and variable IDs. Point `INPUT_MANIFEST_PATH` at that file when running the
container. Local and S3 sources may be mixed in one manifest.

Record the selected manifest with the release. When inputs are materialized
locally, also record SHA-256 checksums; S3 multipart ETags are not MD5 checksums:

```bash
sha256sum cog-input/*.tif
```

## Run the pipeline

The checked-in Compose profile configures PaleoCAR v3 as annual data beginning
in 0103 CE, loads the twelve-variable manifest, slices each input into COGs of
at most 100 bands, converts values to UInt16, and writes output under
`timeseries/ingest/output/paleocar_v3`:

```bash
make ingest
```

The pipeline validates spatial and temporal metadata, creates COG slices and a
self-contained STAC catalog, and generates `lookup.json`. Existing COG slices
are skipped, so an interrupted run can be resumed. Remove or isolate stale
output before intentionally changing input data or ingest parameters; skipped
files are not rebuilt automatically.

## Validate generated output

Run the ingest tests and check the generated package before promotion:

```bash
make test-ingest
python -m json.tool timeseries/ingest/output/paleocar_v3/lookup.json >/dev/null
test -s timeseries/ingest/output/paleocar_v3/stac/catalog.json
find timeseries/ingest/output/paleocar_v3/cogs -type f -name '*.tif'
```

Also verify that:

1. Every intended registry variable occurs in `lookup.json`, and no unexpected
   variable is present.
2. Each variable begins at `0103`, ends at `2000`, and has ordered annual keys.
3. Every lookup file exists beneath the generated storage root and every band
   index is valid.
4. `gdalinfo` reports the expected CRS, transform, extent, data type, nodata
   value, tiling, compression, and band count for representative first and last
   slices.
5. Metadata changes made by the pipeline are reviewed and deliberately copied
   into all applicable API registries.

For an end-to-end development check, copy the generated dataset package—not
the raw source TIFFs—into the local storage root, deploy development, and test
metadata, a representative tile, and a small extraction:

```bash
rsync -a timeseries/ingest/output/paleocar_v3/ cog-input/paleocar_v3/
make deploy-dev
curl --fail --show-error http://127.0.0.1:8001/metadata
```

## Promote to staging and production

Promotion must preserve the validated directory exactly. Record the source
commit, pipeline configuration, source URLs, SHA-256 checksums, output size,
and processing date with the release.

1. Assemble all generated `<dataset_id>` directories into one immutable release
   root such as `/srv/datasets/releases/skope-r-2026.09.12`. Do not mix dataset
   directories from different releases in the live mount.
2. Repeat the lookup, COG, STAC, and `gdalinfo` checks against that complete root.
3. Retain the current release directory as the rollback copy.
4. Deploy staging with
   `make deploy-staging DATASET_RELEASE_ROOT=/srv/datasets/releases/skope-r-2026.09.12`.
5. Deploy the matching API registry using the
   [deployment runbook](deployment.md), then test metadata, a representative
   tile, and a small extraction through the public hostname.
6. Promote the same validated artifact to production; do not rerun the pipeline
   independently for production.

If verification fails, redeploy the previous release root and matching API
commit. Dataset files and API registry versions must be rolled back together.

After staging passes, promote the same immutable directory by passing its exact
CalVer path to the production deployment. Staging and production do not use a
mutable `current` symlink: the selected path remains explicit in the deployment
command and container configuration, and rollback selects the prior immutable
path in the same way.

Exact host copy and atomic-switch commands are intentionally delegated to
`comses/infrastructure`; this repository does not define filesystem ownership,
release-directory naming, or whether staging and production share storage.

## Migrate the complete legacy tree

Transformation is paused pending approval of the proposed
[dataset release specification](specs/dataset-release-v1.md). Do not run
`make migrate-legacy-data` while this pause is in effect. The legacy command
still includes PRISM for reproducibility, but PRISM is outside the cleanroom
migration scope.

The command is retained for reference. For a legacy tree staged at
`/srv/ingest/incoming/skope` containing `datasets/lbda_v2`,
`datasets/paleocar_v2`, `datasets/prism`, and `datasets/srtm`, it is:

```bash
make migrate-legacy-data \
  LEGACY_DATA_ROOT=/srv/ingest/incoming/skope \
  MIGRATED_DATA_ROOT=/srv/datasets/releases/skope-r-2026.09.12 \
  MIGRATION_SCRATCH_ROOT=/srv/dataset-migration-scratch/skope-r-2026.09.12
```

While staging or reviewing the source data, run the same checks without
creating output:

```bash
make preflight-legacy-data
```

The default source root is `/srv/ingest/incoming/skope`. Override
`LEGACY_DATA_ROOT` only when validating a differently staged copy.

When resumed, the migration target reads legacy data without modifying it and
refuses to use a non-empty output directory. It transforms legacy cubes for
LBDA v2, PaleoCAR v2, PRISM, and SRTM, and builds the twelve-variable PaleoCAR
v3 package from the public S3 manifest. Output packages and the metadata
populated with observed raster properties are written beneath
`MIGRATED_DATA_ROOT`. Large temporary TIFFs are written to the host-backed
`MIGRATION_SCRATCH_ROOT` rather than Docker's container storage.

Before creating the output or scratch directories, the migration opens every
source in metadata-only mode and verifies that its band-derived temporal end
matches the declared dataset period. Transformation starts only after all
legacy and S3 inputs pass this preflight.

PRISM is migrated for completeness but is not currently published by the API
registry. Review and add its generated metadata deliberately before exposing
it. Validate the complete generated root using the checks above, then mount
that exact directory in staging with `DATASET_RELEASE_ROOT`. The `_migration`
directory is release provenance and is ignored by the API; dataset packages
remain at the release root as required by `/data/<dataset_id>/lookup.json`.
