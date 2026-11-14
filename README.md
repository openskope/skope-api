# skope-api

[![DOI](https://zenodo.org/badge/338436138.svg)](https://zenodo.org/badge/latestdoi/338436138)
[![skope-api build](https://github.com/openskope/skope-api/actions/workflows/test.yml/badge.svg)](https://github.com/openskope/skope-api/actions/workflows/test.yml)

Backend services for SKOPE: dataset metadata, map tiles, and time series
extracted from SKOPE datasets. See https://api.openskope.org/docs for API
details and examples. The [skopeui](https://github.com/openskope/skopeui) app
is its client.

The repository has two parts, each with its own guide:

- **[`timeseries/ingest/`](timeseries/ingest/README.md):** the release tool,
  which turns a dataset's declaration into an immutable dataset release. Read
  it to add or update a dataset.
- **[`timeseries/app/`](timeseries/app/README.md):** the API, which serves the
  releases an environment pins, and how to deploy it. Read it to choose what an
  environment serves, how it is shown, and to deploy, verify, or roll back.

Decisions are in [`.agents/decisions/`](.agents/decisions/), and the rules both
parts follow are in the [dataset release](docs/specs/dataset-release-v1.md) and
[release consumption](docs/specs/release-consumption-v1.md) specifications.

## Development

Build and run the API, Redis, and TiTiler locally:

```bash
make deploy-dev
```

Try out the API, and list the datasets it serves with their variables, time
axes, and display settings:

```bash
http GET localhost:8001/docs
http GET localhost:8001/metadata
```

Run the tests:

```bash
make test
```

## Dataset releases

### What a release is

A release is an immutable package of **one dataset**: its COGs, the STAC
metadata describing them, a generated overview, and a manifest. One dataset,
one release, one manifest. Releases of different datasets are built, published,
and deployed independently, so changing or breaking one dataset never touches
the others.

Getting data into the app takes two separate acts:

| Act | Who | Produces | Undone by | Guide |
| --- | --- | --- | --- | --- |
| **Build** | Release author | Immutable bytes at one release path | Not deploying the release | [ingest](timeseries/ingest/README.md) |
| **Deploy** | App developer | An updated pin for one environment | A rollback commit | [app](timeseries/app/README.md) |

Publishing a release does not make the app serve it. A release is only served
once an environment's pin names it. Releases are not listed in any public
catalog.

### Where each fact lives

| If you need… | Trust… |
| --- | --- |
| Descriptions, citations, units, category meanings | The STAC Collection (compiled from `curated.yml`) |
| Grid, nodata, data type, scale, offset, statistics | The COG headers; STAC carries checked copies |
| Which timesteps exist | STAC (`cube:dimensions.time.values`, Band names) |
| Whether a release is complete, and its integrity | `release-manifest.json` |
| A one-file summary of all of the above | `overview.yml`, generated from STAC; never edited |
| Colours, ranges, ordering, map view | Display files in `deploy/display/`; never in a release |
| Which releases an environment serves | That environment's pin file, `deploy/releases/<env>.yml` |

Generated copies are checked against their authority on every build. When two
disagree, the build fails rather than picking one.

### The flow

```text
  datasets/<id>/curated.yml       datasets/<id>/source-manifest.yml (+ release: block)
            │                                │
            └──────────────┬─────────────────┘
                           ▼
                  fetch: copy the sources into the local mirror and run
                  any prepare step (optional unless a source is prepared)
                           ▼
                  preflight: validate both, inspect and checksum every
                  source, freeze the declaration
                           ▼
                  build in staging: write COGs → reopen and inspect bytes
                  → STAC → overview.yml → validate everything
                  → release-manifest.json last
                           ▼
                  publish: atomic rename into the release root
                           ▼
        <release-root>/<release-id>/        (immutable from here on)
  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─│─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─   ingest ↑   app ↓
                           ▼
                  pin in deploy/releases/<env>.yml  +  display files
                           ▼
                  promotion: full verification, recreate API and TiTiler
```

### What is in a release

```text
<release-id>/
  release-manifest.json     # written last; its presence means "complete"
  collection.json           # the dataset's STAC Collection, the release root
  overview.yml              # generated summary, read by people and by the app
  items/
    <item-id>.json          # one Item per temporal chunk (temporal datasets only)
  cogs/
    <variable-id>/
      <item-id>.tif         # temporal: one COG per variable per chunk
    <variable-id>.tif       # static: one COG per variable, no Items
```

The layout is closed: any other file fails validation. Internal links are
relative, so a release can be moved or copied as a directory.

A dataset is either a **temporal cube** (`lbda_v2`, `owda_v1`, `paleocar_v2`,
`paleocar_v3`) or a **static raster**, which has no Items, no time axis, and no
chunking. The API serves temporal cubes only.

### Two identifiers per release

- The **release ID** is a label a person assigns:
  `<dataset-id>-r-YYYY.MM.DD`, or `…-N` (from `-2`) for a later declaration of
  the same dataset on the same date. Example: `paleocar_v3-r-2026.11.04`.
- The **declaration digest** is computed: a SHA-256 of the dataset's resolved
  inputs (`curated.yml`, the resolved source manifest including its `release:`
  block, and the source checksums).

A release ID is bound to exactly one declaration digest, forever. The release ID
is not a dataset version: `paleocar_v3` stays version `3` across any number of
releases that only repackage it.

## Extraction job durability

Extraction requests run as background tasks in the API worker that accepts them.
Redis retains job status, results, and internal analysis data for 24 hours, but it
is not an execution queue: a worker restart does not resume an in-flight job.
Clients should treat a missing job or a job that remains nonterminal across a
deployment as abandoned and submit a new extraction request.

This execution model is suitable for the current short, bounded extractions. A
separate durable worker queue is required before offering restart-safe or
resumable jobs.
