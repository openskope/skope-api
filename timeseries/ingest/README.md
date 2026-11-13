# Building dataset releases

`skope_release` turns a dataset's declaration into an immutable release: COGs,
the STAC metadata describing them, a generated overview, and a manifest. This
guide is for **release authors**: preparing a dataset's metadata, building a
release, and publishing it. Serving a release is the app's side, in the
[app README](../app/README.md). The [root README](../../README.md#dataset-releases)
explains what a release is, where each fact lives, and how the two sides fit
together; read it first.

The rules are in the [dataset release specification](../../docs/specs/dataset-release-v1.md);
requirement IDs such as OBS-011 refer to it.

## Commands

Run every command from the repository root. Each one rebuilds its image
first.

| Command | Does |
| --- | --- |
| `make release-fetch ARGS="<dataset>…"` | Copy sources into the source mirror and run prepare steps |
| `make release ARGS="preflight <dataset>…"` | Validate declarations and sources; write nothing |
| `make release ARGS="build <dataset>… --release <dataset>=<release-id>"` | Build and publish one release per dataset |
| `make release ARGS="verify /releases/<release-id>"` | Verify a published release against its manifest |
| `make release ARGS="verify-reproducible /releases/<release-id>"` | Rebuild a release in scratch and compare it byte for byte |
| `make release-pin MANIFEST=<file> PIN_ENV=<env>` | Show the edits that pin a release; add `WRITE=1` to apply them |
| `make test-ingest` | Run the release tool's tests |

Paths come from `RELEASE_SOURCE_MIRROR`, `RELEASE_ROOT`, and
`RELEASE_AUDIT_ROOT`; pass the same values to every command for one release.
Inside the container they are `/sources`, `/releases`, and `/audit`.

## Before building

### Check `curated.yml`

Each dataset has one `curated.yml`, at
`timeseries/ingest/datasets/<dataset-id>/curated.yml`. It holds what a person
knows about the data, and nothing that can be measured or that only affects
display.

It must contain:

- [ ] **Dataset:** title, description, license, providers, and a region name
  (such as `"Southwestern USA"`).
- [ ] **Each variable:** title, description, and unit (or explicitly
  `unitless`).
- [ ] **Each categorical variable:** what every encoded value means.
- [ ] **Temporal datasets:** calendar, precision, time origin, endpoint
  inclusion, what a timestep means, and whether a timestamp is an instant or an
  aggregation period.

It may also contain citations and DOIs, contacts (as an organization-maintained
HTTPS URL), method summaries, uncertainty explanations, valid ranges, ORCID and
ROR identifiers, and a category per variable.

Things to check:

- [ ] **No required field is "unknown".** If a product's meaning is not known,
  leave the product out of the release rather than shipping a placeholder.
- [ ] **No observed facts:** no CRS, transform, shape, extent, nodata, data
  type, statistics, checksums, or sizes. The build measures these.
- [ ] **No presentation:** no colormaps, colours, visualization ranges, legends,
  ordering, or map view. These go in the
  [display files](../app/README.md#add-display-entries).
- [ ] **Temporal values are quoted strings:** `"0103"`, not `0103`.
- [ ] **No personal contact details** copied from legacy metadata.

For a dataset whose metadata predates this format, the answers come from the
dataset creator; see spec §20.2 for the questions.

### Check the source manifest

The source manifest, `source-manifest.yml` beside the dataset's `curated.yml`,
says where the bytes come from and how to encode them. It carries no
descriptive or scientific meaning; that belongs in `curated.yml`.

- [ ] The dataset ID matches `curated.yml`.
- [ ] **Exactly the same variables** as `curated.yml`: none missing, none extra,
  none duplicated.
- [ ] Each variable maps to the right source URI. Source folder names are
  mappings, not identifiers (e.g. S3's `maysept` maps to `may_sept`).
- [ ] A source the build cannot read as published declares a prepare step
  instead of a URI ([Prepared sources](#prepared-sources)), and the curated
  lineage says what the step did.
- [ ] Each source has a checksum, or will get one computed at preflight. A
  prepared source's inputs get theirs computed at fetch.
- [ ] Source bands map correctly onto the reviewed time axis (temporal) or
  band model (static).
- [ ] Per-variable output encoding is set. By default it preserves the source's
  data type, nodata, scale, and offset. A narrower type is allowed only per
  variable, after an experiment proves it lossless.

### Set the `release:` block

The source manifest's `release:` block holds the build settings that are part of
the release's identity:

- [ ] **`created`:** an explicit UTC timestamp for when this declaration was
  approved and frozen. The build never reads the clock.
- [ ] **`chunk_size`:** timesteps per COG for a temporal dataset.
- [ ] **COG options:** interleave, block size, compression, predictor,
  overview resampling per variable (category-preserving for categorical
  variables), BigTIFF.

Changing any of these later means a new release
([Changing things later](#changing-things-later)).

### Choose the release ID

- The date is the UTC date of `release.created`.
- If this dataset already has a release ID for that date, add `-2`, `-3`, ….
- If you are **retrying** a build of the same frozen declaration, keep the same
  release ID and `created`. A changed declaration always needs a new ID, even if
  the output bytes would come out identical.

### Fetch the sources

Preflight reads every source in full, and the build reads each one again. A
source in object storage is streamed every time unless the local source mirror
holds a copy, so fetch the sources once before the first preflight:

```bash
make release-fetch ARGS="<dataset>" RELEASE_SOURCE_MIRROR=<mirror>
```

Fetch copies each `s3://` source to `<mirror>/<bucket>/<key>` and gives the copy
its final name only when it is complete and matches the declared checksum. It
skips a source the mirror already holds, after checking it. For a source with no
declared checksum it prints the computed one; `make release-pin` declares it
after the build ([Pin the release](../app/README.md#pin-the-release)). Pass the
same `RELEASE_SOURCE_MIRROR` to the preflight and the build.

### Prepared sources

Some upstream files cannot be read by the build as published: for example, a
NetCDF file with no CRS, rows running south to north, and NaN for missing
cells, or a raster published as many zipped tiles. For such a source, the
source manifest names the upstream files and a script that turns them into the
one GeoTIFF the build reads:

```yaml
variables:
  - id: "<variable-id>"
    source:
      checksum: "1220…"            # the prepared GeoTIFF: the bytes the build reads
      bands: "descriptions"
      prepare:
        script: "scripts/prepare_<dataset-id>.py"
        inputs:
          - uri: "https://<host>/<path>"
            checksum: "1220…"      # the upstream file
```

A source declares either `uri` or `prepare`, never both. A source assembled from
many files, such as a set of tiles, lists one `inputs` entry per file. Files
the publisher does not provide, such as tiles that would hold only ocean, are
listed in a comment above `inputs`, so a reviewer can check each one.
`lbda_v2` and `owda_v1` are examples, each prepared from one NetCDF file.

**What a prepare step may do.** Change how the data is stored, never its
values: file format, mosaicking, orientation, CRS assignment, nodata encoding,
band descriptions, and leaving out whole timesteps. The curated lineage
describes the step in plain words. Before it exits, the script reads its output
back and checks that every valid value equals the input value it came from.
Review of the script and that check keep the values unchanged; the checksums
only keep the reviewed bytes from changing afterwards.

**The script.** One file in `timeseries/ingest/scripts/`, importing only the
standard library and the packages installed in the ingest image (GDAL, numpy,
netCDF4). It cannot import other files from the repository, so its hash
covers all the logic that ran, and the toolchain covers the rest. Fetch runs it
in the ingest image as `<script> <variable-id> <output> <input>…`, with the
inputs in declared order.

**One output per variable.**

- Many inputs, one output: a set of tiles becomes one mosaic.
- One input, several variables: each variable declares the same input. Fetch
  downloads it once, since its mirror path comes from its URI, and runs the
  script once per variable; the variable ID tells the script which to write.
- A temporal variable's output is still one file: the build splits it into
  `chunk_size` COGs, as for any source.

**Where each checksum lives.** The inputs' and the output's checksums are part
of the declaration, so they enter the declaration digest; the script's hash
does not.

| Where | What | Changes when |
| --- | --- | --- |
| Source manifest (committed) | Each input's checksum, the script path, the output's checksum | A reviewed edit; changing a declared value makes a new release |
| Source mirror (on the build host, not committed) | The inputs, the output, and a note beside the output | Fetch writes them |
| Release manifest's `sources` (published) | The inputs and their checksums, the script path and hash, the output checksum | Never; the release is immutable |

The mirror holds:

```text
<mirror>/<host>/<path>                                        # each input
<mirror>/prepared/<dataset-id>/<variable-id>.tif                # output
<mirror>/prepared/<dataset-id>/<variable-id>.tif.prepared.json  # note
```

The note records how the output was made:

```json
{ "script": "scripts/prepare_<dataset-id>.py", "script_sha256": "…",
  "toolchain": { "gdal_image": "ghcr.io/osgeo/gdal:…@sha256:…", "python_lock_sha256": "…" },
  "inputs": [{ "uri": "https://<host>/<path>", "checksum": "1220…" }],
  "output_checksum": "1220…" }
```

The toolchain is what else can change the script's output: the GDAL base image,
pinned by digest, and the SHA-256 of the locked Python dependencies. Rebuilding
the ingest image for an unrelated code change leaves it unchanged.

**What fetch does with each input.** It never asks the upstream server whether
the file changed. The declared checksum is the only test, and it is computed
from the local copy, with no download:

| Mirror holds | Fetch |
| --- | --- |
| A copy matching the declared checksum | Keeps it; nothing is downloaded |
| No copy | Downloads it and checks it |
| A copy with a different checksum | Fails |
| A copy, and no checksum is declared yet | Keeps it and prints the computed checksum |

**When fetch reruns the script.** When any of these holds:

- there is no output or no note;
- the note's `script_sha256` differs from the script's current hash;
- the note's `toolchain` differs from the current one;
- the note's inputs differ from the declared inputs;
- the output's hash differs from the note's `output_checksum`.

It then checks the new output against `source.checksum`. On a match it writes
the note; on a mismatch it deletes the output and fails, printing the measured
checksum; with nothing declared yet it keeps the output and prints the
checksum. An edited script that writes the same bytes therefore changes nothing
but the note, and one that writes different bytes cannot reach a build until
someone declares the new checksum, which makes a new release. A new toolchain
is treated the same way. Leaving the script's hash out of the
declaration digest is what keeps the first case from needing a new release: the
output's checksum already pins the bytes, and the release manifest records the
script.

**Preflight and build** read the output from the mirror and check it against
`source.checksum`, as for any source. They refuse a prepared source when:

- the mirror has no output: unlike an `s3://` source, there is nowhere else to
  read it from, so run fetch on this host first;
- its note is stale (the script or the toolchain changed since fetch): rerun
  fetch.

**When the upstream file changes.** SKOPE notices only when it downloads the
file again, on a new host or after the mirror is cleared: the checksum fails,
and fetch stops, so the new bytes are never used by accident. Adopting them is
a new release. SKOPE keeps no durable copy of upstream inputs, so if a
publisher removes a file, a release built from it cannot be rebuilt from a clean
host; its published bytes are unaffected.

### Run the preflight

The preflight validates without writing anything. For each requested dataset it:

1. validates `curated.yml` and the source manifest;
2. opens and inspects every source raster: grids, band counts, band
   descriptions, encodings;
3. checksums every source, compares it with the declared checksum, and
   computes any checksum the manifest omits;
4. freezes the resolved declaration and computes its digest.

For a prepared source, steps 2 and 3 read the prepared output in the mirror.

Read the report. Every finding names a requirement ID, a severity, the file, and
the dataset, variable, chunk, or band involved. Fix every error before building.
[Common failures](#common-failures) lists the usual ones.

## Build and publish

### Run the build

The build takes one dataset or several. With several, every dataset is
preflighted before any is built. A dataset that fails is reported and skipped;
the others build and publish normally. The run exits with a failure status that
lists every failed dataset. Retry those on their own later.

What happens inside, in order:

1. Create a unique staging directory on the same filesystem as the release root.
2. Write the COGs.
3. Reopen every COG and read the actual bytes: structure, overviews, encoding,
   band descriptions, embedded statistics.
4. Freeze that observation; STAC is generated only from it.
5. Generate the Collection, Items, and `overview.yml`.
6. Run every validation pass, including cross-checks between COG bytes, STAC,
   the time-to-band rule, and checksums.
7. Write `release-manifest.json`, last.
8. Publish.

If any step fails, no manifest is written and nothing becomes selectable.

### Inspect the result

- **The validation report:** the build prints it to standard output and exits
  with a failure status while any error remains. Zero errors is required; read
  the warnings (for example, a category outside the reviewed vocabulary).
- **`overview.yml`:** the human-readable summary. Check titles, units, extent,
  grid, time axis (`origin`, `step`, `count`, `chunk_size`), and each variable's
  `data_type`, `nodata`, `scale`, and `offset`. Its header records the release ID
  and declaration digest it came from.
- **`release-manifest.json`:** status `complete`, the release ID, the
  declaration digest, every source with its checksum, and an inventory of every
  file in the release.
- **The pin values:** a successful build prints each release's ID, declaration
  digest, and manifest SHA-256. `make release-pin` writes them for you
  ([Pin the release](../app/README.md#pin-the-release)).

### Publish

The staging directory is renamed atomically to `<release-root>/<release-id>`.
This fails if that path already exists or if staging is on a different
filesystem. The build makes the release readable and not writable by every user
(files 0444, directories 0555), so the API and TiTiler, which run as uid 1000,
can read it. A directory without a valid manifest is invisible to readers.
Releases are published to a local release root only; object storage is not
supported yet (TXN-012).

Once published, a release never changes. A fix is a new release.

### After a failed build

A failed build leaves the previously published releases untouched. What it
leaves behind is a staging directory, never a final-looking release path.

Cleanup only ever targets an explicitly named incomplete staging directory,
never a release with a valid manifest. Do not reuse leftover COGs because their
filenames look right; reuse needs matching checksums and full validation. The
release tool has no cleanup command yet; TXN-009 to TXN-015 set what one must
do.

## Changing things later

| Change | New release? | Notes |
| --- | --- | --- |
| Source data (new or corrected source file) | Yes | May also need a new dataset version if the science changed. A prepared source's replaced upstream file is noticed on its next download ([Prepared sources](#prepared-sources)) |
| Any curated text, including a typo in a description | Yes | `curated.yml` is part of the declaration |
| Units, category meanings, temporal semantics | Yes | Ask whether the dataset version changes too |
| Source mapping or per-variable encoding | Yes | |
| A prepare script edited, same output bytes | No | Fetch reruns it and confirms the output |
| A prepare script or toolchain change that alters the output bytes | Yes | Fetch fails until the new output checksum is declared |
| `chunk_size` or COG options | Yes | Same dataset version: a repackaging |
| Which uncertainty product is paired with an estimate | Yes | Needs a dataset-version review if semantics change |
| Display, pins, ordering | No | The app's side ([app README](../app/README.md#changing-things-later)) |

**The release a new one replaces** needs nothing done to it. Pinning the new
release is the whole change; the old one stays where it is, unchanged, as the
rollback target, and its ledger entry stays as it is. `withdrawn` is only for a
release found to be wrong.

**Release or dataset version?** A dataset's `version` is the version of the
source science. Repackaging (chunking, COG options, a metadata correction) is a
new release of the same version. When science review finds a change alters the
scientific quantity itself, it needs a reviewed decision on a new dataset
version or identifier. A new version's release may link back to the previous
version's Collection with `predecessor-version`.

## Troubleshooting and reference

### Common failures

| Stage | Symptom | Likely cause | Requirement |
| --- | --- | --- | --- |
| Curated validation | Missing or unknown required field | A field left blank or marked unknown | META-002, VAL-003 |
| Curated validation | Observed or presentation field rejected | `crs`, `nodata`, `colormap`, ranges for display, etc. in `curated.yml` | META-002, META-003 |
| Curated validation | Temporal value has the wrong type | Unquoted `0103` parsed as a number | META-005 |
| Fetch | Prepare input checksum mismatch | The publisher replaced the file, or the download is corrupt | OBS-011 |
| Fetch | Prepared output checksum mismatch | The prepare script or the toolchain changed the output bytes | OBS-015 |
| Source preflight | Variable sets differ | `curated.yml` and source manifest list different variables | OBS-001 |
| Source preflight | Grid mismatch | Sources differ beyond the declared tolerance | OBS-003 |
| Source preflight | Band count or description mismatch | Source bands do not match the reviewed time axis | OBS-005, OBS-006 |
| Source preflight | Checksum mismatch | Source bytes changed since the manifest was written | META-004, OBS-011 |
| Source preflight | Prepared source missing from the mirror | Fetch has not run on this host | OBS-016 |
| Source preflight | Prepared source stale | The script or the toolchain changed since fetch | OBS-016 |
| COG validation | Encoding rejected | Overflow, truncation, or nodata colliding with a valid value | OBS-008, COG-014 |
| COG validation | Band with no valid pixels | All-nodata band without a declared policy | OBS-007 |
| Cross-artifact | Time-to-band rule disagrees with bytes | Band names or counts differ from the axis and `chunk_size` | API-002, VAL-007 |
| Publish | Destination exists | Release ID already used | TXN-002, REL-008 |
| Publish | Rename refused | Staging on a different filesystem | TXN-002 |

### Where each step is defined

| Step | Release spec section |
| --- | --- |
| Authority model | §4 (AUTH-001–003) |
| Dataset profiles, Items, identifiers | §5 (ORG-001–010) |
| Release layout and release ID | §6 (REL-001–011) |
| `curated.yml` and source manifest | §7 (META-001–013) |
| STAC requirements | §9 (STAC-001–012), §10 (SKOPE-001–009) |
| Fetch, preflight, and build-state invariants | §11 (OBS-001–016) |
| COG profile and encoding | §12 (COG-001–016) |
| Manifest and declaration digest | §13 (MAN-001–011) |
| Publication, recovery, promotion, mounts | §14 (TXN-001–018) |
| Batch builds | §14.4 (ORCH-006, ORCH-008), §18 (MIG-003) |
| Overview and time-to-band rule | §15 (API-001–010) |
| Validation passes and findings | §16 (VAL-001–008) |
| Questions for a dataset creator | §20.2 |

## Code

| Path | Holds |
| --- | --- |
| `datasets/<dataset-id>/` | `curated.yml`, `source-manifest.yml`, and the `releases.yml` ledger |
| `scripts/` | Prepare scripts, and `vendor_stac_schemas.py` |
| `src/skope_release/cli.py` | The `skope-release` command and its subcommands |
| `src/skope_release/fetch.py`, `prepare.py` | Fetch into the source mirror, prepared sources |
| `src/skope_release/preflight.py`, `plan.py` | Source preflight and the frozen build plan |
| `src/skope_release/build.py`, `cog_writer.py`, `inspect_bytes.py` | Staging, COG writing, and byte inspection |
| `src/skope_release/stac_serializer.py`, `stac_validate.py`, `crosscheck.py`, `overview.py` | STAC, its validation, cross-artifact checks, and the overview |
| `src/skope_release/manifest.py`, `identity.py`, `pin.py` | The release manifest, release IDs and digests, and pinning |
| `src/skope_release/timeaxis.py` | The time-to-band rule, copied into the API as `timeseries/app/vendor/timeaxis.py` |
| `src/skope_release/models.py`, `schemas/` | Typed models for every document, and their JSON Schemas |
| `tests/release/` | The release tool's tests |

`src/cog_stac_pipeline/`, `make ingest`, and `make migrate-legacy-data` are the
legacy pipeline, proposed for deletion; see
[docs/data-preparation.md](../../docs/data-preparation.md). Do not use them for a
new dataset.
