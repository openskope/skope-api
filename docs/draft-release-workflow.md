# Dataset release workflow (draft)

> **Draft.** This is the README the release workflow will have once it is
> implemented, written ahead of time to test whether the specs describe a
> workable procedure. Nothing here exists yet. Commands are not named; each step
> says what the tooling does, not how to invoke it.
>
> Written from [ADR 0006](../.agents/decisions/0006-stac-authoritative-dataset-releases.md),
> the [dataset release specification](specs/dataset-release-v1.md) (v0.4), and
> the [release consumption specification](specs/release-consumption-v1.md)
> (v0.1). Gaps found while drafting are collected in the last section, which is
> removed when this becomes the real README.

This guide has two audiences:

- **Release authors** prepare a dataset's metadata, build a release, and publish
  it (sections 2 and 3).
- **App developers** choose which releases a deployment serves and how they are
  shown (section 4).

Both should read section 1 first.

## 1. The big picture

### 1.1 What a release is

A release is an immutable package of **one dataset**: its COGs, the STAC
metadata describing them, a generated overview, and a manifest. One dataset,
one release, one manifest. Releases of different datasets are built, published,
and deployed independently, so changing or breaking one dataset never touches
the others.

Getting data into the app takes three separate acts:

| Act | Who | Produces | Undone by |
| --- | --- | --- | --- |
| **Build** | Release author | Immutable bytes at one release path | Not deploying the release |
| **Deploy** | App developer | An updated pin for one environment | A rollback commit |
| **List** (public catalog) | Maintainers | An entry in the public STAC Catalog | A commit removing the entry |

Publishing a release does not make the app serve it. A release is only served
once an environment's pin names it.

### 1.2 Where each fact lives

| If you need… | Trust… |
| --- | --- |
| Descriptions, citations, units, category meanings | The STAC Collection (compiled from `curated.yml`) |
| Grid, nodata, data type, scale, offset, statistics | The COG headers; STAC carries checked copies |
| Which timesteps exist | STAC (`cube:dimensions.time.values`, Band names) |
| Whether a release is complete, and its integrity | `release-manifest.json` |
| A one-file summary of all of the above | `overview.yml`, generated from STAC; never edited |
| Colours, ranges, ordering, map view | Display files in `deploy/display/`; never in a release |
| Which releases an environment serves | That environment's pin file |

Generated copies are checked against their authority on every build. When two
disagree, the build fails rather than picking one.

### 1.3 The flow

```text
  datasets/<id>/curated.yml       datasets/<id>/source-manifest.yml (+ release: block)
            │                                │
            └──────────────┬─────────────────┘
                           ▼
                  fetch (optional): copy the sources into the local mirror
                           ▼
                  preflight: validate both, inspect and checksum every
                  source, freeze the declaration
                           ▼
                  build in staging: write COGs → reopen and inspect bytes
                  → STAC → overview.yml → validate everything
                  → release-manifest.json last
                           ▼
                  publish: atomic rename (local) or manifest-last upload (S3)
                           ▼
        <release-root>/<release-id>/        (immutable from here on)
                           ▼
                  pin in deploy/releases/<env>.yml  +  display files
                           ▼
                  promotion: full verification, recreate API and TiTiler
```

### 1.4 What is in a release

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

A dataset is either a **temporal cube** (`lbda_v2`, `paleocar_v2`,
`paleocar_v3`) or a **static raster** (`srtm`). Static rasters have no Items, no
time axis, and no chunking.

### 1.5 Two identifiers per release

- The **release ID** is a label a person assigns:
  `<dataset-id>-r-YYYY.MM.DD`, or `…-N` (from `-2`) for a later declaration of
  the same dataset on the same date. Example: `paleocar_v3-r-2026.09.12`.
- The **declaration digest** is computed: a SHA-256 of the dataset's resolved
  inputs (`curated.yml`, the resolved source manifest including its `release:`
  block, and the source checksums).

A release ID is bound to exactly one declaration digest, forever. The release ID
is not a dataset version: `paleocar_v3` stays version `3` across any number of
releases that only repackage it.

## 2. Release author: before building

### 2.1 Check `curated.yml`

Each dataset has one `curated.yml`, at
`timeseries/ingest/datasets/<dataset-id>/curated.yml`. It holds what a person knows about the data,
and nothing that can be measured or that only affects display.

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
  ordering, or map view. These go in the display files (section 4.2).
- [ ] **Temporal values are quoted strings:** `"0103"`, not `0103`.
- [ ] **No personal contact details** copied from legacy metadata.

For a dataset whose metadata predates this format (PaleoCAR v3 today), the
answers come from the dataset creator; see spec §20.2 for the questions.

### 2.2 Check the source manifest

The source manifest, `source-manifest.yml` beside the dataset's `curated.yml`,
says where the bytes come from and how to encode them. It
carries no descriptive or scientific meaning; that belongs in `curated.yml`.

- [ ] The dataset ID matches `curated.yml`.
- [ ] **Exactly the same variables** as `curated.yml`: none missing, none extra,
  none duplicated.
- [ ] Each variable maps to the right source URI. Source folder names are
  mappings, not identifiers (e.g. S3's `maysept` maps to `may_sept`).
- [ ] Each source has a checksum, or will get one computed at preflight.
- [ ] Source bands map correctly onto the reviewed time axis (temporal) or
  band model (static).
- [ ] Per-variable output encoding is set. By default it preserves the source's
  data type, nodata, scale, and offset. A narrower type is allowed only per
  variable, after an experiment proves it lossless.

### 2.3 Set the `release:` block

The source manifest's `release:` block holds the build settings that are part of
the release's identity:

- [ ] **`created`:** an explicit UTC timestamp for when this declaration was
  approved and frozen. The build never reads the clock.
- [ ] **`chunk_size`:** timesteps per COG for a temporal dataset.
- [ ] **COG options:** interleave, block size, compression, predictor,
  overview resampling per variable (category-preserving for categorical
  variables), BigTIFF.

Changing any of these later means a new release (section 5).

### 2.4 Choose the release ID

- The date is the UTC date of `release.created`.
- If this dataset already has a release ID for that date, add `-2`, `-3`, ….
- If you are **retrying** a build of the same frozen declaration, keep the same
  release ID and `created`. A changed declaration always needs a new ID, even if
  the output bytes would come out identical.

### 2.5 Fetch the sources

Preflight reads every source in full, and the build reads each one again. A
source in object storage is streamed every time unless the local source mirror
holds a copy, so fetch the sources once before the first preflight:

```bash
make release-fetch ARGS="<dataset>" RELEASE_SOURCE_MIRROR=<mirror>
```

Fetch copies each `s3://` source to `<mirror>/<bucket>/<key>` and gives the copy
its final name only when it is complete and matches the declared checksum. It
skips a source the mirror already holds, after checking it. For a source with no
declared checksum it prints the computed one, for you to commit to the source
manifest. Pass the same `RELEASE_SOURCE_MIRROR` to the preflight and the build.

### 2.6 Run the preflight

The preflight validates without writing anything. For each requested dataset it:

1. validates `curated.yml` and the source manifest;
2. opens and inspects every source raster: grids, band counts, band
   descriptions, encodings;
3. checksums every source, compares it with the declared checksum, and
   computes any checksum the manifest omits;
4. freezes the resolved declaration and computes its digest.

Read the report. Every finding names a requirement ID, a severity, the file, and
the dataset, variable, chunk, or band involved. Fix every error before building.
Section 6.1 lists common ones.

## 3. Release author: build and publish

### 3.1 Run the build

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
8. Publish (section 3.3).

If any step fails, no manifest is written and nothing becomes selectable.

### 3.2 Inspect the result

- **The validation report:** the build prints it to standard output and exits
  with a failure status while any error remains. Zero errors is required; read the warnings (for
  example, a category outside the reviewed vocabulary).
- **`overview.yml`:** the human-readable summary. Check titles, units, extent,
  grid, time axis (`origin`, `step`, `count`, `chunk_size`), and each variable's
  `data_type`, `nodata`, `scale`, and `offset`. Its header records the release ID
  and declaration digest it came from.
- **`release-manifest.json`:** status `complete`, the release ID, the
  declaration digest, every source with its checksum, and an inventory of every
  file in the release.
- **The pin values:** a successful build prints each release's ID, declaration
  digest, and manifest SHA-256. `make release-pin` writes them for you (4.1).

### 3.3 Publish

- **Local:** the staging directory is renamed atomically to
  `<release-root>/<release-id>`. This fails if that path already exists or if
  staging is on a different filesystem. The build makes the release readable
  and not writable by every user (files 0444, directories 0555), so the API and
  TiTiler, which run as uid 1000, can read it.
- **Object storage:** objects are uploaded under `<prefix>/<release-id>/`, data
  and STAC first, sizes and checksums verified, and the manifest uploaded last.
  A retry reuses an object already there only if its size and checksum match;
  a mismatch fails without overwriting.

In both cases, anything without a valid manifest is invisible to readers.

Once published, a release never changes. A fix is a new release.

### 3.4 After a failed build

A failed build leaves the previously published releases untouched. It reports
what it left behind:

- **Local:** a staging directory, never a final-looking release path.
- **Object storage:** objects under a prefix with no manifest, and possibly
  incomplete multipart uploads (these need an explicit abort; they do not
  expire on their own).

Cleanup only ever targets an explicitly named incomplete staging directory or
prefix, never a release with a valid manifest. Do not reuse leftover COGs because
their filenames look right; reuse needs matching checksums and full validation.

The cleanup procedure itself depends on open decision 2 (keep the full
report/authorization/grace-period process, or trim it).

### 3.5 Listing in the public catalog

Not decided for v1 (open decision 1). If adopted, listing is a separate reviewed
commit to a listing record, from which the public catalog is generated. A build
never writes to it.

## 4. App developer: deploying a release

A deployment serves exactly the releases its environment pins. At startup the
API reads each pinned release's `overview.yml` once, verifies it, and composes
the overviews unchanged into the app registry. It never reads the public
catalog, and it never discovers datasets on its own.

A deploy that adds or updates a dataset is usually one change containing:

- the pin file entry (4.1); and
- display entries for any new dataset or variable (4.2), plus a palette if a new
  one is needed.

Put both in the same change. A pinned variable without a display entry stops
the API from starting (4.4).

### 4.1 Pin the release

Each environment has one pin file, `deploy/releases/<env>.yml`:

```yaml
release_root: /srv/datasets/releases      # or an object-storage prefix
releases:
  - dataset: paleocar_v3
    release_id: paleocar_v3-r-2026.09.12
    declaration_digest: a8c92e7d…
    manifest_sha256: 6b1f0e3a…
```

- **`release_root`:** the single storage root every pinned release sits under,
  as a resolved physical path or an object-storage prefix. A release built
  somewhere else has to be copied there first. The API and TiTiler mounts are
  derived from this value, so it is the only place the root is configured.
- **`release_id`:** the release's directory name under the root.
- **`declaration_digest`:** copy it from the manifest's `declaration.digest` (or
  `declaration_digest` in `overview.yml`).
- **`manifest_sha256`:** the SHA-256 of the `release-manifest.json` file itself,
  in hex (for example, `sha256sum <release_root>/<release_id>/release-manifest.json`).
  This digest pins the published bytes, because the manifest records the
  checksum of every other file. The declaration digest pins only the inputs.

`make release-pin` makes these edits from the release's manifest, so nothing is
copied by hand. Run it where the release is, or anywhere with a copy of its
`release-manifest.json`:

```bash
make release-pin MANIFEST=<release_root>/<release_id>/release-manifest.json PIN_ENV=staging
```

It changes nothing; it prints, as a diff, the three edits it would make:

- the ledger entry in `datasets/<dataset>/releases.yml`;
- the dataset's entry in `deploy/releases/<PIN_ENV>.yml`, replaced or added,
  leaving every other entry and comment alone (delete a comment that no longer
  applies, such as one describing the old release);
- a `checksum` under the `uri` of each source declared without one, as the
  build measured it. This leaves the declaration digest unchanged.

It refuses, and edits nothing, if the ledger records that release ID with a
different digest, if a declared checksum differs from the measured one, or if
the source manifest no longer lists the release's variables. Read the diff, then
run the same command with `WRITE=1` to apply it, and review `git diff` before
committing.

Rules:

- Change only the entry for the dataset you are updating. Promoting one dataset
  never requires re-selecting the others.
- Each dataset appears at most once.
- To stop serving a dataset, remove its entry. The display files cannot hide a
  pinned dataset, because display choices are not an access control.
- Environments differ only in their pin files; the display files are shared, so
  every environment renders a variable the same way.

### 4.2 Add display entries

Presentation lives in two files shared by all environments:
`deploy/display/preferences.yml` (how each dataset and variable is shown) and
`deploy/display/palettes.yml` (named colour lists). Changing them never needs a
new release.

```yaml
datasets:
  paleocar_v3:
    order: 20
    map_view:                  # optional; default fits the dataset's bbox
      center: { lon: -108.5, lat: 37.0 }
      zoom: 6
    default_variable: ppt_annual   # optional; default is the lowest order
    variables:
      ppt_annual:
        order: 10
        palette: skope-precip
        range: [0, 1800]
        note: "98th percentile over 0103-2000; upper tail compressed"
```

Checklist:

- [ ] Every pinned dataset has a dataset entry, and every variable of every
  pinned release has a variable entry. Entries for datasets an environment does
  not pin are fine.
- [ ] Every dataset and variable has a numeric `order`. Clients sort by it,
  then by identifier; position in the file means nothing.
- [ ] Every variable names a `palette`, defined in `palettes.yml`. Common
  palettes such as `viridis` are copied in as colour lists too.
- [ ] **Continuous variable:** a `range` with finite endpoints, lower less than
  upper, in physical values (after scale and offset). It sets both the tile
  colours and the legend, used exactly as written. If a percentile or outlier
  rule chose the endpoints, say so in `note`. The palette's `kind` is `ramp`.
- [ ] **Categorical variable** (exactly the variables whose `curated.yml`
  declares category meanings): the palette's `kind` is `set`, with at least as
  many colours as categories, bound to categories in order. Optional `labels`
  may shorten a category's curated meaning, but only for categories the curated
  metadata declares.
- [ ] Explicit `ticks`, if any, lie within the range.
- [ ] `map_view`, if any, uses named `lon` and `lat` keys within WGS 84 ranges.
- [ ] `default_variable`, if any, names a variable of the pinned release.

TiTiler registers every palette as a named colormap when its image is built, and
the API reads legend colours from the same file. A palette edit therefore
rebuilds the TiTiler image, and both images must be built from the same commit.

### 4.3 What is checked where

| When | Checks | Needs release data? |
| --- | --- | --- |
| **CI**, on every change to the display files | Referenced palettes exist; `kind` matches a stated `type`; `palette[i]` indexes in range; ranges finite with lower < upper; explicit ticks within range; numeric `order`; `map_view` keys and ranges; no unknown fields | No |
| **Promotion**, for each newly pinned release | Every file's size and checksum against its manifest (full byte verification) | Yes |
| **API startup**, for every pinned release | Manifest SHA-256 matches the pin; `overview.yml` checksum matches its manifest entry; the overview's release ID and declaration digest match the pin; every inventoried file exists with its recorded size; no dataset pinned twice; every pinned dataset and variable has a display entry; `type` agrees with curated categories; `default_variable` exists; `set` palettes have enough colours | Yes, overviews and manifests only |

### 4.4 Promote and start

Promotion, for each newly pinned release:

1. resolves `<release_root>/<release_id>` (local) or
   `<release_root>/<release_id>/` (object storage); a symlink or other mutable
   pointer is never used;
2. fully verifies it (4.3);
3. records the release ID, declaration digest, manifest digest, resolved path,
   and verification result in deployment state, together with the commit of the
   palettes file the images were built from;
4. recreates both the API and TiTiler containers. Locally, both mount the same
   release root read-only. With object storage, both read releases directly from
   their prefixes without copying them.

The API then runs the startup checks in 4.3. **Any failure stops the whole
API**, not just the dataset at fault: it does not fall back to an older release
or serve the remaining datasets. The API logs one structured error per failed
check, naming the requirement ID, dataset, release ID, and the expected and
observed values, then exits with a failure status. One bad pin entry therefore
takes the environment down, which is why promotion checks every byte first.

### 4.5 Roll back

Revert the pin commit (and any display change made with it) and redeploy.
Promotion recreates both containers. The rolled-back release is untouched,
because deployment never modifies or deletes a release.

### 4.6 The first deployment: the API/UI cutover

The first deployment under this workflow is also the switch to the new
`/metadata` contract, so the API and SKOPE UI ship as one pair: they pass the
staging acceptance suite together, deploy as one recorded change, and roll back
together if either fails. After the cutover, updating a dataset is an ordinary
pin change.

## 5. Changing things later

| Change | New release? | Notes |
| --- | --- | --- |
| Source data (new or corrected source file) | Yes | May also need a new dataset version if the science changed |
| Any curated text, including a typo in a description | Yes | `curated.yml` is part of the declaration |
| Units, category meanings, temporal semantics | Yes | Ask whether the dataset version changes too |
| Source mapping or per-variable encoding | Yes | |
| `chunk_size` or COG options | Yes | Same dataset version: a repackaging |
| Which uncertainty product is paired with an estimate | Yes | Needs a dataset-version review if semantics change |
| Colours, palettes, visualization ranges, ticks, labels | No | Display files; a palette edit rebuilds the TiTiler image |
| Dataset or variable ordering, map view, default variable | No | Display files |
| Which release an environment serves, or hiding a dataset | No | Pin file |
| Listing, deprecation, "latest version" | No | Recorded outside releases |

**The release a new one replaces** needs nothing done to it. Pinning the new
release (4.1) is the whole change; the old one stays where it is, unchanged, as
the rollback target (4.5), and its ledger entry stays as it is. `withdrawn` is
only for a release found to be wrong.

**Release or dataset version?** A dataset's `version` is the version of the
source science. Repackaging (chunking, COG options, a metadata correction) is a
new release of the same version. When science review finds a change alters the
scientific quantity itself, it needs a reviewed decision on a new dataset
version or identifier. A new version's release may link back to the previous
version's Collection with `predecessor-version`.

## 6. Troubleshooting and reference

### 6.1 Common failures by stage

| Stage | Symptom | Likely cause | Requirement |
| --- | --- | --- | --- |
| Curated validation | Missing or unknown required field | A field left blank or marked unknown | META-002, VAL-003 |
| Curated validation | Observed or presentation field rejected | `crs`, `nodata`, `colormap`, ranges for display, etc. in `curated.yml` | META-002, META-003 |
| Curated validation | Temporal value has the wrong type | Unquoted `0103` parsed as a number | META-005 |
| Source preflight | Variable sets differ | `curated.yml` and source manifest list different variables | OBS-001 |
| Source preflight | Grid mismatch | Sources differ beyond the declared tolerance | OBS-003 |
| Source preflight | Band count or description mismatch | Source bands do not match the reviewed time axis | OBS-005, OBS-006 |
| Source preflight | Checksum mismatch | Source bytes changed since the manifest was written | META-004, OBS-011 |
| Source preflight (SRTM) | Indexed tile missing | An official CGIAR tile could not be retrieved | MIG-005 |
| COG validation | Encoding rejected | Overflow, truncation, or nodata colliding with a valid value | OBS-008, COG-014 |
| COG validation | Band with no valid pixels | All-nodata band without a declared policy | OBS-007 |
| Cross-artifact | Time-to-band rule disagrees with bytes | Band names or counts differ from the axis and `chunk_size` | API-002, VAL-007 |
| Publish (local) | Destination exists | Release ID already used | TXN-002, REL-008 |
| Publish (local) | Rename refused | Staging on a different filesystem | TXN-002 |
| Publish (object storage) | Existing object mismatch | A retry found different bytes at the prefix | TXN-007 |
| CI (display files) | Display check fails | Undefined palette, `kind` mismatch, range or tick out of order, non-numeric `order`, positional `map_view`, unknown field | DISP-012 |
| Promotion | Checksum or size mismatch | Release bytes changed or were copied incompletely | TXN-011 |
| Serving | Map tiles return 502 "Upstream tile server error" and extraction fails; the API and TiTiler logs show `Permission denied` | The release is not readable by uid 1000, which the API and TiTiler run as, e.g. one published before TXN-018 or copied by hand. Fix with `chmod` to 0444 files and 0555 directories; the bytes and checksums are unchanged | TXN-018 |
| API startup | Digest or identity mismatch | Typo in the pin, or `manifest_sha256` taken from a different file | PIN-004 |
| API startup | Duplicate dataset | Two pin entries for the same dataset | PIN-005 |
| API startup | Missing display entry | A dataset or variable pinned without display entries | DISP-001, DISP-013 |
| API startup | Display does not fit the data | `type` disagrees with curated categories, missing `default_variable`, too few colours in a `set` | DISP-002, DISP-007, DISP-010, DISP-013 |

### 6.2 Where each step is defined

Release spec:

| Step | Section |
| --- | --- |
| Authority model | §4 (AUTH-001–003) |
| Dataset profiles, Items, identifiers | §5 (ORG-001–010) |
| Release layout and release ID | §6 (REL-001–011) |
| `curated.yml` and source manifest | §7 (META-001–012) |
| STAC requirements | §9 (STAC-001–012), §10 (SKOPE-001–009) |
| Fetch, preflight, and build-state invariants | §11 (OBS-001–014) |
| COG profile and encoding | §12 (COG-001–016) |
| Manifest and declaration digest | §13 (MAN-001–011) |
| Publication, recovery, promotion, mounts | §14 (TXN-001–018) |
| Batch builds | §14.4 (ORCH-006, ORCH-008), §18 (MIG-003) |
| Overview and time-to-band rule | §15 (API-001–010) |
| Validation passes and findings | §16 (VAL-001–008) |
| Questions for a dataset creator | §20.2 |

Release consumption spec:

| Step | Section |
| --- | --- |
| Pin file, startup verification, app registry | §3 (PIN-001–006) |
| Display preferences and palettes | §4.1–4.2 (DISP-001–011) |
| CI and startup display checks | §4.3 (DISP-012, DISP-013) |
| `/metadata`, tiles, extraction | §5–6 (PROTO-001–011) |
| API/UI cutover | §9 (CUT-001–003) |

## Gaps found while drafting

These are places where the specs or ADR did not give enough to write a step
clearly. Remove this section when the draft becomes the README.

### Building and publishing (release spec)

1. **Resolved (META-001):** authoring files live at
   `timeseries/ingest/datasets/<dataset-id>/{curated.yml,source-manifest.yml}`.
2. **Resolved, provisionally (spec §20.3):** checksums are recorded in the
   release manifest's `sources`; authors should commit them.
3. **Resolved, provisionally (spec §20.3):** a `releases.yml` ledger beside the
   authoring files; approval is the reviewed commit of `release.created` with its
   ledger entry.
4. **Resolved, provisionally (spec §20.3):** bump `release.created`; two IDs
   never share a digest.
5. **Resolved, provisionally (spec §20.3):** omitted.
6. **Resolved, provisionally (spec §20.3):** never pinned; the ledger marks it
   `withdrawn`.
7. **Resolved (VAL-002):** reports go to standard output, and a successful build
   prints the release ID, declaration digest, and manifest SHA-256 for the pin;
   `make release-pin` writes them (4.1).
8. **Resolved, provisionally (spec §20.3):** on demand, and in CI against a
   synthetic fixture.

### Deploying (consumption spec, with release spec §14)

9. **Resolved (PIN-004, DISP-013):** one structured log error per failed check,
   then a failure exit.
10. **What "deployment state" is.** TXN-011 records release IDs, digests, paths,
    and verification results in it, and DISP-011 adds the palettes commit, but
    neither spec says where it lives, what format it has, or who reads it.
11. **What runs promotion.** TXN-011 defines what promotion does, but neither
    spec ties it to a command: part of `make deploy-<env>`, a separate step
    before it, or CI.
12. **Whether rollback re-verifies.** TXN-011 fully verifies each *newly* pinned
    release. When rollback re-pins a release that was verified before, does it
    get full verification again or only the startup checks?
13. **Resolved (PIN-001):** the API and TiTiler mounts are derived from the pin's
    `release_root`.
14. **Resolved:** TXN-011 now refers to PIN-004.
