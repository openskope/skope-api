# SKOPE Release Consumption Specification v1

- Status: Proposed
- Version: 0.1
- Date: 2026-10-01
- Companion: [SKOPE Dataset Release Specification v1](dataset-release-v1.md)
- Decision record: [ADR
  0006](../../.agents/decisions/0006-stac-authoritative-dataset-releases.md)

This document uses the key words MUST, MUST NOT, SHOULD, SHOULD NOT, and MAY as
described by [RFC 2119](https://www.rfc-editor.org/rfc/rfc2119) and
[RFC 8174](https://www.rfc-editor.org/rfc/rfc8174) when, and only when, they
appear in uppercase.

This document specifies a target architecture. It does not describe current
behavior, and it does not prescribe implementation code. The same approval gate
as the release specification applies.

## 1. Purpose and scope

The release specification ends at each release's `overview.yml`. This
specification starts there and covers how a SKOPE deployment serves releases to
SKOPE UI:

- which releases a deployment serves, and how it verifies and composes them at
  startup (Section 3);
- presentation: the display preferences and palettes files (Section 4);
- the `/metadata` response (Section 5);
- the time and request protocol for tiles and extraction (Section 6); and
- migration from the current application (Section 9).

It specifies how SKOPE consumes and serves releases, not the application as a
whole; other application improvements are out of scope.

Terms defined in the release specification, such as release, release set,
release overview, physical value space, and valid range, keep their meaning
here.

## 2. Terminology

**Pin file**
: The version-controlled record of one environment's release set.

**App registry**
: What one deployment serves at runtime: the release overviews of its pinned
  release set, composed without modification.

**Display preferences**
: `deploy/display/preferences.yml`: how each dataset and variable is shown.

**Palettes file**
: `deploy/display/palettes.yml`: named colour lists referenced by the display
  preferences.

**Static key**
: The reserved, non-ISO key `static`, which stands in for a timestep when a
  request addresses a `StaticRasterDataset`.

## 3. Release-set pin and app registry

- **PIN-001:** Each environment MUST record its release set in the skope-api
  repository as one pin file, `deploy/releases/<environment>.yml`. The pin file
  MUST name the release storage root and, per dataset, the release ID, the
  declaration digest, and the SHA-256 digest of the release's
  `release-manifest.json`. The manifest digest identifies the published bytes,
  because the manifest records the checksum of every other file; the declaration
  digest identifies only the inputs. The pin file MUST NOT be assigned an
  identifier or digest of its own, and MUST NOT be published as a release
  artifact. The deployment MUST derive the API and TiTiler release mounts
  (release specification TXN-011) from the pin file's `release_root` rather
  than configure them separately.

```yaml
release_root: /srv/datasets/releases      # or an object-storage prefix
releases:
  - dataset: paleocar_v3
    release_id: paleocar_v3-r-2026.09.12
    declaration_digest: a8c92e7d4b131029ea9fd04cd7d8ba3131d058cdaf5127add24dc68c175dc13f
    manifest_sha256: 6b1f0e3a9c2d4b7e8f5a1c0d9e2b3a4f5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f
```

- **PIN-002:** The published catalog MUST NOT be an input to the application.
  The application reads its pin file; it does not discover datasets at runtime.
- **PIN-003:** The API MUST read each pinned release's `overview.yml` once at
  startup and serve from that in-memory registry, without a per-request round
  trip to release storage.
- **PIN-004:** Before serving, the API MUST verify, for every pinned release:
  the manifest digest against the pin; the overview's checksum against its
  manifest inventory entry; the overview's recorded release ID and declaration
  digest against the pin; and that every inventoried file exists with its
  recorded size. It MUST refuse to serve on any mismatch rather than falling
  back or serving the remaining datasets. Full byte verification happens at
  promotion (release specification TXN-011). A refusal MUST log one structured
  error per failed check, naming the requirement ID, dataset, release ID, and
  the expected and observed values, and the API process MUST then exit with a
  failure status.
- **PIN-005:** The app registry MUST be the mechanical composition of the pinned
  overviews: selecting and concatenating them, never editing, merging, or
  reinterpreting their contents. The API MUST refuse to start if two pinned
  releases declare the same dataset identifier. Any consistency check that spans
  datasets belongs here, because a release build sees one dataset.
- **PIN-006:** The registry's allowlist MUST cover every dataset and variable of
  every pinned release. The display preferences MUST NOT narrow it: the data
  remains reachable through any STAC client, so display choices are not an
  access boundary. Serving fewer datasets means pinning fewer releases.

## 4. Presentation

A release states what the data is; presentation states how someone chose to show
it. Presentation has different authors and review, and it changes with an
application build rather than with the data. Keeping it in a release would force
republishing unchanged data to change a colour, or mutating an immutable
release. So presentation lives in two files in the skope-api repository, and no
`skope:` field carries a presentation choice.

```text
deploy/display/preferences.yml    # how each dataset and variable is shown
deploy/display/palettes.yml       # named colour lists
```

Both files are shared by every environment. Only the pin file differs per
environment, so dev and production render the same variable the same way.

### 4.1 Display preferences

```yaml
defaults:
  ticks: 5                     # a count, or an explicit list per variable
  nodata: transparent

datasets:
  paleocar_v3:
    order: 20
    map_view:                  # optional; overrides fitting the bbox
      center: { lon: -108.5, lat: 37.0 }
      zoom: 6
    default_variable: ppt_annual
    variables:
      ppt_annual:
        order: 10
        palette: skope-precip
        range: [0, 1800]
        note: "98th percentile over 0103-2000; upper tail compressed"
      gdd_cotton_annual:
        order: 20
        palette: skope-precip
        range: [0, 2200]
        ticks: [0, 550, 1100, 1650, 2200]
  paleocar_v2:
    order: 30
    variables:
      maize_farming_niche:
        order: 30
        palette: niche-binary
        labels: { 0: Outside, 1: Inside }   # optional; falls back to curated text
```

- **DISP-001:** Every pinned dataset MUST have a dataset entry, and every
  variable of every pinned release MUST have a variable entry. Entries for
  datasets or variables absent from the release set are permitted, because a
  development environment may pin a subset.
- **DISP-002:** A variable entry MUST name a `palette`. A continuous variable
  MUST also carry a `range`. `ticks`, `note`, and, for a categorical variable,
  `labels` are optional. A variable is categorical exactly when its curated
  metadata declares category meanings; an optional `type` field, when present,
  MUST agree.
- **DISP-003:** A `range` MUST have finite endpoints with the lower endpoint
  less than the upper. It is a visualization range, distinct from observed
  statistics and from the curated valid range, and it is not published in a
  release.
- **DISP-004:** A `range` MUST be in physical value space, and one range MUST
  serve both the tile rescale and the legend so the two cannot disagree. The map
  renderer and legend MUST use the endpoints unchanged: global percentage
  multipliers, implicit clipping, and other client-side adjustments MUST NOT
  alter them. When an outlier or percentile policy informs the endpoints, the
  method and its scope MUST be recorded in `note` rather than recomputed by the
  client.
- **DISP-005:** Every dataset and variable entry MUST carry a numeric `order`.
  The API MUST list datasets, and each dataset's variables, in `/metadata`
  sorted by `order`, breaking ties by identifier. YAML mapping order carries no
  meaning, so file order MUST NOT be relied on. Clients MAY rely on the served
  order and need not sort the lists themselves.
- **DISP-006:** `map_view` is optional. When present it MUST use named `lon` and
  `lat` keys within WGS 84 ranges, never a positional pair. Without
  `map_view`, a client MUST fit the initial view to the dataset's bbox.
- **DISP-007:** `default_variable`, when present, MUST name a variable of the
  pinned release. Without it, a client selects the lowest-ordered variable,
  which is the first one listed.
- **DISP-008:** `ticks` defaults to a count; an explicit list MAY replace it
  when round numbers read better, and every listed tick MUST lie within the
  range. `labels` MAY shorten the curated meaning of a category for display; a
  missing label falls back to the curated text, and a label for a category the
  curated metadata does not declare MUST be rejected.

### 4.2 Palettes

```yaml
skope-precip:
  kind: ramp
  colors: ["#B5834A", "#CDAA78", "#DEC499", "#EFE0CA",
           "#D8EEEA", "#9ECEC6", "#5CAFA8", "#358C87"]
niche-binary:
  kind: set
  colors: ["#e8e8e8", "#2b8c3e"]
```

- **DISP-009:** Every palette the display preferences reference MUST be defined
  in the palettes file, including common palettes such as `viridis`, which are
  copied in as colour lists. Palettes are not versioned.
- **DISP-010:** A palette's `kind` MUST be `ramp` for a continuous variable and
  `set` for a categorical one. A categorical variable's colours bind to its
  categories in order; `palette-name[i]` MAY override the colour of one
  category. A `set` MUST have at least as many colours as the categories it
  colours.
- **DISP-011:** The TiTiler image MUST register every palette in the palettes
  file as a named colormap at image build, and tile requests MUST pass the
  palette by name. The API MUST resolve legend colours from the same file and
  MUST NOT fetch colours from TiTiler. The API and TiTiler images MUST be built
  from the same commit of the palettes file, and deployment state MUST record
  that commit.

Registering palettes at image build keeps every tile URL short and cacheable and
spares TiTiler from parsing a colormap on each tile. A palette change therefore
rebuilds the TiTiler image, which the paired deployment does anyway.

### 4.3 Validating the display files

- **DISP-012:** CI MUST validate the display preferences against the palettes
  file, without release data: every referenced palette exists; `kind` matches
  the variable type where `type` is stated; every `palette-name[i]` index is in
  range; ranges are finite with lower less than upper; explicit ticks lie within
  the range; every `order` is numeric; `map_view` uses `lon`/`lat` within WGS 84
  ranges; and no unknown field appears.
- **DISP-013:** API startup MUST refuse to serve, reporting as PIN-004
  requires, when a check that needs the pinned overviews fails: DISP-001
  coverage, DISP-002 type agreement, DISP-007 default variables, and DISP-010
  colour counts.

## 5. `/metadata`

- **PROTO-001:** `/metadata` MUST expose one response contract with
  `schema_version = "1.0.0"`, versioned independently of the release overview
  and manifest. The response MUST be a deterministic projection of the app
  registry and the display files. The API and SKOPE UI MUST reject an
  unsupported major version and MUST NOT maintain a parallel legacy response
  mode or versioned route.
- **PROTO-002:** The response MUST follow a versioned JSON Schema defined in
  implementation Phase 1, covering at least the fields in the outline below.
- **PROTO-003:** Rendering fields MUST come from the display entry: the range,
  the resolved palette colours, ticks, and category labels. The response MUST
  NOT contain separate `min` and `max` fields or renderer-specific authority
  fields. SKOPE UI MUST use the served range unchanged, applying no multiplier.
- **PROTO-004:** The API MUST compute a human-readable grid resolution label
  from the overview's grid transform and CRS; it is not curated text.

Response outline, per dataset:

| Field | Required | Source |
| --- | --- | --- |
| `id`, `title`, `description`, `version`, `license` | Yes | Overview |
| `profile` (`TemporalCubeDataset` or `StaticRasterDataset`) | Yes | Overview |
| `region_name` | Yes | Overview |
| `bbox` (`[west, south, east, north]`, WGS 84) | Yes | Overview |
| `resolution_label` | Yes | Computed (PROTO-004) |
| `time` (`kind`: `regular`, `enumerated`, or `static`, with its axis fields) | Yes | Overview; `static` per PROTO-008 |
| `providers`, `citation`, `doi`, `lineage`, `uncertainty`, `links` | When curated | Overview |
| `order`, `map_view`, `default_variable` | `order` yes; others optional | Display preferences |
| `release_id`, `declaration_digest` | Yes | Overview |
| `variables` | Yes | Below |

Per variable:

| Field | Required | Source |
| --- | --- | --- |
| `id`, `title`, `description`, `unit` | Yes | Overview |
| `category` | When curated | Overview |
| `categories` (value → meaning) and `labels` | Categorical only | Overview; labels from display preferences |
| `order` | Yes | Display preferences |
| `display`: `type`, `range`, `colors`, `ticks`, `palette`, `note` | `range` continuous only; `note` optional | Display preferences and palettes |

## 6. Time and request protocol

- **PROTO-005:** The API and SKOPE UI MUST use only the canonical dataset,
  variable, and timestep identifiers published in releases. TiTiler MUST remain
  internal and API-mediated, and time-to-band resolution, storage paths, and
  TiTiler parameters MUST be derived by the API at runtime rather than exposed
  to the UI or curated as metadata.
- **PROTO-006:** The tile route MUST address a timestep by its canonical ISO key
  at the dataset's precision. A tile
  request is an exact lookup: a key that is not on the axis, including one that
  falls between the steps of a regular axis, is rejected.
- **PROTO-007:** An extraction or analysis request's time range selects the
  timesteps it contains. Responses MUST carry the timestep of every value, so
  clients never reconstruct the axis.
- **PROTO-008:** A `StaticRasterDataset` MUST be served with `time.kind =
  "static"` and the single static key `static`. The tile route MUST accept
  `static` for a static dataset, and extraction MUST return one value labelled
  `static`. SKOPE UI MUST accept a single-step range and MUST hide the time
  slider and time-series plot for a static dataset. `static` never appears in a
  release.
- **PROTO-009:** Request errors MUST use these statuses:

  | Request | Status |
  | --- | --- |
  | Unknown dataset, variable, or timestep in a tile path | 404 |
  | Malformed timestep | 422 |
  | Extraction range that selects no timesteps | 422 |

- **PROTO-010:** When a variable's scale is not 1 or its offset is not 0, the
  API MUST pass TiTiler's `unscale` option for tiles and MUST apply scale and
  offset during extraction, so that display ranges and extracted values are both
  in physical value space.
- **PROTO-011:** In v1, a categorical variable MUST be rendered with a discrete
  legend of its labelled categories, and binary categorical variables keep every
  zonal statistic, since the mean of a 0/1 variable is the fraction of the area
  in category 1. Statistics for categorical variables with three or more
  categories are deferred to v2.

## 7. Acceptance tests

| Test ID | Scenario and expected result | Requirements |
| --- | --- | --- |
| APP-AT-001 | Build continuous and categorical display-preference fixtures with their palettes; require one range for both tile rescale and legend; resolve palette names to colours without contacting a tile server; require dataset and variable entries for every pinned dataset and variable while accepting entries for absent ones; reject an override for an undeclared category, a `type` that disagrees with curated categories, a `set` with too few colours, and ambiguous legacy fields; require numeric `order` with identifier tie-breaks, named `lon`/`lat` in WGS 84 ranges, and a present `default_variable`. | DISP-001 through DISP-010, DISP-013 |
| APP-AT-002 | Run the CI display-file checks on good and bad fixtures and prove they need no release data. | DISP-012 |
| APP-AT-003 | Build the TiTiler and API images from one palettes commit; prove tiles use registered names, the API never requests colours from TiTiler, and deployment state records the commit. | DISP-011 |
| APP-AT-004 | Compose a multi-release registry from a pin file; prove composition reads overviews once at startup and edits none, refuses duplicate dataset identifiers, refuses to serve on a manifest-digest, overview-checksum, ID, digest, or file-size mismatch, never reads the published catalog, and keeps an allowlist covering every pinned dataset and variable regardless of the display preferences. | PIN-001 through PIN-006 |
| APP-AT-005 | Serve `/metadata` from the registry and display files; require `schema_version = "1.0.0"`, every outlined field, a computed resolution label, and rendering fields from the display entry; reject unsupported major versions on both sides; and prove no legacy response mode, route, or `min`/`max` field exists. | PROTO-001 through PROTO-004 |
| APP-AT-006 | For positive, nonzero-minimum, and negative-minimum variables, carry an exact physical display range from the display preferences through `/metadata` into SKOPE UI tile parameters and legend endpoints; reject percentage multipliers, hidden clipping, encoded-space reuse, and tile/legend disagreement; verify time-series values and summary statistics are unchanged by display choices. | DISP-003, DISP-004, PROTO-003 |
| APP-AT-007 | Request tiles by canonical ISO key, including first, middle, and last timesteps; reject off-axis and between-step keys with 404 and malformed keys with 422; extract ranges that select a subset and that select nothing (422); require a timestep for every returned value; serve SRTM with the `static` key for tiles and extraction and render it in SKOPE UI without a slider or plot. | PROTO-005 through PROTO-009 |
| APP-AT-008 | Serve a fixture variable with scale 10 and offset 0; prove tiles pass `unscale` and extracted values are physical. | PROTO-010 |
| APP-AT-009 | Require `gdd_maize_may_sept` throughout API and UI state and reject the `gdd_may_sept` alias at every protocol boundary. | CUT-003 |
| APP-AT-010 | Render `maize_farming_niche` with a discrete two-category legend and verify every zonal statistic is available for it. | PROTO-011 |
| APP-AT-011 | Deploy the new API and SKOPE UI as one staged pair; exercise metadata, static and temporal maps, legends, time series, extraction, and summary statistics; promote and roll back both together; show an update-required state for a stale UI fixture; and prove no legacy metadata mode, alias, static-dataset year key, rendering field, or independently deployable compatibility path remains. | CUT-001, CUT-002 |

## 8. Alternatives considered

### 8.1 Where presentation lives

| Alternative | Advantages | Costs and conclusion |
| --- | --- | --- |
| STAC Rendering 2.0.0 as the authority | A recognizable STAC extension that maps closely to common tile parameters | Pilot maturity, temporal Item duplication, renderer-shaped concepts, and incomplete support for custom legends make it an unstable authority. Rejected for v1. |
| Style assets inside the release, optionally projected to Rendering 2.0.0 | Checksummed beside the data and discoverable by standards-oriented clients | Ties a presentation change to a new immutable data release and gives an opinion the standing of a measurement. Rejected. |
| Display files in the skope-api repository | Reviewed and versioned where the application is built; cannot affect release identity or force republishing unchanged data | Generic STAC clients see no rendering hint, and the API must serve explicit rendering fields. Selected. |

### 8.2 Display ranges

| Alternative | Advantages | Costs and conclusion |
| --- | --- | --- |
| Treat legacy `min`/`max` as observed extrema | Needs no review before generating statistics | The values have no provenance, differ from byte-derived statistics, and repeat implausibly across unlike variables. Rejected. |
| Treat legacy `min`/`max` as valid ranges | Preserves simple variable bounds | Several values appear chosen for display or for scaled products and cannot establish scientific validity. Rejected without per-variable evidence. |
| Keep the UI's global percentage cap | Reproduces current visual contrast | A fixed fraction of an absolute maximum is not a percentile, misbehaves for nonzero or negative minima, and hides presentation policy in client code. Rejected. |
| Exact reviewed display ranges | Deterministic tiles and legends, per-variable outlier choices, and a clean separation from statistics and validity | Needs a coordinated API/UI migration and per-variable review. Selected. Legacy `min`/`max` values remain migration evidence only. |

### 8.3 Delivering palettes to TiTiler

| Alternative | Advantages | Costs and conclusion |
| --- | --- | --- |
| Pass the colours with every tile request (`colormap=` JSON) | No TiTiler image change for a palette edit | Every tile URL carries about 256 colour entries, TiTiler parses a colormap per tile, and caching suffers. Rejected. |
| Rely on TiTiler's built-in palette table | Nothing to maintain | Couples rendering to whichever TiTiler version is deployed, and the API must ask TiTiler for legend colours. Rejected. |
| Register the palettes file in TiTiler at image build | Short, cacheable tile URLs; one source of colours for tiles and legends | A palette edit rebuilds the TiTiler image. Selected. |

### 8.4 API/UI cutover

| Alternative | Advantages | Costs and conclusion |
| --- | --- | --- |
| Preserve legacy behavior indefinitely | Minimizes immediate client changes | Retains ambiguous fields, aliases, synthetic temporal behavior, and duplicate registries. Rejected. |
| Parallel old and new endpoints for a fixed transition | Supports independently deployed clients | Adds version routing, fallback logic, duplicated fixtures, and a deprecation period for consumers SKOPE does not have. Rejected. |
| One breaking coordinated API/UI cutover | One explicit contract and no fallback code | Requires complete staging proof and paired deployment and rollback. Selected. |

## 9. Migration from the current application

SKOPE UI is the only supported API consumer, so the API and UI move to this
specification in one breaking, coordinated cutover with no compatibility period.
Today the registries store `region.center` latitude first, the tile route names
its timestep `year`, and SKOPE UI scales `max` by `COLOR_MAX_PCT`; DISP-006,
PROTO-006, and PROTO-003 replace those behaviors.

- **CUT-001:** The API and SKOPE UI MUST pass the complete staging acceptance
  suite as a pair before production, MUST be deployed as one recorded change,
  and MUST be rolled back together if either side fails its schema-version
  handshake or functional health checks. A stale or cached UI that meets a
  mismatched major version MUST show an update-required state rather than
  interpret unknown metadata.
- **CUT-002:** The contract tests for `/metadata`, static and temporal tiles,
  extraction, summary statistics, grid-based request limits, display ranges,
  schema-version mismatch, environment pins, paired deployment, and paired
  rollback MUST pass before `timeseries/metadata.yml` is deleted or the breaking
  production cutover is deployed.

- **CUT-003:** The API and SKOPE UI MUST use the canonical PaleoCAR v3
  identifier `gdd_maize_may_sept`. The legacy `gdd_may_sept` alias MUST NOT be
  emitted, accepted, or stored after the cutover.

The cutover removes, from both the API and SKOPE UI: the legacy `wmsLayer`,
`timeseriesServiceUri`, `min`, and `max` fields; the `gdd_may_sept` alias; the
SRTM `2009` key; `COLOR_MAX_PCT`; the API's runtime colour fetch from TiTiler;
and the checked-in `deploy/metadata/{dev,staging,prod}.yml` registries, whose
content moves to release overviews, the display files, and the pin files.

## 10. Open questions

- Whether `maize_farming_niche` is a binary in-or-out mask or a proportion in
  [0, 1]. Its name says "Proportion w/i Niche", its description reads as a
  binary test, and the release specification's EXP-002 treats it as the
  categorical `uint8` candidate. The answer decides whether PROTO-011 applies to
  it.
- Whether and where to publish the catalog for external researchers is deferred
  to a separate decision (release specification REL-011).
