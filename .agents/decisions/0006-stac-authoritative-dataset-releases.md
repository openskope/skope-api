# 0006: Use STAC-Authoritative Immutable Dataset Releases

- Status: Proposed
- Date: 2026-09-11
- Revised: 2026-09-13
- Specification: [SKOPE Dataset Release Specification v1](../../docs/specs/dataset-release-v1.md)
- Relates to: [0001: Separate Dataset Metadata From Storage Lookups](0001-dataset-registry-and-lookup-contract.md)

## Context

SKOPE currently duplicates dataset metadata across API and ingest YAML files.
The ingest pipeline mutates one of those files with observed raster facts, emits
one STAC Collection per variable, and generates `lookup.json` as the API's
storage index. Existing output files can be reused by filename without proving
their identity or validity.

PR 48 proposes splitting curated metadata by dataset, recording observed facts
in `dataset-facts.json`, composing packages across partial variable runs, and
building the API registry from curated files plus release facts. The `titiler`
branch separately adds per-variable temporal endpoint validation and a complete
whole-migration preflight before transformation. Both bodies of work expose the
need for an explicit authority model, byte validation, and transactional release
publication.

Review of the initial proposal identified three additional boundaries that the
architecture must make explicit: temporal assets need a standard band-to-time
mapping that a client can resolve without a separate index file; planned output
properties must not be confused with observations of final bytes; and static
rasters must not be forced through temporal-cube requirements.

A second review identified two more. First, a release serves two audiences — the
SKOPE application and external researchers — and the application's needs must be
a stated, generated consequence of the release rather than a parallel curated
document. Second, a release and an application build change on different
cadences: data changes in months or years, while presentation changes with a
build. Anything on the faster cadence that is stored inside an immutable release
forces either a needless republication of unchanged data or a forbidden
mutation.

STAC 1.1.0 and its Projection, File Info, Scientific Citation, Raster,
Datacube, Versioning, and Processing extensions provide standards-based homes
for most published metadata. COG headers remain the direct description of the
encoded raster bytes. Release completion and integrity are operational facts,
not geospatial metadata.

## Proposed decision

Subject to approval of the linked specification:

- STAC Collections and Items will be the published metadata authority.
- COG headers will be authoritative for byte-level raster properties, including
  grid, encoding, nodata, scale, offset, band descriptions, and embedded
  statistics. Corresponding STAC fields will be validated projections of those
  facts.
- A release will carry exactly one manifest, at its root, containing integrity,
  provenance, object inventory, release identity, selected dataset composition,
  and completion state only. It will not duplicate geospatial or statistical
  metadata. Per-dataset manifests will not be emitted, because releases are
  built and published as one package of every dataset SKOPE exposes at that
  time.
- The application will read one generated app registry, compacted from validated
  STAC as the final step of a release build. It will record the release ID and
  declaration digest it came from, be verified against them before serving, and
  be materialized where the API reads it without a per-request round trip to
  release storage.
- The time-to-band mapping will be a rule in that registry, storing four values
  per variable — axis origin, step, count, and chunk size — from which the COG
  path and one-based band index of any timestep are computed. Paths will not be
  stored, because the Item ID and COG filename conventions already determine
  them. The rule will be verified during the build against ordered STAC Band
  names and actual COG band counts. `lookup.json` will not be emitted, and a
  rule that disagrees with the bytes will fail the build rather than produce a
  different artifact.
- The typed model will have an immutable `ValidatedBuildPlan` consumed by the
  COG writer and a separate immutable `FinalObservation` consumed by metadata
  serializers after byte inspection.
- Dataset releases will declare either a temporal-cube or static-raster profile;
  SRTM will not acquire synthetic scientific time metadata for API convenience.
- Published releases will be immutable and selected only after a valid root
  release manifest acts as the publication commit marker.
- `dataset-facts.json` will not be part of the target architecture.
- Curated human-authored metadata will be compiled to STAC and will not contain
  observed raster facts. Its authoring form will be one `curated.yml` per
  dataset, and no published counterpart will be emitted: everything a consumer
  needs will be in the STAC Collection, with SKOPE-specific fields under the
  `skope:` prefix.
- Presentation — colormaps, colour stops, visualization ranges, legends, and
  ticks — will not be a release artifact. It will live in one document in the
  application repository, because it is an opinion about display rather than a
  property of the data and it changes on the application's cadence. The API will
  serve explicit rendering and legend fields, and clients will not compute their
  own ranges or apply global multipliers.
- The existing public API and API-mediated TiTiler boundary will remain
  compatible during the initial migration.

## Consequences

- A dataset release can be validated and understood independently of an API
  image or mutable registry file.
- The ingest workflow needs a typed observation model and strict agreement
  checks among COG bytes, STAC, the derived time-to-band rule, and the root
  manifest.
- Local publication needs same-filesystem staging and atomic rename. Object
  storage needs a root-manifest-last commit protocol because it has no
  multi-object transaction.
- One logical dataset version is represented by one Collection; temporal Items
  align variable assets by chunk.
- A presentation change never requires a new release, but the API and SKOPE UI
  must migrate together to serve and consume explicit rendering fields.
- A release is read through its root manifest and STAC tree. Relocating a single
  dataset directory on its own is no longer self-describing, which is accepted
  because releases are published whole and are never referenced across releases.
- The application depends on the release only through the generated registry, so
  the release-to-application data flow must be specified and tested rather than
  left to a curated document maintained beside it.
- Candidate STAC extensions require pinned versions and serialization adapters.
- Ambiguous legacy scientific fields require human review rather than automatic
  interpretation.
- Non-publishing experiments may gather evidence for unresolved decisions, but
  their artifacts cannot become releases or production dependencies without
  approval.
- Accepted ADR 0001 remains in force during compatibility migration. A later
  accepted ADR may supersede its registry duplication consequences after the
  STAC-derived registry is proven.

## Approval gate

This ADR remains Proposed. No production implementation phase may begin until
the linked specification's unresolved decisions are reviewed and this ADR is
accepted or revised. The specification's isolated, non-publishing Phase 0
experiments may run solely to gather decision evidence.
