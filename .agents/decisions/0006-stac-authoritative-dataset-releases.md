# 0006: Use STAC-Authoritative Immutable Dataset Releases

- Status: Accepted
- Date: 2026-09-11
- Revised: 2026-10-01
- Specifications: [SKOPE Dataset Release Specification v1](../../docs/specs/dataset-release-v1.md) and [SKOPE Release Consumption Specification v1](../../docs/specs/release-consumption-v1.md)
- Supersedes: [0001: Separate Dataset Metadata From Storage Lookups](0001-dataset-registry-and-lookup-contract.md)
- Amends: [0002: Mediate COG Tiles Through Skope API](0002-api-mediated-cog-tiles.md)

## Context

SKOPE duplicates dataset metadata across API and ingest YAML files. The ingest
pipeline mutates one of those files with observed raster facts, emits one STAC
Collection per variable, and generates `lookup.json` as the API's storage index.
Existing output files can be reused by filename without proving their identity
or validity. PR 48 proposed splitting curated metadata by dataset and recording
observed facts in `dataset-facts.json`; the `titiler` branch added temporal
endpoint validation and a whole-migration preflight. Both exposed the need for
an explicit authority model, byte validation, and transactional publication.

The design has to respect these forces:

- A temporal asset needs a band-to-time mapping a client can resolve without a
  separate index file.
- Planned output properties must not be confused with observations of the final
  bytes.
- Static rasters must not be forced through temporal-cube requirements.
- A release serves both the SKOPE application and external researchers, so what
  the application needs must be a generated consequence of the release, not a
  parallel curated document.
- Data changes over months or years; presentation changes with an application
  build. Anything on the faster cadence stored in an immutable release forces a
  needless republication or a forbidden mutation.
- Datasets are authored, reviewed, and revised independently, so one dataset's
  change or failure must not republish or block the others.

STAC 1.1.0 and its Projection, File Info, Scientific Citation, Raster,
Datacube, Versioning, and Processing extensions provide standards-based homes
for most published metadata. COG headers describe the encoded bytes directly.
Release completion and integrity are operational facts, not geospatial metadata.

## Decision

- STAC Collections and Items are the published metadata authority. COG headers
  are authoritative for byte-level raster properties, and the corresponding
  STAC fields are validated projections of them.
- A release packages exactly one dataset, with that dataset's Collection as its
  root STAC object, and exactly one manifest recording integrity, provenance,
  inventory, identity, and completion only. Its identifier carries the dataset
  identifier and is not a dataset version.
- Several datasets are served together through a per-environment pin of each
  release's ID, declaration digest, and manifest digest, and discovered through a
  mutable STAC Catalog published outside releases. No aggregate release exists,
  and a multi-dataset build is a batch of independent releases.
- Each release carries one overview generated from its validated STAC, readable
  by people and read once at startup by the application. It carries the
  time-to-band rule: one axis and chunk size per dataset, from which the COG
  path and band of any timestep are computed. `lookup.json` is not emitted.
- The build separates an immutable `ValidatedBuildPlan`, consumed by the COG
  writer, from an immutable `FinalObservation` of the written bytes, consumed by
  metadata serializers.
- A dataset declares a temporal-cube or static-raster profile; static rasters
  get no synthetic time.
- Published releases are immutable and become eligible only once a valid
  manifest, written last, commits them.
- Curated metadata is authored as one `curated.yml` per dataset, compiled to
  STAC, and never contains observed facts.
- Presentation is not a release artifact. It lives in display files in the
  skope-api repository (`deploy/display/`), and the API serves explicit
  rendering fields.
- TiTiler remains internal and API-mediated (ADR 0002). The public metadata and
  time parameters change in one coordinated API/UI cutover, with no
  compatibility period.

## Consequences

- A release can be validated, understood, and relocated on its own, independent
  of any API image or registry file.
- The ingest workflow needs a typed observation model and strict agreement
  checks among COG bytes, STAC, the time-to-band rule, and the manifest.
- Local publication needs same-filesystem staging and atomic rename; object
  storage needs a manifest-last commit protocol.
- One dataset changing does not republish the others, and one dataset failing
  preflight or validation does not block the rest.
- Which datasets a deployment serves becomes an explicit, reviewed pin. A
  consistency check spanning datasets belongs at composition, because a release
  build sees one dataset.
- Nothing carries a single identity for all SKOPE data at a given time; a
  citable corpus identity, if ever needed, should be minted against the
  published catalog.
- A presentation change never requires a new release, but the API and SKOPE UI
  must migrate together to explicit rendering fields.
- Candidate STAC extensions require pinned versions and serialization adapters,
  and ambiguous legacy fields require human review.
- This ADR supersedes ADR 0001 at the coordinated cutover (release consumption
  specification CUT-002), and amends ADR 0002: the tile route's `year`
  parameter becomes a canonical timestep key.
