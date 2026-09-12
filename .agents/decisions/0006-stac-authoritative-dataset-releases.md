# 0006: Use STAC-Authoritative Immutable Dataset Releases

- Status: Proposed
- Date: 2026-09-11
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
mapping from which `lookup.json` is reproducible; planned output properties must
not be confused with observations of final bytes; and static rasters must not
be forced through temporal-cube requirements.

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
- Release manifests will contain integrity, provenance, object inventory, and
  completion state only. They will not duplicate geospatial or statistical
  metadata.
- Generated API metadata and `lookup.json` indexes will be derived compatibility
  artifacts.
- Temporal lookup indexes will be derived from ordered STAC Band names checked
  against the Collection temporal dimension and COG band descriptions.
- The typed model will have an immutable `ValidatedBuildPlan` consumed by the
  COG writer and a separate immutable `FinalObservation` consumed by metadata
  serializers after byte inspection.
- Dataset releases will declare either a temporal-cube or static-raster profile;
  SRTM will not acquire synthetic scientific time metadata for API convenience.
- Published releases will be immutable and selected only after a valid root
  release manifest acts as the publication commit marker.
- `dataset-facts.json` will not be part of the target architecture.
- Curated human-authored metadata will be compiled to STAC and will not contain
  observed raster facts.
- The existing public API and API-mediated TiTiler boundary will remain
  compatible during the initial migration.

## Consequences

- A dataset release can be validated and understood independently of an API
  image or mutable registry file.
- The ingest workflow needs a typed observation model and strict agreement
  checks among COG bytes, STAC, lookup indexes, and manifests.
- Local publication needs same-filesystem staging and atomic rename. Object
  storage needs a root-manifest-last commit protocol because it has no
  multi-object transaction.
- One logical dataset version is represented by one Collection; temporal Items
  align variable assets by chunk.
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
