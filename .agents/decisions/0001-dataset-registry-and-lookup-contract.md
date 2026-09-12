# 0001: Separate Dataset Metadata From Storage Lookups

- Status: Accepted
- Date: 2026-08-11

## Context

Public dataset metadata and physical raster layout change for different reasons. Encoding storage paths directly in API routes or UI metadata would couple clients to COG organization and make storage migrations difficult.

## Decision

Use the environment's metadata registry to validate datasets and variables. Resolve physical files and bands from `/data/{dataset_id}/lookup.json` or the equivalent configured storage URI. Keep storage resolution inside `skope-api`.

## Consequences

- Every deployed dataset needs a valid, ordered `lookup.json` and all referenced COGs.
- Registry changes may require corresponding edits in `timeseries/metadata.yml` and `deploy/metadata/{dev,staging,prod}.yml`.
- Clients identify datasets, variables, and timesteps without knowing filesystem or object-storage paths.

