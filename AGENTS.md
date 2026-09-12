# Skope API Agent Guide

Skope API is a FastAPI service for dataset metadata, COG raster tiles, and asynchronous time-series extraction and analysis. Application code lives under `timeseries/app/`.

## Working Conventions

- Use the root Make targets; tests and services run through Docker Compose.
- `timeseries/metadata.yml` is the public dataset registry. Image builds also consume `deploy/metadata/{dev,staging,prod}.yml`, so registry changes may require updates in both places.
- Dataset storage is resolved through `{dataset_id}/lookup.json`. Staging and production mount `/srv/datasets` read-only at `/data` in both the API and TiTiler containers.
- Keep TiTiler internal. Public raster requests must pass through the API so identifiers are validated and storage paths remain private.
- Extraction is asynchronous: clients submit, poll status, then analyze. Redis job records expire after 24 hours.
- `base_series` is internal job state used by analysis and must never appear in status responses.
- Keep service logs on standard output and standard error for Docker-managed collection.
- `comses/infrastructure` owns host provisioning. `skope-terraform` is retired.

## Validation And Deployment

Run `make test` for the isolated pytest suite. Render Compose changes for `dev`, `staging`, and `prod` with `make config ENVIRONMENT=<environment>`.

Deploy only through `make deploy-dev`, `make deploy-staging`, or `make deploy-production`. Do not restore `./configure`, `config.mk`, or generated root Compose files.

## Git And Decisions

- Use Conventional Commit messages and keep commits logically scoped.
- Do not create commits unless the user explicitly asks.
- Architectural decisions live in `.agents/decisions/`. Add an ADR only for durable, cross-cutting choices; preserve accepted ADRs and supersede them with a new ADR when a decision changes.
