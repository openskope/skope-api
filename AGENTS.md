# Skope API Agent Guide

Skope API is a FastAPI service for dataset metadata, COG raster tiles, and asynchronous time-series extraction and analysis. Application code lives under `timeseries/app/`.

## Working Conventions

- Use the root Make targets; tests and services run through Docker Compose.
- The API serves the dataset releases pinned in `deploy/releases/<environment>.yml` (`docs/specs/release-consumption-v1.md`). At startup it checks each pinned release and composes its registry from their `overview.yml` files; any failed check stops the API.
- The pin's `release_root` is mounted read-only at `/releases` in both the API and TiTiler containers; the Makefile reads it from the pin. A timestep resolves to its COG and band through the release's time-to-band rule (`timeseries/app/vendor/timeaxis.py`, a copy of the release build's).
- `timeseries/metadata.yml` and `deploy/metadata/*.yml` are no longer read; they are deleted at the API/UI cutover.
- Keep TiTiler internal. Public raster requests must pass through the API so identifiers are validated and storage paths remain private.
- Extraction is asynchronous: clients submit, poll status, then analyze. Redis job records expire after 24 hours.
- `base_series` is internal job state used by analysis and must never appear in status responses.
- Keep service logs on standard output and standard error for Docker-managed collection.
- `comses/infrastructure` owns host provisioning. `skope-terraform` is retired.

## Validation And Deployment

Run `make test` for the isolated pytest suite. Render Compose changes for `dev`, `staging`, and `prod` with `make config ENVIRONMENT=<environment>`.

Deploy only through `make deploy-dev`, `make deploy-staging`, or `make deploy-production`. Do not restore `./configure`, `config.mk`, or generated root Compose files.

## Git And Decisions

- Use Conventional Commit messages and keep commits logically scoped. Treat the diff as the record of what changed; commit bodies should capture why and any non-obvious decisions.
- Do not create commits unless the user explicitly asks.
- Use scopes only for established repository areas.
- For non-trivial changes, use concise lowercase `-` bullets for rationale, tradeoffs, or operationally relevant details that are not evident from the diff.
- Omit a body when it would only restate the subject or diff.
- Prefer branch-wide issue references in the pull request description. Do not repeat the same issue footer across every commit in a branch. Add a lowercase commit footer only when standalone traceability matters, such as for a direct fix, backport, or independently cherry-picked change.
- Architectural decisions live in `.agents/decisions/`. Add an ADR only for durable, cross-cutting choices; preserve accepted ADRs and supersede them with a new ADR when a decision changes.
- Consider whether a backend feature or change would be significantly simpler with coordinating frontend changes in `openskope/skopeui`.
