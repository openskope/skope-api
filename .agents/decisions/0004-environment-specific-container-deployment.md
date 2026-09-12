# 0004: Use Explicit Environment-Specific Container Deployments

- Status: Accepted
- Date: 2026-09-09

## Context

The previous `./configure` workflow generated local deployment state and obscured which Compose layers and settings were active.

## Decision

Use root Make targets and layered Compose files for `dev`, `staging`, and `prod`. Build settings and metadata into each API image, wait for API, Redis, and TiTiler health during deployment, and keep service logs on standard output and standard error.

Host provisioning remains owned by `comses/infrastructure`. Staging and production use `/srv/apps/skope-api`, mount `/srv/datasets` read-only at `/data`, and expose the API on host port `8001`.

## Consequences

- `make deploy-dev`, `make deploy-staging`, and `make deploy-production` are the canonical deployment commands.
- Compose changes must render successfully for every environment.
- `skope-terraform`, `./configure`, `config.mk`, and generated root Compose workflows are retired.

