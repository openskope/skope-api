# Deployment Runbook

This runbook covers the Compose-based development, staging, and production
deployments. Run every command from the repository root. Host provisioning,
DNS, TLS, and the public reverse proxy are owned by `comses/infrastructure`.

| Environment | Checkout/data location | API listener | Deploy command |
| --- | --- | --- | --- |
| Development | Local checkout; data in `./cog-input` | `127.0.0.1:8001` | `make deploy-dev` |
| Staging | `/srv/apps/skope-api`; release selected by `DATASET_RELEASE_ROOT` | `0.0.0.0:8001` | `make deploy-staging` |
| Production | `/srv/apps/skope-api`; release selected by `DATASET_RELEASE_ROOT` | `0.0.0.0:8001` | `make deploy-production` |

## Before deploying

1. Select the exact reviewed commit to deploy. On staging and production,
   update `/srv/apps/skope-api` to that commit using the project's release
   process and confirm that `git status --short` is empty. Do not deploy an
   unreviewed branch or a dirty checkout.
2. Confirm CI passed for that commit. Locally, run the same test suite with:

   ```bash
   make test
   ```

3. If dataset metadata changed, keep `timeseries/metadata.yml` and the relevant
   `deploy/metadata/<environment>.yml` registry in sync. For staging and
   production, confirm `DATASET_RELEASE_ROOT` is readable and that each
   deployed dataset has a valid `{dataset_id}/lookup.json` with all referenced
   COGs present. Follow the [dataset preparation runbook](data-preparation.md)
   to build, validate, and promote dataset artifacts.
4. Coordinate around active extraction jobs. Redis retains job state for 24
   hours, but an API worker restart abandons work executing in that worker.
   Clients must resubmit jobs that remain nonterminal across a deployment.
5. Render the selected Compose configuration before changing services:

   ```bash
   make config ENVIRONMENT=dev
   make config ENVIRONMENT=staging
   make config ENVIRONMENT=prod
   ```

   Only the command for the environment being deployed is required. Render all
   three when changing shared Compose or deployment files.

## Deploy

Staging and production mount one complete, immutable dataset release into
`/data` in both containers. Staging and production require
`DATASET_RELEASE_ROOT` to name an explicit immutable CalVer release directory
beneath `/srv/datasets/releases`; there is no mutable default:

```bash
make deploy-dev
make deploy-staging DATASET_RELEASE_ROOT=/srv/datasets/releases/skope-r-2026.09.12
make deploy-production DATASET_RELEASE_ROOT=/srv/datasets/releases/skope-r-2026.09.12
```

The target builds the selected images with refreshed base images, recreates the
Compose containers with the selected immutable release, removes orphaned
containers, and waits up to 120 seconds for the API, Redis, and TiTiler health
checks. It refuses a staging or production
deploy when the selected release root does not follow the CalVer path
convention or is absent or unreadable. TiTiler remains on the internal Compose
network; public tile requests pass through the API.

## Verify

Set `ENVIRONMENT` to `dev`, `staging`, or `prod`, matching the deployment:

```bash
make ps ENVIRONMENT=<environment>
curl --fail --show-error http://127.0.0.1:8001/metadata
curl --fail --show-error http://127.0.0.1:8001/docs
```

All three services must report healthy, and both HTTP requests must succeed.
For staging and production, also exercise a known dataset through the public
hostname: request its metadata, one representative tile, and—when the release
affects extraction—submit and poll a small extraction. This verifies the
reverse proxy, API-to-TiTiler path, dataset mount, and Redis job path.

Follow service output when verification fails:

```bash
make logs ENVIRONMENT=<environment>
```

## Roll back

1. Record the failed commit and capture relevant logs.
2. Select the previous known-good dataset release and, if necessary, restore
   the checkout to its matching API commit. Confirm the checkout is clean.
3. Run the canonical deploy target with that release's explicit
   `DATASET_RELEASE_ROOT`.
4. Repeat all verification checks above.

Rollback rebuilds the images from the selected commit; image tags alone are
not a rollback mechanism. Treat extraction jobs interrupted by either deploy
as abandoned and resubmit them.

## Routine operations

```bash
make ps ENVIRONMENT=<environment>
make logs ENVIRONMENT=<environment>
make restart ENVIRONMENT=<environment>
make down ENVIRONMENT=<environment>
```

`restart` can interrupt active extractions. `down` stops and removes the
environment's containers and should not be part of a routine deployment.
