# Deployment Runbook

This runbook covers the Compose-based development, staging, and production
deployments. Run every command from the repository root. Host provisioning,
DNS, TLS, and the public reverse proxy are owned by `comses/infrastructure`.
Building, pinning, and promoting dataset releases is described in the
[release workflow](draft-release-workflow.md).

| Environment | Checkout | Releases served | API listener | Deploy command |
| --- | --- | --- | --- | --- |
| Development | Local checkout | `deploy/releases/dev.yml` | `127.0.0.1:8001` | `make deploy-dev` |
| Staging | `/srv/apps/skope-api` | `deploy/releases/staging.yml` | `0.0.0.0:8001` | `make deploy-staging` |
| Production | `/srv/apps/skope-api` | `deploy/releases/prod.yml` | `0.0.0.0:8001` | `make deploy-production` |

## Before deploying

1. Select the exact reviewed commit to deploy. On staging and production,
   update `/srv/apps/skope-api` to that commit using the project's release
   process and confirm that `git status --short` is empty. Do not deploy an
   unreviewed branch or a dirty checkout.
2. Confirm CI passed for that commit. Locally, run the same test suite with:

   ```bash
   make test
   ```

3. If the releases served change, the commit must pin them in the
   environment's `deploy/releases/<environment>.yml` (`make release-pin`) and
   add display entries for any new dataset or variable; see the release
   workflow, section 4. Every pinned release must be present under the pin's
   `release_root` on the host and readable by every user.
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

```bash
make deploy-dev
make deploy-staging
make deploy-production
```

The pin file selects the releases; the target takes no release argument. It:

1. builds the selected images with refreshed base images;
2. fully verifies every pinned release against its manifest, logging its
   progress. This reads every byte, about an hour for a 60 GB release, and runs
   on every deploy, including a rollback or a redeploy of the same releases;
3. recreates the Compose containers, removing orphaned ones, with the pin's
   release root mounted read-only at `/releases` in the API and TiTiler, and
   waits up to 120 seconds for the API, Redis, and TiTiler health checks;
4. appends one line describing the deploy to `deploy-state/<environment>.jsonl`
   (release workflow, section 4.7).

It stops before changing any container when the pin's release root is missing
or any pinned release fails verification, and adds no log line when any step
fails. At startup the API checks every pinned release again and refuses to
start on any mismatch. TiTiler remains on the internal Compose network; public
tile requests pass through the API. Run deploys inside `tmux` or similar on
staging and production, so a dropped connection does not stop one.

## Verify

Set `ENVIRONMENT` to `dev`, `staging`, or `prod`, matching the deployment:

```bash
make ps ENVIRONMENT=<environment>
curl --fail --show-error http://127.0.0.1:8001/metadata
curl --fail --show-error http://127.0.0.1:8001/docs
tail -n 1 deploy-state/<environment>.jsonl
```

All three services must report healthy, both HTTP requests must succeed, and
the last log line must show this deploy's time and commit. For staging and
production, also exercise a known dataset through the public hostname: request
its metadata, one representative tile, and—when the release affects
extraction—submit and poll a small extraction. This verifies the reverse proxy,
API-to-TiTiler path, release mount, and Redis job path.

Follow service output when verification fails:

```bash
make logs ENVIRONMENT=<environment>
```

## Roll back

1. Record the failed commit and capture relevant logs.
2. Revert the commit that changed the pin (and any display change made with
   it), or restore the checkout to the previous known-good commit. Confirm the
   checkout is clean.
3. Run the canonical deploy target. It verifies the pinned releases in full
   again.
4. Repeat all verification checks above.

Rollback rebuilds the images from the selected commit; image tags alone are
not a rollback mechanism. A rollback never modifies or deletes a release. Treat
extraction jobs interrupted by either deploy as abandoned and resubmit them.

## Routine operations

```bash
make ps ENVIRONMENT=<environment>
make logs ENVIRONMENT=<environment>
make restart ENVIRONMENT=<environment>
make down ENVIRONMENT=<environment>
```

`restart` can interrupt active extractions. `down` stops and removes the
environment's containers and should not be part of a routine deployment.
