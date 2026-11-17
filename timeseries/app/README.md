# Deploying the API

The API serves dataset metadata, map tiles (through an internal TiTiler), and
asynchronous time-series extraction and analysis, from the dataset releases an
environment pins. This guide is for **app developers and operators**: choosing
which releases a deployment serves, how they are shown, and deploying, checking,
and rolling back the Compose stack. Building a release is the other side, in the
[ingest README](../ingest/README.md). The [root README](../../README.md#dataset-releases)
explains what a release is and how the two sides fit together.

The rules are in the [release consumption specification](../../docs/specs/release-consumption-v1.md)
and, for promotion, §14 of the [dataset release specification](../../docs/specs/dataset-release-v1.md).
Host provisioning, DNS, TLS, and the public reverse proxy are owned by
`comses/infrastructure`. Run every command from the repository root.

| Environment | Checkout | Releases served | API listener | Deploy command |
| --- | --- | --- | --- | --- |
| Development | Local checkout | `deploy/releases/dev.yml` | `127.0.0.1:8001` | `make deploy-dev` |
| Staging | `/srv/apps/skope-api` | `deploy/releases/staging.yml` | `0.0.0.0:8001` | `make deploy-staging` |
| Production | `/srv/apps/skope-api` | `deploy/releases/prod.yml` | `0.0.0.0:8001` | `make deploy-production` |

Make runs the development API and tests with the invoking user's UID/GID, so
bind-mounted source stays readable and generated files belong to that user.
Staging and production API containers, and TiTiler in every environment, keep
their fixed unprivileged image identity. No host account is baked into an image.

## Serving releases

A deployment serves exactly the releases its environment pins. At startup the
API reads each pinned release's `overview.yml` once, verifies it, and composes
the overviews unchanged into the app registry. It never discovers datasets on
its own.

A deploy that adds or updates a dataset is usually one change containing:

- the pin file entry ([Pin the release](#pin-the-release)); and
- display entries for any new dataset or variable
  ([Add display entries](#add-display-entries)), plus a palette if a new one is
  needed.

Put both in the same change. A pinned variable without a display entry stops
the API from starting.

### Pin the release

Each environment has one pin file, `deploy/releases/<env>.yml`:

```yaml
release_root: /srv/datasets/releases
releases:
  - dataset: paleocar_v3
    release_id: paleocar_v3-r-2026.11.04
    declaration_digest: ebb5bb56…
    manifest_sha256: 899b8be2…
```

- **`release_root`:** the single local directory every pinned release sits
  under, as a resolved physical path. A release built somewhere else has to be
  copied there first. The API and TiTiler mounts are derived from this value,
  so it is the only place the root is configured. Object storage is not
  supported yet (TXN-012).
- **`release_id`:** the release's directory name under the root.
- **`declaration_digest`:** the manifest's `declaration.digest` (or
  `declaration_digest` in `overview.yml`).
- **`manifest_sha256`:** the SHA-256 of the `release-manifest.json` file itself,
  in hex. This digest pins the published bytes, because the manifest records
  the checksum of every other file. The declaration digest pins only the inputs.

`make release-pin` makes these edits from the release's manifest, so nothing is
copied by hand. Run it where the release is, or anywhere with a copy of its
`release-manifest.json`:

```bash
make release-pin MANIFEST=<release_root>/<release_id>/release-manifest.json PIN_ENV=staging
```

It changes nothing; it prints, as a diff, the three edits it would make:

- the ledger entry in `timeseries/ingest/datasets/<dataset>/releases.yml`;
- the dataset's entry in `deploy/releases/<PIN_ENV>.yml`, replaced or added,
  leaving every other entry and comment alone (delete a comment that no longer
  applies, such as one describing the old release);
- a `checksum` under the `uri` of each source declared without one, as the
  build measured it, and for a prepared source under each input and the
  output. This leaves the declaration digest unchanged.

It refuses, and edits nothing, if the ledger records that release ID with a
different digest, if a declared checksum differs from the measured one, or if
the source manifest no longer lists the release's variables. Read the diff, then
run the same command with `WRITE=1` to apply it, and review `git diff` before
committing. To pin several releases in one change, run it once per manifest;
each run adds to the files the previous one wrote.

Rules:

- Change only the entry for the dataset you are updating. Promoting one dataset
  never requires re-selecting the others.
- Each dataset appears at most once.
- To stop serving a dataset, remove its entry. The display files cannot hide a
  pinned dataset, because display choices are not an access control.
- Environments differ only in their pin files; the display files are shared, so
  every environment renders a variable the same way.

### Add display entries

Presentation lives in two files shared by all environments:
`deploy/display/preferences.yml` (how each dataset and variable is shown) and
`deploy/display/palettes.yml` (named colour lists). Changing them never needs a
new release.

```yaml
datasets:
  paleocar_v3:
    order: 10
    map_view:                  # optional; default fits the dataset's bbox
      center: { lon: -108.5, lat: 37.0 }
      zoom: 4.25
    default_variable: ppt_annual   # optional; default is the lowest order
    variables:
      ppt_annual:
        order: 10
        palette: skope-precip
        range: [0, 1807.5]
        note: "Half the legacy registry's maximum for prediction_scaled"
```

Checklist:

- [ ] Every pinned dataset has a dataset entry, and every variable of every
  pinned release has a variable entry. Entries for datasets an environment does
  not pin are fine.
- [ ] Every dataset and variable has a numeric `order`. Clients sort by it,
  then by identifier; position in the file means nothing.
- [ ] Every variable names a `palette`, defined in `palettes.yml`. Common
  palettes such as `viridis` are copied in as colour lists too.
- [ ] **Continuous variable:** a `range` with finite endpoints, lower less than
  upper, in physical values (after scale and offset). It sets both the tile
  colours and the legend, used exactly as written. If a percentile or outlier
  rule chose the endpoints, say so in `note`. The palette's `kind` is `ramp`.
- [ ] **Categorical variable** (exactly the variables whose `curated.yml`
  declares category meanings): the palette's `kind` is `set`, with at least as
  many colours as categories, bound to categories in order. Optional `labels`
  may shorten a category's curated meaning, but only for categories the curated
  metadata declares.
- [ ] Explicit `ticks`, if any, lie within the range.
- [ ] `map_view`, if any, uses named `lon` and `lat` keys within WGS 84 ranges.
- [ ] `default_variable`, if any, names a variable of the pinned release.

TiTiler registers every palette as a named colormap when its image is built, and
the API reads legend colours from the same file. A palette edit therefore
rebuilds the TiTiler image, and both images must be built from the same commit.

### What is checked where

| When | Checks | Needs release data? |
| --- | --- | --- |
| **CI**, on every change to the display files | Referenced palettes exist; `kind` matches a stated `type`; `palette[i]` indexes in range; ranges finite with lower < upper; explicit ticks within range; numeric `order`; `map_view` keys and ranges; no unknown fields | No |
| **Deployment**, by default, for every pinned release | Every file's size and checksum against its manifest (full byte verification) | Yes |
| **Application-only redeploy**, with `SKIP_FULL_VERIFICATION=1` | Startup release checks below; asset contents are not hashed | Yes, overviews, manifests, and file metadata |
| **API startup**, for every pinned release | Manifest SHA-256 matches the pin; the manifest is valid for this API; `overview.yml` checksum matches its manifest entry; the overview's release ID and declaration digest match the pin; every inventoried file exists with its recorded size; no dataset pinned twice; every pinned dataset and variable has a display entry; `type` agrees with curated categories; `default_variable` exists; `set` palettes have enough colours | Yes, overviews and manifests only |

The API serves temporal datasets only; it refuses a pinned static raster
(PROTO-008).

## Before deploying

1. Select the exact reviewed commit to deploy. On staging and production,
   update `/srv/apps/skope-api` to that commit using the project's release
   process and confirm that `git status --short` is empty. Do not deploy an
   unreviewed branch or a dirty checkout.
2. Confirm CI passed for that commit. The workflow also runs weekly and on
   manual dispatch, pulling fresh runtime bases, testing real containers, and
   scanning for fixable high/critical vulnerabilities. Locally, run:

   ```bash
   make test BUILD_FLAGS=--pull
   make test-runtime
   make scan-runtime-images
   ```

3. If the releases served change, the commit must pin them and add display
   entries for any new dataset or variable ([Serving releases](#serving-releases)).
   Every pinned release must be present under the pin's `release_root` on the
   host and readable by every user.
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

On either application host, use a clean checkout of the commit to deploy and
run the matching target:

```bash
make deploy-staging
make deploy-production
```

This requires host `python3`, Docker Compose, and the GitHub CLI (`gh`)
authenticated with permission to read the repository's Actions artifacts.
It finds the newest successful `test.yml` run for that exact commit with an
unexpired bundle from its current attempt, downloads the bundle, loads its
images, and runs the deployment checks below. It never switches commits or
builds staging or production images locally. A missing bundle fails with
instructions to run CI or select an existing local bundle explicitly.

Bundles are cached outside the checkout under
`${XDG_CACHE_HOME:-~/.cache}/skope-api/image-bundles/`; set
`STAGING_BUNDLES_DIR` or `PRODUCTION_BUNDLES_DIR` to use another directory outside
the checkout for that environment. They remain available for rollback until an
operator removes them. Discovery uses
`openskope/skope-api`; set `STAGING_CI_REPOSITORY=owner/repo` or
`PRODUCTION_CI_REPOSITORY=owner/repo` to use a fork's CI for that environment.
Automatic selection requires GitHub access even for a cached bundle. For offline
deployment or rollback, pass its `images-staging.json` or `images-prod.json` as
`DEPLOY_IMAGES`.

Automatic selection uses the newest available successful bundle for the commit.
If CI rebuilt that commit after staging was tested, production may select a
different bundle. To promote the exact staging-tested bundle, explicitly select
that bundle's `images-prod.json` using `DEPLOY_IMAGES`.

For explicit bundle selection in either environment, download the
`runtime-images-<run>-<attempt>` artifact from a successful default-branch or
manually dispatched CI run for the selected commit. Pull requests run the checks
without retaining large image archives. Extract the artifact outside the checkout
and load its images on the application host:

```bash
docker load --input /srv/apps/image-bundles/<run>/images.tar.gz
make deploy-staging DEPLOY_IMAGES=/srv/apps/image-bundles/<run>/images-staging.json
make deploy-production DEPLOY_IMAGES=/srv/apps/image-bundles/<run>/images-prod.json
```

The selections name exact local image IDs for the API, TiTiler, and Redis.
Make refuses a missing image, a different environment or source revision, or a
dirty checkout. It also requires the bundle's successful scan manifest and checks
that its image IDs match the selection. It requires host `python3` for these
checks. Each bundle contains staging and production API images with their
environment-specific configuration, tested separately with the same TiTiler
and Redis.
GitHub-hosted CI builds Linux amd64 images; the target host must support that
platform. Download bundles you intend to deploy or retain before their 30-day
Actions retention expires. Keep the current and previous known-good bundles
on the host for rollback; neither deploy nor image export prunes them.

Development still builds locally with `make deploy-dev`. To reproduce a CI
bundle locally, run the checks above followed by `make export-runtime-images`.
Bundles built from a dirty checkout are marked `-dirty` and cannot be deployed
to staging or production.

TiTiler follows upstream `latest`; Python and Redis follow their existing
version-scoped tags. The Dockerfiles upgrade installed OS packages while
building the API (Debian), TiTiler (Wolfi), and Redis (Alpine) replacement
images. CI tests the actual runtime images, including custom palettes,
ZSTD tile-interleaved COGs, one-band VRT reads, transparent nodata,
extraction values and subsequent analysis. The ingest GDAL image stays pinned
because it produces reproducible scientific release artifacts. A newer runtime
is never fetched during staging or production deployment.

The runtime smoke test also runs the shared Make deployment recipe against
synthetic releases, checking both verification modes, HTTP readiness, and the logged
image IDs before testing tiles, extraction, and analysis.

Full byte verification remains the default. For an application-only deployment
using releases already fully verified on this host, explicitly request the faster
startup checks:

```bash
make deploy-staging SKIP_FULL_VERIFICATION=1
make deploy-production SKIP_FULL_VERIFICATION=1
```

This still checks manifest digests and release identities, overview checksums,
file existence and sizes, and all other startup checks. It does not read and hash
the COG contents, so same-size asset corruption can go undetected. The deployment
log records `verification: metadata` instead of `full`. Use the default full check
when promoting new dataset releases. `make verify-releases ENVIRONMENT=staging`
(or `prod`) always performs full verification, regardless of this flag.

The pin file selects the releases; the target takes no release argument. It:

1. checks the loaded CI image selection against the clean source checkout
   (or builds fresh local images in development);
2. by default, fully verifies every pinned release against its manifest, logging its
   progress. This reads every byte, about an hour for a 60 GB release, and runs
   on every deploy, including a rollback or a redeploy of the same releases,
   unless `SKIP_FULL_VERIFICATION=1` selects the faster checks above.
   `make verify-releases` runs the same verification on its own;
3. recreates changed Compose services, removing orphaned ones, with the pin's
   release root mounted read-only at `/releases` in the API and TiTiler, and
   waits up to 180 seconds for the API, Redis, and TiTiler health checks.
   TiTiler's HTTP readiness probe allows 120 seconds for startup before failed
   checks count toward an unhealthy result; readiness can succeed sooner.

   Unchanged Redis containers stay running, preserving cached job records. Redis
   persistence is disabled, so recreating Redis or restarting its process still
   discards those records.

4. appends one line describing the deploy to the
   [deployment log](#the-deployment-log).

It stops before changing any container when the pin's release root is missing
or any pinned release fails verification, and adds no log line when any step
fails. A symlink or other mutable pointer is never used for a release path.

The API then runs its startup checks ([What is checked where](#what-is-checked-where)).
**Any failure stops the whole API**, not just the dataset at fault: it does not
fall back to an older release or serve the remaining datasets. The API logs one
structured error per failed check, naming the requirement ID, dataset, release
ID, and the expected and observed values, then exits with a failure status. One
bad pin entry therefore takes the environment down, which is why promotion of
new releases must check every byte first.

TiTiler remains on the internal Compose network; public tile requests pass
through the API. Run deploys inside `tmux` or similar on staging and
production, so a dropped connection does not stop one.

### Weekly security maintenance

The application maintainer owns a weekly maintenance deployment. Monday's CI
run at 06:00 America/Phoenix prepares a candidate automatically; an operator
promotes it through staging and production during that week's maintenance
window. A passing build alone does not patch the running services.

1. **Review dependency updates.** Dependabot proposes API package, base image,
   and GitHub Actions updates weekly. Scientific libraries and the release
   toolchain are grouped separately for review; ingest lockfile updates are
   proposed monthly. Review and merge appropriate updates before selecting the
   candidate commit. Major upgrades and changes to scientific encoding or
   numerical behavior need their own review. Enable Dependabot alerts and
   security updates in the repository settings as well; version-update PRs do
   not replace vulnerability alerts.
2. **Build the patched candidate.** Scheduled and manually dispatched workflow
   runs pull current bases and disable runtime layer reuse (`--pull
   --no-cache`). OS upgrades and package installation therefore run even when a
   base tag has not moved. BuildKit download caches still avoid unnecessary
   downloads. Unit tests, real staging/production container checks and the
   vulnerability gate must pass before an image archive is produced. The
   manifest records build time, maintenance mode, image IDs, and scan time.
   To reproduce this process locally:

   ```bash
   make maintenance-candidate
   ```

3. **Deploy to staging.** Retain the current and previous known-good bundles,
   load the new archive, and use `make deploy-staging DEPLOY_IMAGES=...` from its
   clean source checkout. Check the public `/metadata`, a known dataset's map
   and legend, and an extraction followed by analysis in the UI. Also check
   service logs and the deployment log. Container health checks make actual
   HTTP requests; the API has 45 seconds to stop gracefully. A Redis upgrade
   restarts its process and discards cached jobs because persistence is disabled.
   Choose a low-traffic window and expect interrupted jobs to need resubmission.
4. **Promote the same bundle.** After staging passes, run
   `make deploy-production DEPLOY_IMAGES=.../images-prod.json`, then repeat the
   public checks and inspect errors and resource use. Do not rebuild between
   staging and production or install packages inside running containers. If
   verification fails, use the [rollback procedure](#roll-back) with the prior
   bundle and its source commit. Keep that bundle until a later deployment is
   verified.
5. **Handle failures and urgent fixes.** A failed candidate leaves the current
   deployment running and retains scan reports. Review the finding, update its
   source package or base, and rerun the workflow; do not bypass the gate to
   complete a routine maintenance window. For an urgent security fix, dispatch
   the same workflow immediately and expedite staging and production promotion
   instead of waiting for Monday. The runtime process does not rebuild or
   mutate published scientific dataset releases.

The `comses/infrastructure` maintainers separately patch and reboot hosts,
update Docker Engine and the reverse proxy, and maintain TLS. Container package
upgrades do not patch the host kernel or Docker daemon. Coordinate host
maintenance with the same verification and rollback window.

### The first production deployment

Production pins no release yet, so the API refuses to start there. Its first
deployment is also the switch to the new `/metadata` contract, so the API and
SKOPE UI ship as one pair: they pass the staging acceptance suite together,
deploy as one recorded change, and roll back together if either fails
(CUT-001 to CUT-003). After that, updating a dataset is an ordinary pin change.

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
3. Load the previous known-good CI bundle if necessary, then run the canonical
   deploy target with that bundle's `DEPLOY_IMAGES` selection for the environment.
   It verifies the pinned releases in full again.
4. Repeat all verification checks above.

Rollback uses the previously tested images without rebuilding against today's
upstream tags. A rollback never modifies or deletes a release. Treat
extraction jobs interrupted by either deploy as abandoned and resubmit them.
Completed jobs retain their source release ID, declaration digest, and temporal
precision in their internal record. Analysis returns HTTP 410 if the currently
served release differs, or if a job uses an older, unsupported record format;
clients must submit a new extraction. These internal fields are not included
in job status responses.

## The deployment log

Each successful `make deploy-<env>` appends one JSON line to
`deploy-state/<env>.jsonl` in the checkout: when, where, and by whom it was
deployed, the commit the images (and so the palettes file) were built from,
marked `-dirty` if the checkout had uncommitted changes, the ID of each running
image, and every pinned release with its digests, path, and verification
result. This is the deployment state TXN-011 and DISP-011 ask for: what an
environment served, from when, and verified how. The pin's git history shows
what was meant to be deployed; the log shows what went live.

```json
{"deployed_at": "2026-11-05T14:02:11Z", "environment": "staging", "deployed_by": "…",
 "commit": "8fbfa9a…", "images": {"redis": "sha256:…", "server": "sha256:…", "titiler": "sha256:…"},
 "releases": [{"dataset": "paleocar_v3", "release_id": "paleocar_v3-r-2026.11.04",
               "declaration_digest": "ebb5…", "manifest_sha256": "899b…",
               "path": "/releases/paleocar_v3-r-2026.11.04", "verification": "full", "result": "passed"}]}
```

A deploy that fails verification or startup adds no line; its output says why.
The log is ignored by git, so `git clean -x` or a fresh clone loses it. Where
that matters, keep it elsewhere with `DEPLOY_LOG=<path>` on every deploy.

## Routine operations

```bash
make ps ENVIRONMENT=<environment>
make logs ENVIRONMENT=<environment>
make restart ENVIRONMENT=<environment>
make down ENVIRONMENT=<environment>
```

`restart` can interrupt active extractions. `down` stops and removes the
environment's containers and should not be part of a routine deployment.

## Changing things later

| Change | New release? | Where |
| --- | --- | --- |
| Colours, palettes, visualization ranges, ticks, labels | No | Display files; a palette edit rebuilds the TiTiler image |
| Dataset or variable ordering, map view, default variable | No | Display files |
| Which release an environment serves, or hiding a dataset | No | Pin file |
| Anything in a release's data or metadata | Yes | The release side ([ingest README](../ingest/README.md#changing-things-later)) |

Replacing a release needs nothing done to the old one: pinning the new release
is the whole change, and the old one stays where it is, unchanged, as the
rollback target.

## Troubleshooting and reference

### Common failures

| Stage | Symptom | Likely cause | Requirement |
| --- | --- | --- | --- |
| CI (display files) | Display check fails | Undefined palette, `kind` mismatch, range or tick out of order, non-numeric `order`, positional `map_view`, unknown field | DISP-012 |
| Promotion | Checksum or size mismatch | Release bytes changed or were copied incompletely | TXN-011 |
| Serving | Map tiles return 502 "Upstream tile server error" and extraction fails; the API and TiTiler logs show `Permission denied` | The release is not readable by uid 1000, which the API and TiTiler run as, e.g. one copied by hand. Fix with `chmod` to 0444 files and 0555 directories; the bytes and checksums are unchanged | TXN-018 |
| API startup | Digest or identity mismatch | Typo in the pin, or `manifest_sha256` taken from a different file | PIN-004 |
| API startup | Release manifest is not valid | The release was built by a newer release tool than this API understands; deploy an API that does | PIN-004 |
| API startup | Duplicate dataset | Two pin entries for the same dataset | PIN-005 |
| API startup | Missing display entry | A dataset or variable pinned without display entries | DISP-001, DISP-013 |
| API startup | Display does not fit the data | `type` disagrees with curated categories, missing `default_variable`, too few colours in a `set` | DISP-002, DISP-007, DISP-010, DISP-013 |

### Where each step is defined

| Step | Release consumption spec section |
| --- | --- |
| Pin file, startup verification, app registry | §3 (PIN-001–006) |
| Display preferences and palettes | §4.1–4.2 (DISP-001–011) |
| CI and startup display checks | §4.3 (DISP-012, DISP-013) |
| `/metadata`, tiles, extraction | §5–6 (PROTO-001–011) |
| API/UI cutover | §9 (CUT-001–003) |

Promotion, mounts, and file permissions are in the dataset release
specification, §14 (TXN-001–018).
