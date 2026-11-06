ENVIRONMENT ?= dev
COMPOSE_PROJECT_NAME ?= skope-api
LEGACY_DATA_ROOT ?= /srv/ingest/incoming/skope
MIGRATED_DATA_ROOT ?= timeseries/ingest/output/legacy-migration
MIGRATION_SCRATCH_ROOT ?= timeseries/ingest/output/legacy-migration-scratch
COMPOSE = docker compose --project-name $(COMPOSE_PROJECT_NAME) \
	--project-directory . \
	-f deploy/compose/base.yml \
	-f deploy/compose/$(ENVIRONMENT).yml
# PIN-001: the API and TiTiler mount the release root the environment's pin
# names, resolved to its physical path (release spec TXN-011).
PIN_FILE = deploy/releases/$(ENVIRONMENT).yml
SKOPE_RELEASE_ROOT = $(shell scripts/release-root.sh $(PIN_FILE) 2>/dev/null)
export SKOPE_RELEASE_ROOT
# Tests build their own releases, so their containers mount a fixture directory.
TEST_COMPOSE = SKOPE_RELEASE_ROOT=$(CURDIR)/timeseries/app/tests/registry/data \
	docker compose --project-name $(COMPOSE_PROJECT_NAME)-test \
	--project-directory . \
	-f deploy/compose/base.yml \
	-f deploy/compose/dev.yml

.PHONY: help check-environment check-release-pin verify-releases prepare config build deploy \
	deploy-dev deploy-staging deploy-production down restart logs ps ingest release release-fetch release-pin \
	preflight-legacy-data migrate-legacy-data test test-api test-ingest

# Make 'help' the default target if someone just types `make`
.DEFAULT_GOAL := help

help:   ##- Instructions for using this Makefile.
	@echo "usage: make [target] ..."
	@echo "targets:"
	@sed -e '/#\{2\}-/!d; s/\\$$//; s/:[^#\t]*/:/; s/#\{2\}- *//' $(MAKEFILE_LIST)

check-environment:
	@case "$(ENVIRONMENT)" in \
	  dev|staging|prod) ;; \
	  *) echo "ENVIRONMENT must be dev, staging, or prod (got '$(ENVIRONMENT)')" 1>&2; exit 2;; \
	esac

prepare: check-environment
	@if [ "$(ENVIRONMENT)" = dev ]; then mkdir -p cog-input timeseries/ingest/output/releases; fi

check-release-pin: check-environment
	@scripts/release-root.sh $(PIN_FILE) >/dev/null

config: check-environment ##- Render and validate the selected Compose configuration
	@$(COMPOSE) config

build: prepare ##- Build the selected environment's images
	@$(COMPOSE) config --quiet
	$(COMPOSE) build --pull

verify-releases: check-release-pin ##- Fully verify the pinned releases' bytes (promotion, TXN-011)
	$(COMPOSE) run --rm --no-deps server python -m app.registry.verify

# Deployment state (TXN-011, DISP-011): one JSON line per successful deploy,
# outside git. `git clean -x` deletes it, so point DEPLOY_LOG elsewhere where
# that matters. The API image runs as the calling user, so the files stay theirs.
DEPLOY_LOG ?= deploy-state/$(ENVIRONMENT).jsonl
DEPLOY_LOG_DIR = $(abspath $(dir $(DEPLOY_LOG)))
DEPLOY_PENDING = /deploy-state/.pending-$(ENVIRONMENT).json
DEPLOY_RUN = $(COMPOSE) run --rm --no-deps --user "$$(id -u):$$(id -g)" \
	-v "$(DEPLOY_LOG_DIR)":/deploy-state server

deploy: check-release-pin build ##- Deploy ENVIRONMENT (defaults to dev): verify, start, wait for healthy services, and log the deploy
	@mkdir -p "$(DEPLOY_LOG_DIR)"
	$(DEPLOY_RUN) python -m app.registry.verify --record $(DEPLOY_PENDING)
	$(COMPOSE) up -d --remove-orphans --force-recreate --wait --wait-timeout 120
	$(DEPLOY_RUN) python -m app.registry.deployment $(DEPLOY_PENDING) /deploy-state/$(notdir $(DEPLOY_LOG)) \
		--environment $(ENVIRONMENT) \
		--commit "$$(git rev-parse HEAD)$$(test -z "$$(git status --porcelain)" || echo -dirty)" \
		--deployed-by "$$(id -un)" \
		$$(docker inspect --format '--image {{index .Config.Labels "com.docker.compose.service"}}={{.Image}}' $$($(COMPOSE) ps -q))

deploy-dev: override ENVIRONMENT=dev
deploy-dev: deploy ##- Build and deploy the local development environment

deploy-staging: override ENVIRONMENT=staging
deploy-staging: deploy ##- Build and deploy staging on its application host

deploy-production: override ENVIRONMENT=prod
deploy-production: deploy ##- Build and deploy production on its application host

down: check-environment ##- Stop and remove the selected environment's containers
	$(COMPOSE) down

restart: check-environment ##- Restart the selected environment's services
	$(COMPOSE) restart

logs: check-environment ##- Follow logs for the selected environment
	$(COMPOSE) logs --follow --tail=200

ps: check-environment ##- Show service status for the selected environment
	$(COMPOSE) ps

ingest: prepare ##- Build and run the local COG/STAC ingest pipeline container
	$(COMPOSE) --profile ingest build ingest
	$(COMPOSE) --profile ingest run --rm ingest

RELEASE_ROOT ?= ./timeseries/ingest/output/releases
RELEASE_AUDIT_ROOT ?= ./timeseries/ingest/output/audit
RELEASE_SOURCE_MIRROR ?= ./cog-input
export RELEASE_ROOT RELEASE_AUDIT_ROOT RELEASE_SOURCE_MIRROR

release: prepare ##- Run the dataset release tool, e.g. make release ARGS="preflight paleocar_v3"
	@mkdir -p "$(RELEASE_ROOT)" "$(RELEASE_AUDIT_ROOT)"
	@$(COMPOSE) --profile ingest build release
	PRODUCER_REVISION="$$(git rev-parse HEAD)$$(test -z "$$(git status --porcelain)" || echo -dirty)" \
		$(COMPOSE) --profile ingest run --rm release $(ARGS)

# The mirror is mounted writable only here; preflight and build read it read-only.
release-fetch: prepare ##- Copy declared sources into the source mirror, e.g. make release-fetch ARGS="paleocar_v3"
	@mkdir -p "$(RELEASE_SOURCE_MIRROR)"
	@$(COMPOSE) --profile ingest build release
	$(COMPOSE) --profile ingest run --rm -v "$(abspath $(RELEASE_SOURCE_MIRROR))":/mirror \
		-e SKOPE_RELEASE_SOURCE_MIRROR=/mirror release fetch $(ARGS)

# The ledgers and pin files are mounted writable only here, and the command runs
# as the calling user so the files it writes stay theirs.
release-pin: prepare ##- Show the edits that pin a release, e.g. make release-pin MANIFEST=path/to/release-manifest.json PIN_ENV=staging; add WRITE=1 to apply them
	@test -n "$(MANIFEST)" || { echo "set MANIFEST to a release's release-manifest.json (or a copy of it)" 1>&2; exit 2; }
	@test -n "$(PIN_ENV)" || { echo "set PIN_ENV to the environment to pin it in (dev, staging, or prod)" 1>&2; exit 2; }
	@$(COMPOSE) --profile ingest build release
	$(COMPOSE) --profile ingest run --rm --user "$$(id -u):$$(id -g)" \
		-v "$(abspath timeseries/ingest/datasets)":/pin/datasets \
		-v "$(abspath deploy/releases)":/pin/releases \
		-v "$(abspath $(MANIFEST))":/pin/release-manifest.json:ro \
		release --datasets /pin/datasets pin /pin/release-manifest.json \
		--env $(PIN_ENV) --pins /pin/releases $(if $(WRITE),--write)

preflight-legacy-data: ##- Validate all legacy migration inputs without creating output
	MIGRATION_PREFLIGHT_ONLY=true ./scripts/migrate-legacy-datasets.sh \
		"$(LEGACY_DATA_ROOT)" \
		"$(MIGRATED_DATA_ROOT)" \
		"$(MIGRATION_SCRATCH_ROOT)"

migrate-legacy-data: ##- Preflight and transform legacy cubes into dataset packages
	./scripts/migrate-legacy-datasets.sh \
		"$(LEGACY_DATA_ROOT)" \
		"$(MIGRATED_DATA_ROOT)" \
		"$(MIGRATION_SCRATCH_ROOT)"

##
## Testing
##

.PHONY: test

test: test-api test-ingest ##- Run all API and ingest tests

test-api: override ENVIRONMENT=dev
test-api: prepare ##- Build the development image and run the API tests
	@cmp -s timeseries/ingest/src/skope_release/timeaxis.py timeseries/app/vendor/timeaxis.py || \
		{ echo "timeseries/app/vendor/timeaxis.py differs from the release build's; copy it again" 1>&2; exit 1; }
	@$(TEST_COMPOSE) config --quiet
	$(TEST_COMPOSE) build server titiler
	@trap '$(TEST_COMPOSE) down --remove-orphans' EXIT INT TERM; \
		$(TEST_COMPOSE) run --rm server sh -c \
		'black --check --extend-exclude "app/vendor/" app && pytest -c app/pytest.ini app/tests'

test-ingest: override ENVIRONMENT=dev
test-ingest: prepare ##- Build and run the ingest tests
	@$(TEST_COMPOSE) --profile test config --quiet
	$(TEST_COMPOSE) --profile test build ingest-test
	@trap '$(TEST_COMPOSE) --profile test down --remove-orphans' EXIT INT TERM; \
		$(TEST_COMPOSE) --profile test run --rm --no-deps ingest-test
