ENVIRONMENT ?= dev
COMPOSE_PROJECT_NAME ?= skope-api
export SKOPE_BUILD_ID ?= $(shell git describe --tags --always --dirty)
BUILD_FLAGS ?=
DEPLOY_IMAGES ?=
LOCAL_BUILD ?= 0
PULL ?= 1
PR ?=
export PR LOCAL_BUILD DEPLOY_IMAGES
export FULL_VERIFICATION ?= 0
RUNTIME_IMAGES_DIR ?= runtime-images
RUNTIME_REFRESH ?= 0
export RUNTIME_IMAGES_DIR RUNTIME_REFRESH
# Bind-mounted source and generated files belong to the invoking operator.
export SKOPE_OPERATOR_UID := $(shell id -u)
export SKOPE_OPERATOR_GID := $(shell id -g)
COMPOSE = docker compose --project-name $(COMPOSE_PROJECT_NAME) \
	--project-directory . \
	-f deploy/compose/base.yml \
	-f deploy/compose/$(ENVIRONMENT).yml \
	$(if $(DEPLOY_IMAGES),-f "$(abspath $(DEPLOY_IMAGES))")
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

.PHONY: help check-environment check-verification-mode check-deploy-source check-release-pin verify-releases prepare prepare-release prepare-deploy config build deploy \
	deploy-dev deploy-staging deploy-production down restart logs ps release release-fetch release-pin \
	test test-api test-ingest test-runtime scan-runtime-images export-runtime-images maintenance-candidate

# Make 'help' the default target if someone just types `make`
.DEFAULT_GOAL := help

help:   ##- Instructions for using this Makefile.
	@echo "usage: make [target] ..."
	@echo "targets:"
	@sed -e '/#\{2\}-/!d; s/\\$$//; s/:[^#\t]*/:/; s/#\{2\}- *//' $(MAKEFILE_LIST)

check-environment:
	@case "$(PULL)" in 0|1) ;; *) echo 'PULL must be 0 or 1' >&2; exit 2;; esac
	@case "$(ENVIRONMENT)" in \
	  dev|staging|prod) ;; \
	  *) echo "ENVIRONMENT must be dev, staging, or prod (got '$(ENVIRONMENT)')" 1>&2; exit 2;; \
	esac

prepare: check-environment
	@if [ "$(ENVIRONMENT)" = dev ]; then mkdir -p cog-input timeseries/ingest/output/releases; fi

check-release-pin: check-environment
	@scripts/release-root.sh $(PIN_FILE) >/dev/null

check-verification-mode:
	@if [ -n "$(SKIP_FULL_VERIFICATION)" ]; then \
		echo 'SKIP_FULL_VERIFICATION was replaced by FULL_VERIFICATION=1 or 0' >&2; exit 2; \
	fi
	@case "$(FULL_VERIFICATION)" in \
	  0|1) ;; \
	  *) echo 'FULL_VERIFICATION must be 0 or 1' >&2; exit 2;; \
	esac

check-deploy-source:
	@case "$$LOCAL_BUILD" in 0|1) ;; *) echo 'LOCAL_BUILD must be 0 or 1' >&2; exit 2;; esac
	@if [ "$$LOCAL_BUILD" = 1 ]; then \
		test -z "$$PR$$DEPLOY_IMAGES" || { echo 'LOCAL_BUILD cannot be combined with PR or DEPLOY_IMAGES' >&2; exit 2; }; \
		status=$$(git status --porcelain) || exit $$?; \
		test -z "$$status" || { echo 'deploy local images from a clean checkout' >&2; exit 2; }; \
	fi

config: check-environment ##- Render and validate the selected Compose configuration
	@$(COMPOSE) config

build: prepare ##- Build the selected environment's images
	@$(COMPOSE) config --quiet
	$(COMPOSE) build $(if $(filter 1,$(PULL)),--pull)

verify-releases: check-release-pin ##- Fully verify the pinned releases' bytes (promotion, TXN-011)
	$(COMPOSE) run --rm --no-deps server python -m app.registry.verify

# Deployment state (TXN-011, DISP-011): one JSON line per successful deploy,
# outside git. `git clean -x` deletes it, so point DEPLOY_LOG elsewhere where
# that matters. The API image runs as the calling user, so the files stay theirs.
DEPLOY_LOG ?= deploy-state/$(ENVIRONMENT).jsonl
export DEPLOY_LOG
DEPLOY_LOG_DIR = $(abspath $(dir $(DEPLOY_LOG)))
DEPLOY_PENDING = /deploy-state/.pending-$(ENVIRONMENT).json
DEPLOY_RUN = $(COMPOSE) run --rm --no-deps --pull never --user "$(SKOPE_OPERATOR_UID):$(SKOPE_OPERATOR_GID)" \
	-v "$(DEPLOY_LOG_DIR)":/deploy-state server

prepare-deploy: prepare check-deploy-source
	@if [ -n "$(DEPLOY_IMAGES)" ]; then \
		python3 scripts/runtime-images.py check "$(DEPLOY_IMAGES)" "$(ENVIRONMENT)"; \
	elif [ "$(ENVIRONMENT)" = dev ] || [ "$$LOCAL_BUILD" = 1 ]; then \
		$(MAKE) build ENVIRONMENT=$(ENVIRONMENT); \
	else \
		echo 'set DEPLOY_IMAGES to a tested bundle selection or LOCAL_BUILD=1 to build here (see timeseries/app/README.md)' >&2; exit 2; \
	fi

deploy: check-verification-mode check-release-pin prepare-deploy ##- Deploy tested images, check release metadata (FULL_VERIFICATION=1 to hash all files), wait for health, and log the deploy
	@mkdir -p "$(DEPLOY_LOG_DIR)"
	$(DEPLOY_RUN) python -m app.registry.verify $(if $(filter 1,$(FULL_VERIFICATION)),,--metadata-only) --record $(DEPLOY_PENDING)
	$(COMPOSE) up -d --no-build --pull never --remove-orphans --wait --wait-timeout 180
	$(DEPLOY_RUN) python -m app.registry.deployment $(DEPLOY_PENDING) /deploy-state/$(notdir $(DEPLOY_LOG)) \
		--environment $(ENVIRONMENT) \
		--commit "$$(git rev-parse HEAD)$$(test -z "$$(git status --porcelain)" || echo -dirty)" \
		--deployed-by "$$(id -un)" \
		$$(docker inspect --format '--image {{index .Config.Labels "com.docker.compose.service"}}={{.Image}}' $$($(COMPOSE) ps -q))

deploy-dev: override ENVIRONMENT=dev
deploy-dev: deploy ##- Build and deploy the local development environment

deploy-staging: override ENVIRONMENT=staging
deploy-staging: check-verification-mode check-deploy-source ##- Deploy staging (PR=69 selects CI work; LOCAL_BUILD=1 builds here)

deploy-production: override ENVIRONMENT=prod
deploy-production: check-verification-mode check-deploy-source ##- Deploy production CI images (LOCAL_BUILD=1 builds here)

deploy-staging deploy-production:
	@if [ "$$LOCAL_BUILD" = 1 ]; then \
		$(MAKE) deploy ENVIRONMENT=$(ENVIRONMENT); \
	elif [ -n "$$PR" ]; then \
		python3 scripts/runtime-images.py $@; \
	elif [ -n "$(DEPLOY_IMAGES)" ]; then \
		$(MAKE) deploy ENVIRONMENT=$(ENVIRONMENT); \
	else \
		python3 scripts/runtime-images.py $@; \
	fi

down: check-environment ##- Stop and remove the selected environment's containers
	$(COMPOSE) down

restart: check-environment ##- Restart the selected environment's services
	$(COMPOSE) restart

logs: check-environment ##- Follow logs for the selected environment
	$(COMPOSE) logs --follow --tail=200

ps: check-environment ##- Show service status for the selected environment
	$(COMPOSE) ps

RELEASE_ROOT ?= ./timeseries/ingest/output/releases
RELEASE_AUDIT_ROOT ?= ./timeseries/ingest/output/audit
RELEASE_SOURCE_MIRROR ?= ./cog-input
export RELEASE_ROOT RELEASE_AUDIT_ROOT RELEASE_SOURCE_MIRROR

prepare-release: prepare
	@mkdir -p "$(RELEASE_ROOT)" "$(RELEASE_AUDIT_ROOT)" "$(RELEASE_SOURCE_MIRROR)"

release: prepare-release ##- Run the dataset release tool, e.g. make release ARGS="preflight paleocar_v3"
	@$(COMPOSE) --profile ingest build release
	PRODUCER_REVISION="$$(git rev-parse HEAD)$$(test -z "$$(git status --porcelain)" || echo -dirty)" \
		$(COMPOSE) --profile ingest run --rm release $(ARGS)

# The mirror is mounted writable only here; preflight and build read it read-only.
release-fetch: prepare-release ##- Copy declared sources into the source mirror, e.g. make release-fetch ARGS="paleocar_v3"
	@$(COMPOSE) --profile ingest build release
	$(COMPOSE) --profile ingest run --rm -v "$(abspath $(RELEASE_SOURCE_MIRROR))":/mirror \
		-e SKOPE_RELEASE_SOURCE_MIRROR=/mirror release fetch $(ARGS)

# The ledgers and pin files are mounted writable only here, and the command runs
# as the calling user so the files it writes stay theirs.
release-pin: prepare-release ##- Show the edits that pin a release, e.g. make release-pin MANIFEST=path/to/release-manifest.json PIN_ENV=staging; add WRITE=1 to apply them
	@test -n "$(MANIFEST)" || { echo "set MANIFEST to a release's release-manifest.json (or a copy of it)" 1>&2; exit 2; }
	@test -n "$(PIN_ENV)" || { echo "set PIN_ENV to the environment to pin it in (dev, staging, or prod)" 1>&2; exit 2; }
	@$(COMPOSE) --profile ingest build release
	$(COMPOSE) --profile ingest run --rm \
		-v "$(abspath timeseries/ingest/datasets)":/pin/datasets \
		-v "$(abspath deploy/releases)":/pin/releases \
		-v "$(abspath $(MANIFEST))":/pin/release-manifest.json:ro \
		release --datasets /pin/datasets pin /pin/release-manifest.json \
		--env $(PIN_ENV) --pins /pin/releases $(if $(WRITE),--write)

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
	$(TEST_COMPOSE) build $(BUILD_FLAGS) server titiler redis
	@trap '$(TEST_COMPOSE) down --remove-orphans' EXIT INT TERM; \
		$(TEST_COMPOSE) run --rm -v "$(CURDIR)/tests/contracts:/code/tests/contracts:ro" server sh -c \
		'black --check --extend-exclude "app/vendor/" app && pytest -o cache_dir=/tmp/pytest-cache -c app/pytest.ini app/tests'

test-ingest: override ENVIRONMENT=dev
test-ingest: prepare ##- Build and run the ingest tests
	@$(TEST_COMPOSE) --profile test config --quiet
	$(TEST_COMPOSE) --profile test build $(BUILD_FLAGS) ingest-test
	@trap '$(TEST_COMPOSE) --profile test down --remove-orphans' EXIT INT TERM; \
		$(TEST_COMPOSE) --profile test run --rm --no-deps ingest-test

test-runtime: prepare ##- Build fresh runtime images and exercise real tiles, extraction and analysis
	python3 -m unittest discover -s tests -p test_runtime_images.py
	python3 scripts/runtime-images.py test

scan-runtime-images: ##- Scan the tested runtime images for fixable high/critical vulnerabilities
	python3 scripts/runtime-images.py scan

export-runtime-images: ##- Archive tested runtime images for deployment and rollback
	python3 scripts/runtime-images.py export

maintenance-candidate: ##- Rebuild patched runtime images without cached layers, test, scan and archive a maintenance candidate
	$(MAKE) test BUILD_FLAGS=--pull
	$(MAKE) test-runtime RUNTIME_REFRESH=1
	$(MAKE) scan-runtime-images
	$(MAKE) export-runtime-images
