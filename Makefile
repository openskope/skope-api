include config.mk

UID=$(shell id -u)
GID=$(shell id -g)

GEOSERVER_ADMIN_PASSWORD_PATH=geoserver/docker/secrets/geoserver_admin_password
DOCKER_SHARE_MOUNT=docker/shared

.PHONY: help
help: 	##- Instructions for using this Makefile. Run ./configure {dev|staging|prod} first
	@echo "usage: make [target] ..."
	@echo "targets:"
	@sed -e '/#\{2\}-/!d; s/\\$$//; s/:[^#\t]*/:/; s/#\{2\}- *//' $(MAKEFILE_LIST)

.PHONY: build
build: docker-compose.yml | $(GEOSERVER_ADMIN_PASSWORD_PATH) 	##- Build and pull the required docker images 
	docker compose build --pull

$(GEOSERVER_ADMIN_PASSWORD_PATH):
	echo "Creating geoserver secret"; \
	mkdir -p geoserver/docker/secrets; \
	echo -n $$(openssl rand -base64 22) > geoserver/docker/secrets/geoserver_admin_password

docker-compose.yml: deploy/base.yml deploy/$(ENVIRONMENT).yml timeseries/deploy/Dockerfile config.mk
	case "$(ENVIRONMENT)" in \
	  dev|prod) docker compose -f deploy/base.yml -f "deploy/$(ENVIRONMENT).yml" --project-directory . config > docker-compose.yml;; \
	  *) echo "invalid environment. must be dev or prod" 1>&2; exit 1;; \
	esac

.PHONY: deploy
deploy: build 	##- build and deploy the web app 
	mkdir -p $(DOCKER_SHARE_MOUNT)
	docker compose up -d

##
## Testing
##

.PHONY: test

test: 	##- Run python unit tests
	docker compose run --rm server pytest
