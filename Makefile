include config.mk

UID=$(shell id -u)
GID=$(shell id -g)

.PHONY: help
# Instructions for using this Makefile
help:
	@cat $(MAKEFILE_LIST) | docker run --rm -i xanders/make-help

.PHONY: build
# Build and pull the required docker images
build: docker-compose.yml geoserver/docker/secrets/geoserver_admin_password
	docker-compose build --pull

geoserver/docker/secrets/geoserver_admin_password:
	echo "Creating geoserver secret"; \
	mkdir -p geoserver/docker/secrets; \
	echo -n $$(head /dev/urandom | tr -dc '[:alnum:]' | head -c22) > geoserver/docker/secrets/geoserver_admin_password

docker-compose.yml: deploy/dc/base.yml deploy/dc/$(ENVIR).yml timeseries/deploy/Dockerfile config.mk
	case "$(ENVIR)" in \
	  dev|prod) docker-compose -f deploy/dc/base.yml -f "deploy/dc/$(ENVIR).yml" --project-directory . config > docker-compose.yml;; \
	  *) echo "invalid environment. must be dev or prod" 1>&2; exit 1;; \
	esac

.PHONY: deploy
# Deploy the web app after `build`
deploy: build
	docker-compose up -d

##
## Testing
##

.PHONY: test-unit
# Run python unit tests
test-unit:
	docker-compose run --user "$(UID):$(GID)" --rm -v $(PWD):/code server pytest
