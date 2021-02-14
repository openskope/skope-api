include config.mk

UID=$(shell id -u)
GID=$(shell id -g)

.PHONY: build
build: docker-compose.yml
	docker-compose build --pull

docker-compose.yml: deploy/dc/base.yml deploy/dc/$(ENVIR).yml deploy/Dockerfile
	case "$(ENVIR)" in \
	  dev|prod) docker-compose -f deploy/dc/base.yml -f "deploy/dc/$(ENVIR).yml" --project-directory . config > docker-compose.yml;; \
	  *) echo "invalid environment. must be dev or prod" 1>&2; exit 1;; \
	esac

.PHONY: test
test:
	docker-compose run --user "$(UID):$(GID)" --rm -v $(PWD):/code server pytest