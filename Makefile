include config.mk

.PHONY: build
build: docker-compose.yml
	docker-compose build --pull

docker-compose.yml: deploy/base.yml deploy/$(ENVIR).yml deploy/Dockerfile
	case "$(ENVIR)" in \
	  dev) docker-compose -f deploy/base.yml -f "deploy/$(ENVIR).yml" --project-directory . config > docker-compose.yml;; \
	  *) echo "invalid environment. must be dev" 1>&2; exit 1;; \
	esac

.PHONY: test
test:
	docker-compose run --rm -v $(PWD):/code server pytest