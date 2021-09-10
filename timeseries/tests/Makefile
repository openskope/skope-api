all: install test

node_modules:
	npm ci

.PHONY:
install:
	npm ci

.PHONY:
test: | node_modules
	npm test

