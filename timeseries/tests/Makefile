all: install test

node_modules:
	npm install

.PHONY:
install:
	npm install

.PHONY:
test: | node_modules
	npm test

