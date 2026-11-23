# Shared test contracts

Files both test suites read, so the release tool and the API cannot drift apart.

## Release identity

`release-identity.json` defines the agreement required between a release manifest
and its generated overview. Both the API startup verifier and the ingest release
verifier apply these cases to synthetic releases. Tests update file checksums
after each edit, so a refusal demonstrates an identity check rather than merely
detecting changed bytes.

## Release documents

`release-documents/` holds the `overview.yml` and `release-manifest.json` the
release tool writes for a small synthetic release. The ingest tests build that
release and fail when its documents differ from these; the API tests parse
these with the API's models. A change to the release format therefore fails the
ingest tests until the files are rewritten, and the rewritten files then reach
the API's tests.

Values that depend on GDAL's bytes or on the test's paths (file sizes,
checksums, declaration digests, source paths) are replaced by placeholders.
After a deliberate format change, rewrite the files and review the diff:

```bash
make test-ingest UPDATE_CONTRACTS=1
```

## Where they are mounted

The API Make target mounts this directory read-only at `tests/contracts`. The
ingest test image copies it there, and `make test-ingest` mounts it over the
copy: read-only, or writable with `UPDATE_CONTRACTS=1`. Production images do not
include it.
