# Release identity contract

`release-identity.json` defines the agreement required between a release manifest
and its generated overview. Both the API startup verifier and the ingest release
verifier apply these cases to synthetic releases. Tests update file checksums
after each edit, so a refusal demonstrates an identity check rather than merely
detecting changed bytes.

The API Make target mounts this directory read-only at `tests/contracts`;
the ingest test image copies it there. Production images do not include it.
