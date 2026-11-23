`paleocar_v3-r-2026.11.04/` holds copies of the `overview.yml` and
`release-manifest.json` of a real paleocar_v3 release, the one staging serves,
so the tests see a real dataset's variables, time axis and COGs next to the
committed display files. The COGs are not here.

The release format itself is checked through `tests/contracts/release-documents/`,
which the release tool's tests keep current; this copy does not need replacing
when the format changes.
