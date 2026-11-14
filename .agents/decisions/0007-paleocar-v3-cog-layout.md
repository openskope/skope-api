# 0007: Encode PaleoCAR v3 COGs In 128-Pixel Tile-Interleaved Chunks Of 250 Timesteps

- Status: Proposed
- Date: 2026-11-03
- Specifications: [SKOPE Dataset Release Specification v1](../../docs/specs/dataset-release-v1.md), EXP-002 to EXP-004
- Evidence: [PaleoCAR v3 COG encoding experiment](../../docs/experiments/paleocar-v3-encoding.md)

## Context

Each PaleoCAR v3 variable has 1,898 annual timesteps on a 1560 × 1440 grid,
split into COG files of a fixed number of timesteps. The specification leaves
that number, the block size, the interleave and the data type to EXP-003. They
affect both read paths: a map tile reads one band, and an extraction reads every
timestep in its range over the study area. The experiment measured candidate
layouts on both, and selected with a rule set before the deciding run: rank by
full-range extraction time summed over a point and three polygon sizes, and
take the leader unless its map views framed on those areas are more than 0.75 s
per slider step slower than the fastest candidate's.

Extraction ranks first because it is the slower operation and holds server
capacity: each of the API's 4 workers runs one extraction at a time (ADR 0005).
Releases are served from a local volume on an 8-vCPU, 16 GB host, and the
experiment ran in a simulation of that host.

## Decision

PaleoCAR v3 temporal variables will be written as COGs with:

- `BLOCKSIZE=128`;
- `INTERLEAVE=TILE`;
- 250 timesteps per file;
- `UInt16` with nodata 65535 for each variable whose preflight scan proves every
  valid value representable (COG-014), and the source-preserving `UInt32` in the
  same layout otherwise.

All other COG creation options stay as the specification sets them.

## Consequences

- Full-range extraction takes about half as long as with 512-pixel blocks
  (6.0 s against 13.0 s summed over the four areas), reads half the bytes on a
  large area, and returns identical results.
- Map views framed on a study area stay within 0.08 s per step of the fastest
  layout. The whole-region view at the initial zoom is slower: 1.6 s per step
  against 1.1 s with 512-pixel blocks.
- A 12-variable release is 96 COGs and 60.9 GB in `UInt16` as built
  (`paleocar_v3-r-2026.11.04`). The experiment's estimate, 45 GB, multiplied
  `ppt_annual` by twelve; the growing degree day variables compress less well
  (5.3–5.9 GB each against 3.1–3.8 GB). `UInt32` is about 13% larger and 9%
  slower to extract.
- Peak memory for a large-area extraction is about 1.3 GB, so 4 concurrent ones
  fit on the host beside the other services.
- An extraction of a few hundred years is up to 3.5 times slower than with
  25-timestep files, because it falls within one file and gets no parallel
  reads. skopeui's first extraction always covers the full range, but changing
  the range on the visualize page starts a new extraction. That page should
  instead cut the range from the stored extraction through
  `/timeseries/analyze`.
- The release models and the source-manifest schema allow a block size of 128.
- The layout is fixed per release; changing it later needs a new release.
- The evidence covers local storage only. Serving releases from object storage
  needs a new experiment, because request latency, header size and the number
  of files weigh differently there. Categorical variables and static rasters
  need their own evidence before adopting this layout.
