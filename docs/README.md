# Documentation

The guides live beside the code they describe:

| To… | Read |
| --- | --- |
| Get an overview, run the API locally, or learn what a dataset release is | [Root README](../README.md) |
| Add or update a dataset: declare it, build a release, and publish it | [Ingest README](../timeseries/ingest/README.md) |
| Choose what an environment serves, how it is shown, and deploy, verify, or roll back | [App README](../timeseries/app/README.md) |

This folder holds the reference material behind them:

| Path | Holds |
| --- | --- |
| [`specs/dataset-release-v1.md`](specs/dataset-release-v1.md) | The rules for building and publishing a release; requirement IDs such as OBS-011 |
| [`specs/release-consumption-v1.md`](specs/release-consumption-v1.md) | The rules for serving releases: pins, display files, `/metadata`; requirement IDs such as PIN-004 |
| [`experiments/`](experiments/) | Experiment reports that decisions rest on, such as the PaleoCAR v3 COG encoding |
| [`data-preparation.md`](data-preparation.md) | The legacy dataset pipeline, proposed for deletion |

Decisions and their reasons are in [`.agents/decisions/`](../.agents/decisions/).
