# skope-api

[![DOI](https://zenodo.org/badge/338436138.svg)](https://zenodo.org/badge/latestdoi/338436138)
[![skope-api build](https://github.com/openskope/skope-api/actions/workflows/test.yml/badge.svg)](https://github.com/openskope/skope-api/actions/workflows/test.yml)

Backend services for dataset metadata and timeseries data extracted from SKOPE datasets - see https://api.openskope.org/docs for API details and examples

## Project Setup

### Dataset Metadata

Dataset metadata currently needs to be specified twice and should :

- `timeseries/metadata.yml` contains the full dataset metadata exposed by the [metadata endpoint](https://api.openskope.org/docs#/metadata/metadata_metadata_get) and consumed by the [skopeui](https://github.com/openskope/skopeui) app
- `timeseries/deploy/metadata/prod.yml` contains a subset of the dataset metadata used by the backend services to handle timeseries extraction (time ranges, time resolution, and available variables) from the datacubes available in the store (runtime settings for the store dynamically generated at build time at `timeseries/deploy/settings/config.yml`)

### Development

Set up `dev` configuration

```bash
./configure dev
```

Build the project & run the backend server and geoserver

```bash
make deploy
```

Try out the analysis endpoint

```bash
http --json POST localhost:8002/datasets/yearly < yearly.json 
http --json POST localhost:8002/datasets/monthly < monthly.json 
```

Run the tests

```bash
make test
```

## Production

Set up for `prod` deployment

```bash
./configure prod
```

build & deploy

```bash
make deploy
```
