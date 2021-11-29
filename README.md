# skope-api
Backend services for dataset metadata and timeseries data for SKOPE

## Project Setup

### Dataset Metadata

Dataset metadata currently needs to be specified twice, in `timeseries/metadata.yml` and
`timeseries/deploy/metadata/prod.yml`. 

### Development

Put into `dev` configuration

```bash
./configure dev
```

Build the project

```bash
make build
```

Run the server

```bash
docker-compose up -d
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

Put into `prod` configuration

```bash
./configure prod
```

Build the project

```bash
make build
```

Run the server

```bash
docker-compose up -d
```
