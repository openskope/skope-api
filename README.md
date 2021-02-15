# skope-api
Backend services for dataset metadata and timeseries data for SKOPE

## Project Setup

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
http --json POST localhost:8002/datasets/yearly < req.json 
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
