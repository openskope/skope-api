# skope-api
Backend services for dataset metadata and timeseries data for SKOPE

## Project Setup (Development)

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
http --json POST localhost:8002/datasets/monthly_5x5x60_dataset_float32_variable < req.json
```
