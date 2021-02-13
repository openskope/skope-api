# skope-api
Backend services for dataset metadata and timeseries data for SKOPE

## Project Setup (Development)

Install the dependencies

```bash
conda env create -n timeseries-api -f environment.yml
```

Run the server

```bash
uvicorn main:app --reload
```

Try out the analysis endpoint

```bash
http --json POST localhost:8000/datasets/monthly_5x5x60_dataset_float32_variable < req.json
```
