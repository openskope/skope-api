# Backend math validation

Checks the API's time series against an independent computation from the source GeoTIFF.

## Files

| File | What it does |
| --- | --- |
| `validate.py` | Extracts each area through a running API, then compares the extraction and every `/analyze` case with the reference. Its docstring states the definitions it checks. |
| `areas.geojson` | The study areas: a point, boxes with edges a quarter cell from any center, a triangle, a box whose bounds start 0.6 into a cell, two disjoint boxes, a sliver between cell centers, and a box on the grid's east edge. |

The reference reads the source file, which holds every timestep in one file with its year in each band description,
not the release COGs, so it also checks the release's time-to-band layout. It shares no code with the API.

## Running it

- A running API serving the dataset (`make deploy-dev` serves the dev pin).
- A local copy of each variable's source GeoTIFF, the URI that `timeseries/ingest/datasets/<dataset>/source-manifest.yml`
  lists for it (`make release-fetch` copies them into the source mirror).

Run it inside the API image, which has every dependency, with the sources mounted read-only:

```bash
docker run --rm \
  -v "$PWD/experiments/backend-math-validation:/validation:ro" \
  -v /path/to/sources:/sources:ro -v /path/to/results:/results \
  -w /validation openskope/skope-api:dev \
  python validate.py --api http://host.docker.internal:8001 --dataset paleocar_v3 --areas areas.geojson \
    --source ppt_annual=/sources/ppt_annual/prediction_scaled.tif \
    --source gdd_cotton_annual=/sources/gdd_cotton_annual/prediction_scaled.tif \
    --out /results/validation.json
```

It prints one line per extraction and a total, writes every mismatch to `--out`, and exits non-zero on any mismatch.
Values must agree within a relative tolerance of 1e-5 (absolute 1e-6); nulls must be in the same places.
