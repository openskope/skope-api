"""Check the API's time series against an independent computation from the source GeoTIFF.

The reference reads the source file (one file, every timestep, years in the band
descriptions), not the release, and shares no code with the API: the cells come
from point-in-polygon tests on cell centers, the statistics from NumPy, and the
transforms and smoothers from plain loops over the definitions below.

Definitions, for a series x over the dataset's axis and a requested range R:

- Cells: those whose center lies inside the area; when none does (a point, a
  sliver), those the area touches. Nodata cells are skipped.
- Zonal mean and median: over the area's cells with a value, per timestep;
  null when no cell has one.
- Z-score, fixed interval with reference period P (R when not given):
  (x_t - mean(x over P)) / std(x over P), population std, skipping nulls;
  null everywhere when P has no values or that std is 0.
- Z-score, moving window w: x_t against x_{t-w} .. x_{t-1}, population std;
  null when x_t or any of those is null, when t has fewer than w predecessors,
  or when their std is 0.
- Trailing average w: mean of x_{t-w+1} .. x_t; centered average w (odd):
  x_{t-(w-1)/2} .. x_{t+(w-1)/2}. Null when any of those is null or falls
  outside the axis.
- Transform and smoother apply to the whole axis; the response is then cut to R.
- Summary statistics: mean, median and population std of the cut's values.

Usage (inside the API image, which has every dependency):

    python validate.py --api http://host.docker.internal:8001 \
        --dataset paleocar_v3 --areas areas.geojson \
        --source ppt_annual=/sources/ppt_annual.tif \
        --source gdd_cotton_annual=/sources/gdd_cotton_annual.tif \
        --out /results/validation.json
"""

import argparse
import json
import math
import sys
import time
import urllib.error
import urllib.request

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.windows import Window
from shapely.geometry import Point, box, shape
from shapely.ops import transform as reproject
from shapely.ops import unary_union

RELATIVE_TOLERANCE = 1e-5
ABSOLUTE_TOLERANCE = 1e-6

TRANSFORMS = {
    "none": {"type": "NoTransform"},
    "zscore_selected": {"type": "ZScoreFixedInterval"},
    "zscore_fixed_1000_1100": {
        "type": "ZScoreFixedInterval",
        "time_range": {"gte": "1000", "lte": "1100"},
    },
    "zscore_fixed_in_empty_years": {
        "type": "ZScoreFixedInterval",
        "time_range": {"gte": "0450", "lte": "0500"},
    },
    "zscore_moving_10": {"type": "ZScoreMovingInterval", "width": 10},
}
SMOOTHERS = {
    "none": {"type": "NoSmoother"},
    "trailing_5": {"type": "MovingAverageSmoother", "method": "trailing", "width": 5},
    "centered_7": {"type": "MovingAverageSmoother", "method": "centered", "width": 7},
}
# Full axis; across the empty years 0417-0589 and several COG files; a late window.
RANGES = [None, ("0350", "0700"), ("1500", "1600")]


# ---------------------------------------------------------------------------
# HTTP


def _post(api: str, path: str, body: dict) -> tuple[int, dict]:
    request = urllib.request.Request(
        f"{api}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read() or b"{}")


def _get(api: str, path: str) -> dict:
    with urllib.request.urlopen(f"{api}{path}", timeout=120) as response:
        return json.loads(response.read())


def extract(api: str, dataset: str, variable: str, geometry: dict) -> dict:
    """Extract the whole axis with no transform and return the finished job."""
    body = {
        "dataset_id": dataset,
        "variable_id": variable,
        "selected_area": geometry,
        "zonal_statistic": "mean",
        "transform": {"type": "NoTransform"},
        "requested_series_options": [
            {"name": "raw", "smoother": {"type": "NoSmoother"}}
        ],
        "time_range": None,
    }
    for _ in range(60):
        status, job = _post(api, "/timeseries/extract", body)
        if status != 503:
            break
        time.sleep(5)
    if status != 202:
        raise RuntimeError(f"extract {variable}: {status} {job}")
    job_id = job["job_id"]
    while True:
        job = _get(api, f"/timeseries/status/{job_id}")
        if job["status"] in ("SUCCESS", "FAILED", "TIMED_OUT"):
            break
        time.sleep(1)
    if job["status"] != "SUCCESS":
        raise RuntimeError(f"extract {variable}: {job}")
    job["job_id"] = job_id
    return job


# ---------------------------------------------------------------------------
# Reference: cells and zonal statistics from the source file


def to_shapely(geometry: dict):
    """A GeoJSON geometry, Feature or FeatureCollection as one shape."""
    if geometry["type"] == "FeatureCollection":
        return unary_union([shape(f["geometry"]) for f in geometry["features"]])
    if geometry["type"] == "Feature":
        return shape(geometry["geometry"])
    return shape(geometry)


def area_cells(src, geometry: dict) -> tuple[Window, np.ndarray]:
    """The read window and mask of the area's cells, from cell-center tests."""
    to_grid = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
    area = reproject(to_grid.transform, to_shapely(geometry))
    minx, miny, maxx, maxy = area.bounds
    inverse = ~src.transform
    cols = [inverse * (x, y) for x, y in [(minx, maxy), (maxx, miny)]]
    # Two cells of margin on every side, so the window never decides membership.
    col0 = max(0, math.floor(min(c[0] for c in cols)) - 2)
    row0 = max(0, math.floor(min(c[1] for c in cols)) - 2)
    col1 = min(src.width, math.ceil(max(c[0] for c in cols)) + 2)
    row1 = min(src.height, math.ceil(max(c[1] for c in cols)) + 2)
    window = Window(col0, row0, col1 - col0, row1 - row0)

    def cell_center(row, col):
        return Point(src.transform * (col0 + col + 0.5, row0 + row + 0.5))

    def cell_box(row, col):
        x0, y0 = src.transform * (col0 + col, row0 + row)
        x1, y1 = src.transform * (col0 + col + 1, row0 + row + 1)
        return box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))

    shape_rows, shape_cols = int(window.height), int(window.width)
    inside = np.zeros((shape_rows, shape_cols), dtype=bool)
    for row in range(shape_rows):
        for col in range(shape_cols):
            inside[row, col] = area.contains(cell_center(row, col))
    if not inside.any():
        for row in range(shape_rows):
            for col in range(shape_cols):
                inside[row, col] = area.intersects(cell_box(row, col))
    return window, inside


def reference_base_series(path: str, geometry: dict) -> dict:
    with rasterio.open(path) as src:
        window, inside = area_cells(src, geometry)
        years = list(src.descriptions)
        data = src.read(window=window).astype(np.float64)
        if src.nodata is not None:
            data[data == src.nodata] = np.nan
    cells = data[:, inside]
    means, medians = [], []
    for values in cells:
        valid = values[~np.isnan(values)]
        means.append(float(valid.mean()) if valid.size else None)
        medians.append(float(np.median(valid)) if valid.size else None)
    return {
        "timesteps": years,
        "mean": means,
        "median": medians,
        "n_cells": int(inside.sum()),
    }


# ---------------------------------------------------------------------------
# Reference: transforms, smoothers and statistics, from the definitions


def _mean(values):
    return sum(values) / len(values)


def _population_std(values):
    m = _mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / len(values))


def _reference_period(years: list, transform: dict, requested) -> tuple:
    period = transform.get("time_range")
    if period:
        return period["gte"], period["lte"]
    return requested or (years[0], years[-1])


def zscore(series: list, years: list, transform: dict, requested) -> list:
    kind = transform["type"]
    if kind == "NoTransform":
        return list(series)
    if kind == "ZScoreFixedInterval":
        lo, hi = _reference_period(years, transform, requested)
        reference = [
            v for y, v in zip(years, series) if lo <= y <= hi and v is not None
        ]
        if not reference or _population_std(reference) == 0:
            return [None] * len(series)
        std = _population_std(reference)
        m = _mean(reference)
        return [None if v is None else (v - m) / std for v in series]
    width = transform["width"]
    result = []
    for t, value in enumerate(series):
        window = series[t - width : t] if t >= width else None
        if value is None or window is None or any(v is None for v in window):
            result.append(None)
            continue
        std = _population_std(window)
        result.append(None if std == 0 else (value - _mean(window)) / std)
    return result


def smooth(series: list, smoother: dict) -> list:
    if smoother["type"] == "NoSmoother":
        return list(series)
    width = smoother["width"]
    if smoother["method"] == "trailing":
        before, after = width - 1, 0
    else:
        before = after = (width - 1) // 2
    result = []
    for t in range(len(series)):
        window = series[t - before : t + after + 1] if t >= before else []
        complete = len(window) == before + after + 1 and None not in window
        result.append(_mean(window) if complete else None)
    return result


def summary(values: list) -> dict:
    present = [v for v in values if v is not None]
    if not present:
        return {"mean": None, "median": None, "stdev": None}
    m = _mean(present)
    stdev = math.sqrt(sum((v - m) ** 2 for v in present) / len(present))
    return {"mean": m, "median": float(np.median(present)), "stdev": stdev}


# ---------------------------------------------------------------------------
# Comparison


def _close(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return math.isclose(a, b, rel_tol=RELATIVE_TOLERANCE, abs_tol=ABSOLUTE_TOLERANCE)


def compare_values(expected: list, actual: list) -> dict:
    mismatches = [
        i for i, (e, a) in enumerate(zip(expected, actual)) if not _close(e, a)
    ]
    differences = [
        abs(e - a) / max(abs(e), 1e-12)
        for e, a in zip(expected, actual)
        if e is not None and a is not None
    ]
    return {
        "length_matches": len(expected) == len(actual),
        "mismatches": len(mismatches) + abs(len(expected) - len(actual)),
        "first_mismatch": mismatches[0] if mismatches else None,
        "max_relative_difference": max(differences, default=0.0),
    }


def check_case(
    api,
    job,
    reference,
    variable,
    area,
    statistic,
    transform_name,
    smoother_name,
    requested,
):
    years = reference["timesteps"]
    body = {
        "extraction_id": job["job_id"],
        "zonal_statistic": statistic,
        "transform": TRANSFORMS[transform_name],
        "requested_series_options": [
            {"name": "series", "smoother": SMOOTHERS[smoother_name]}
        ],
        "time_range": {"gte": requested[0], "lte": requested[1]} if requested else None,
    }
    status, response = _post(api, "/timeseries/analyze", body)
    case = {
        "variable": variable,
        "area": area,
        "statistic": statistic,
        "transform": transform_name,
        "smoother": smoother_name,
        "range": list(requested) if requested else "all",
        "status": status,
    }
    if status != 200:
        case.update(ok=False, error=response)
        return case

    full = smooth(
        zscore(reference[statistic], years, TRANSFORMS[transform_name], requested),
        SMOOTHERS[smoother_name],
    )
    lo, hi = requested or (years[0], years[-1])
    keep = [i for i, y in enumerate(years) if lo <= y <= hi]
    expected = [full[i] for i in keep]
    expected_years = [years[i] for i in keep]

    series = response["series"][0]
    values = compare_values(expected, series["values"])
    stats = response["summary_stats"][0]
    expected_stats = summary(expected)
    stats_ok = all(
        _close(expected_stats[k], stats[k]) for k in ("mean", "median", "stdev")
    )
    case.update(
        values=values,
        timesteps_match=series["timesteps"] == expected_years,
        time_range_matches=series["time_range"]
        == {"gte": expected_years[0], "lte": expected_years[-1]},
        stats_match=stats_ok,
        n_cells_match=response["n_cells"] == reference["n_cells"],
    )
    if not stats_ok:
        case["stats"] = {"expected": expected_stats, "actual": stats}
    case["ok"] = (
        values["mismatches"] == 0
        and case["timesteps_match"]
        and case["time_range_matches"]
        and stats_ok
        and case["n_cells_match"]
    )
    return case


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--api", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--areas", required=True)
    parser.add_argument(
        "--source", action="append", required=True, help="VARIABLE=PATH"
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with open(args.areas) as f:
        areas = json.load(f)["features"]
    sources = dict(item.split("=", 1) for item in args.source)

    cases, extractions = [], []
    for variable, path in sources.items():
        for feature in areas:
            name = feature["properties"]["name"]
            reference = reference_base_series(path, feature["geometry"])
            job = extract(args.api, args.dataset, variable, feature["geometry"])
            raw = job["result"]["series"][0]
            extraction = {
                "variable": variable,
                "area": name,
                "n_cells": {
                    "expected": reference["n_cells"],
                    "actual": job["result"]["n_cells"],
                },
                "timesteps_match": raw["timesteps"] == reference["timesteps"],
                "values": compare_values(reference["mean"], raw["values"]),
            }
            extraction["ok"] = (
                extraction["timesteps_match"]
                and extraction["values"]["mismatches"] == 0
                and reference["n_cells"] == job["result"]["n_cells"]
            )
            extractions.append(extraction)
            print(
                f"{variable} {name}: extraction {'ok' if extraction['ok'] else 'MISMATCH'}",
                flush=True,
            )
            for statistic in ("mean", "median"):
                for transform_name in TRANSFORMS:
                    for smoother_name in SMOOTHERS:
                        for requested in RANGES:
                            cases.append(
                                check_case(
                                    args.api,
                                    job,
                                    reference,
                                    variable,
                                    name,
                                    statistic,
                                    transform_name,
                                    smoother_name,
                                    requested,
                                )
                            )

    failed = [c for c in cases if not c["ok"]]
    report = {
        "extractions": extractions,
        "cases": len(cases),
        "failed_cases": failed,
    }
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    bad_extractions = [e for e in extractions if not e["ok"]]
    print(
        f"{len(extractions) - len(bad_extractions)}/{len(extractions)} extractions match; "
        f"{len(cases) - len(failed)}/{len(cases)} analyze cases match."
    )
    return 0 if not failed and not bad_extractions else 1


if __name__ == "__main__":
    sys.exit(main())
