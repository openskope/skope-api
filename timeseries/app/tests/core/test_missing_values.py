"""Timesteps without a value: they come back as null, statistics skip them, and
no NaN reaches a response.

PaleoCAR v3 has no data for 0417-0589, so every series that crosses those
years has such timesteps.
"""

import json
import warnings

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin

from app.core.timeseries_processing import (
    apply_temporal_transform,
    execute_analyze_request,
    execute_timeseries_job,
    summarize_series,
)
from app.schemas.timeseries import (
    MovingAverageSmoother,
    NoSmoother,
    NoTransform,
    SeriesOptions,
    TimeseriesAnalyzeRequest,
    TimeseriesRequest,
    ZonalStatistic,
    ZScoreFixedInterval,
)

NODATA = 4294967295
TIMESTEPS = ["0001", "0002", "0003", "0004", "0005"]

# ---------------------------------------------------------------------------
# Summary statistics


def test_single_value_has_zero_stdev():
    stat = summarize_series("raw", pd.Series([4.0], index=["0001"]))

    assert stat.mean == 4.0
    assert stat.median == 4.0
    assert stat.stdev == 0.0


def test_statistics_skip_missing_values():
    series = pd.Series([1.0, np.nan, 3.0, np.nan], index=TIMESTEPS[:4])

    stat = summarize_series("raw", series)

    assert stat.mean == 2.0
    assert stat.median == 2.0
    assert stat.stdev == pytest.approx(np.std([1.0, 3.0], ddof=0))


def test_statistics_of_no_values_are_null():
    stat = summarize_series("raw", pd.Series([np.nan, np.nan], index=TIMESTEPS[:2]))

    assert (stat.mean, stat.median, stat.stdev) == (None, None, None)


# ---------------------------------------------------------------------------
# Smoothing


@pytest.mark.parametrize(
    "method, null_positions",
    [
        # Trailing: the first 2 lack a full window, and the gap at 4 reaches
        # the 2 timesteps after it.
        ("trailing", [0, 1, 4, 5, 6]),
        # Centered: 1 at each end, and the gap reaches 1 on either side.
        ("centered", [0, 3, 4, 5, 8]),
    ],
)
def test_smoothing_needs_a_full_window(method, null_positions):
    values = [1.0, 2.0, 3.0, 4.0, np.nan, 6.0, 7.0, 8.0, 9.0]
    series = pd.Series(values, index=[f"{year:04d}" for year in range(1, 10)])

    smoothed = apply_temporal_transform(
        series, MovingAverageSmoother(method=method, width=3)
    )

    assert list(np.flatnonzero(smoothed.isna())) == null_positions


# ---------------------------------------------------------------------------
# Analyze (no raster reads)


def _analyze(base, **overrides):
    payload = {
        "extraction_id": "job-001",
        "transform": NoTransform(),
        "requested_series_options": [SeriesOptions(name="raw", smoother=NoSmoother())],
        "zonal_statistic": ZonalStatistic.mean,
        **overrides,
    }
    metadata = {"dataset_id": "ds", "variable_id": "v", "area": 1.0, "n_cells": 1}
    return execute_analyze_request(TimeseriesAnalyzeRequest(**payload), base, metadata)


def _base(values):
    return {"timesteps": TIMESTEPS[: len(values)], "mean": values, "median": values}


def test_analyze_keeps_missing_timesteps_null():
    result = _analyze(_base([1.0, None, None, 4.0, 5.0]))

    assert result.series[0].values == [1.0, None, None, 4.0, 5.0]
    assert result.summary_stats[0].mean == pytest.approx(10.0 / 3)


def test_analyze_zscore_skips_missing_timesteps():
    result = _analyze(
        _base([1.0, None, 3.0, None, 5.0]),
        transform=ZScoreFixedInterval(time_range=None),
    )

    values = result.series[0].values
    assert values[1] is None and values[3] is None
    # Mean 3 and population stdev sqrt(8/3) of the three values that exist.
    expected = (np.array([1.0, 3.0, 5.0]) - 3.0) / np.sqrt(8 / 3)
    assert [values[0], values[2], values[4]] == pytest.approx(list(expected))


def test_analyze_series_with_no_values_returns_nulls():
    result = _analyze(_base([None, None]))

    assert result.series[0].values == [None, None]
    assert result.summary_stats[0].mean is None


def test_analyze_response_is_strict_json():
    result = _analyze(_base([2.0]))

    json.dumps(result.model_dump(), allow_nan=False)


# ---------------------------------------------------------------------------
# Extraction from a raster with an empty band


@pytest.fixture
def raster_with_empty_band(tmp_path):
    """Three 4x4 UInt32 bands at 1 degree per pixel; band 2 holds only nodata."""
    path = tmp_path / "cube.tif"
    bands = np.stack(
        [
            np.full((4, 4), 10, dtype=np.uint32),
            np.full((4, 4), NODATA, dtype=np.uint32),
            np.full((4, 4), 30, dtype=np.uint32),
        ]
    )
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=4,
        height=4,
        count=3,
        dtype="uint32",
        nodata=NODATA,
        crs="EPSG:4326",
        transform=from_origin(-110.0, 40.0, 1.0, 1.0),
    ) as dst:
        dst.write(bands)
    return str(path)


def _extract_request():
    return TimeseriesRequest.model_validate(
        {
            "dataset_id": "ds",
            "variable_id": "v",
            "selected_area": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [-109.0, 37.0],
                        [-108.0, 37.0],
                        [-108.0, 38.0],
                        [-109.0, 38.0],
                        [-109.0, 37.0],
                    ]
                ],
            },
            "zonal_statistic": "mean",
            "transform": {"type": "NoTransform"},
            "requested_series_options": [
                {"name": "raw", "smoother": {"type": "NoSmoother"}}
            ],
            "time_range": None,
        }
    )


async def _extract(path, timesteps):
    return await execute_timeseries_job(
        request=_extract_request(),
        file_mapping={path: [1, 2, 3]},
        timestep_list=timesteps,
        dataset_crs="EPSG:4326",
        dataset_transform_array=[1.0, 0.0, -110.0, 0.0, -1.0, 40.0],
    )


async def test_extraction_returns_an_empty_band_as_null(raster_with_empty_band):
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        response, base = await _extract(raster_with_empty_band, TIMESTEPS[:3])

    assert response.series[0].values == [10.0, None, 30.0]
    assert response.series[0].timesteps == TIMESTEPS[:3]
    assert base["mean"] == [10.0, None, 30.0]
    assert base["median"] == [10.0, None, 30.0]
    assert response.summary_stats[0].mean == 20.0
    json.dumps(response.model_dump(), allow_nan=False)
    json.dumps(base, allow_nan=False)


async def test_extraction_fails_when_values_and_timesteps_differ(
    raster_with_empty_band,
):
    with pytest.raises(RuntimeError, match="3 values for 4 timesteps"):
        await _extract(raster_with_empty_band, TIMESTEPS[:4])
