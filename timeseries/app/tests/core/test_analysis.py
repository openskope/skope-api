import numpy as np
import pytest
import pandas as pd

from app.core.analysis import (
    apply_temporal_transform,
    apply_zscore_transform,
    execute_analyze_request,
)
from app.schemas.timeseries import (
    MovingAverageSmoother,
    NoSmoother,
    NoTransform,
    SeriesOptions,
    TimeRange,
    TimeseriesAnalyzeRequest,
    ZonalStatistic,
    ZScoreFixedInterval,
    ZScoreMovingInterval,
)

# ---------------------------------------------------------------------------
# apply_zscore_transform


def test_zscore_no_transform_passthrough(base_series):
    result = apply_zscore_transform(base_series, NoTransform())
    pd.testing.assert_series_equal(result, base_series)


def test_zscore_none_passthrough(base_series):
    result = apply_zscore_transform(base_series, None)
    pd.testing.assert_series_equal(result, base_series)


def test_zscore_fixed_interval_full_series(base_series):
    result = apply_zscore_transform(base_series, ZScoreFixedInterval(time_range=None))
    expected = (base_series - base_series.mean()) / base_series.std(ddof=0)
    np.testing.assert_allclose(result.values, expected.values)


def test_zscore_fixed_interval_non_overlapping_range_raises(base_series):
    transform = ZScoreFixedInterval(time_range=TimeRange(gte="0200", lte="0300"))
    with pytest.raises(ValueError):
        apply_zscore_transform(base_series, transform)


# ---------------------------------------------------------------------------
# apply_temporal_transform


def test_temporal_transform_none_passthrough(base_series):
    result = apply_temporal_transform(base_series, None)
    pd.testing.assert_series_equal(result, base_series)


def test_temporal_transform_no_smoother_passthrough(base_series):
    result = apply_temporal_transform(base_series, NoSmoother())
    pd.testing.assert_series_equal(result, base_series)


# ---------------------------------------------------------------------------
# execute_analyze_request


def _make_base_payload(timesteps=None):
    if timesteps is None:
        timesteps = [
            "0100",
            "0101",
            "0102",
            "0103",
            "0104",
            "0105",
            "0106",
            "0107",
            "0108",
            "0109",
        ]
    mean_vals = [float(i + 1) for i in range(len(timesteps))]
    median_vals = [float(i) * 0.5 for i in range(len(timesteps))]
    return {"timesteps": timesteps, "mean": mean_vals, "median": median_vals}


def _make_extraction_metadata():
    return {"dataset_id": "test-ds", "variable_id": "ppt", "area": 1000.0, "n_cells": 5}


def _make_request(**overrides):
    defaults = {
        "extraction_id": "job-001",
        "transform": NoTransform(),
        "requested_series_options": [SeriesOptions(name="raw", smoother=NoSmoother())],
        "time_range": None,
        "zonal_statistic": ZonalStatistic.mean,
    }
    defaults.update(overrides)
    return TimeseriesAnalyzeRequest(**defaults)


def test_execute_analyze_request_mean_no_transform():
    payload = _make_request()
    base = _make_base_payload()
    result = execute_analyze_request(payload, base, _make_extraction_metadata())
    assert result.dataset_id == "test-ds"
    assert result.series[0].values == base["mean"]


def test_execute_analyze_request_median_selected():
    payload = _make_request(zonal_statistic=ZonalStatistic.median)
    base = _make_base_payload()
    result = execute_analyze_request(payload, base, _make_extraction_metadata())
    assert result.series[0].values == base["median"]


def test_execute_analyze_request_empty_time_range_raises():
    payload = _make_request(time_range=TimeRange(gte="0200", lte="0300"))
    base = _make_base_payload()
    with pytest.raises(ValueError):
        execute_analyze_request(payload, base, _make_extraction_metadata())


def test_execute_analyze_request_smooths_before_slicing():
    payload = _make_request(
        time_range=TimeRange(gte="0103", lte="0105"),
        requested_series_options=[
            SeriesOptions(
                name="smoothed",
                smoother=MovingAverageSmoother(method="trailing", width=3),
            )
        ],
    )
    result = execute_analyze_request(
        payload, _make_base_payload(), _make_extraction_metadata()
    )
    # Means at 0101-0105 are 2-6; the window at 0103 reaches 0101 and 0102.
    assert result.series[0].values == [3.0, 4.0, 5.0]


def test_execute_analyze_request_value_does_not_depend_on_the_range():
    options = [
        SeriesOptions(
            name="smoothed",
            smoother=MovingAverageSmoother(method="centered", width=3),
        )
    ]
    transform = ZScoreMovingInterval(width=2)
    base = _make_base_payload()
    whole = execute_analyze_request(
        _make_request(transform=transform, requested_series_options=options),
        base,
        _make_extraction_metadata(),
    )
    narrow = execute_analyze_request(
        _make_request(
            transform=transform,
            requested_series_options=options,
            time_range=TimeRange(gte="0104", lte="0106"),
        ),
        base,
        _make_extraction_metadata(),
    )
    by_year = dict(zip(whole.series[0].timesteps, whole.series[0].values))
    assert narrow.series[0].values[0] is not None
    assert narrow.series[0].values == [by_year[t] for t in narrow.series[0].timesteps]


def test_execute_analyze_request_fixed_interval_defaults_to_the_requested_range():
    payload = _make_request(
        transform=ZScoreFixedInterval(time_range=None),
        time_range=TimeRange(gte="0103", lte="0106"),
    )
    result = execute_analyze_request(
        payload, _make_base_payload(), _make_extraction_metadata()
    )
    # Means 4-7: reference mean 5.5, population stdev sqrt(1.25).
    expected = (np.array([4.0, 5.0, 6.0, 7.0]) - 5.5) / np.sqrt(1.25)
    assert result.series[0].values == pytest.approx(list(expected))
