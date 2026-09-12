import numpy as np
import pytest
import pandas as pd

from app.exceptions import SelectedAreaPolygonIsTooLarge
from app.core.timeseries_processing import (
    apply_temporal_transform,
    apply_zscore_transform,
    calculate_safe_chunk_size,
    execute_analyze_request,
    generate_band_chunks,
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
# calculate_safe_chunk_size


def test_chunk_size_normal():
    result = calculate_safe_chunk_size(width=100, height=50, max_cells=500_000)
    assert result == 100  # 500_000 // 5_000


def test_chunk_size_zero_width_raises():
    with pytest.raises(ValueError, match="0-pixel"):
        calculate_safe_chunk_size(width=0, height=50, max_cells=500_000)


def test_chunk_size_zero_height_raises():
    with pytest.raises(ValueError, match="0-pixel"):
        calculate_safe_chunk_size(width=100, height=0, max_cells=500_000)


def test_chunk_size_area_exceeds_max_cells_raises():
    # 1000*1000 = 1_000_000 > 500_000 → n_bands_per_chunk = 0
    with pytest.raises(SelectedAreaPolygonIsTooLarge):
        calculate_safe_chunk_size(width=1000, height=1000, max_cells=500_000)


def test_chunk_size_exact_fit():
    # 100*100 = 10_000, max_cells = 10_000 → result = 1
    result = calculate_safe_chunk_size(width=100, height=100, max_cells=10_000)
    assert result == 1


# ---------------------------------------------------------------------------
# generate_band_chunks


def test_generate_band_chunks_exact_division():
    chunks = list(generate_band_chunks([1, 2, 3, 4, 5, 6], chunk_size=3))
    assert chunks == [[1, 2, 3], [4, 5, 6]]


def test_generate_band_chunks_partial_last_chunk():
    chunks = list(generate_band_chunks([1, 2, 3, 4, 5], chunk_size=3))
    assert chunks == [[1, 2, 3], [4, 5]]


def test_generate_band_chunks_single_element():
    chunks = list(generate_band_chunks([7], chunk_size=3))
    assert chunks == [[7]]


def test_generate_band_chunks_empty():
    chunks = list(generate_band_chunks([], chunk_size=3))
    assert chunks == []


def test_generate_band_chunks_chunk_size_larger_than_list():
    chunks = list(generate_band_chunks([1, 2], chunk_size=10))
    assert chunks == [[1, 2]]


# ---------------------------------------------------------------------------
# apply_zscore_transform


def test_zscore_no_transform_passthrough(base_series):
    result = apply_zscore_transform(base_series, NoTransform())
    pd.testing.assert_series_equal(result, base_series)


def test_zscore_none_passthrough(base_series):
    result = apply_zscore_transform(base_series, None)
    pd.testing.assert_series_equal(result, base_series)


def test_zscore_moving_interval_leading_nan(base_series):
    result = apply_zscore_transform(base_series, ZScoreMovingInterval(width=3))
    assert pd.isna(result.iloc[0])
    assert pd.isna(result.iloc[1])
    assert pd.isna(result.iloc[2])
    # idx 3 compares 4 against the prior values [1,2,3], using population std.
    assert np.isclose(result.iloc[3], (4.0 - 2.0) / np.std([1.0, 2.0, 3.0]))


def test_zscore_fixed_interval_full_series(base_series):
    result = apply_zscore_transform(base_series, ZScoreFixedInterval(time_range=None))
    expected = (base_series - base_series.mean()) / base_series.std(ddof=0)
    np.testing.assert_allclose(result.values, expected.values)


def test_zscore_fixed_interval_with_time_range(base_series):
    # ref = values at 0102, 0103, 0104 = [3, 4, 5]
    transform = ZScoreFixedInterval(time_range=TimeRange(gte="0102", lte="0104"))
    result = apply_zscore_transform(base_series, transform)
    assert len(result) == len(base_series)
    ref_mean = 4.0
    ref_std = np.std([3.0, 4.0, 5.0])
    np.testing.assert_allclose(result.iloc[0], (1.0 - ref_mean) / ref_std)
    np.testing.assert_allclose(result.iloc[2], (3.0 - ref_mean) / ref_std)


def test_zscore_fixed_interval_non_overlapping_range_raises(base_series):
    transform = ZScoreFixedInterval(time_range=TimeRange(gte="0200", lte="0300"))
    with pytest.raises(ValueError):
        apply_zscore_transform(base_series, transform)


def test_zscore_fixed_interval_std_zero_returns_zero_series(constant_series):
    result = apply_zscore_transform(
        constant_series, ZScoreFixedInterval(time_range=None)
    )
    assert (result == 0.0).all()
    assert list(result.index) == list(constant_series.index)


# ---------------------------------------------------------------------------
# apply_temporal_transform


def test_temporal_transform_none_passthrough(base_series):
    result = apply_temporal_transform(base_series, None)
    pd.testing.assert_series_equal(result, base_series)


def test_temporal_transform_no_smoother_passthrough(base_series):
    result = apply_temporal_transform(base_series, NoSmoother())
    pd.testing.assert_series_equal(result, base_series)


def test_temporal_transform_trailing_moving_average(base_series):
    smoother = MovingAverageSmoother(method="trailing", width=3)
    result = apply_temporal_transform(base_series, smoother)
    # min_periods=1 means: idx 0 = mean([1]) = 1.0, idx 2 = mean([1,2,3]) = 2.0
    assert np.isclose(result.iloc[0], 1.0)
    assert np.isclose(result.iloc[1], 1.5)
    assert np.isclose(result.iloc[2], 2.0)


def test_temporal_transform_centered_moving_average(base_series):
    smoother = MovingAverageSmoother(method="centered", width=3)
    result = apply_temporal_transform(base_series, smoother)
    # center=True, min_periods=1: idx 0 sees [1,2] → 1.5; idx 1 sees [1,2,3] → 2.0
    assert np.isclose(result.iloc[0], 1.5)
    assert np.isclose(result.iloc[1], 2.0)
    assert np.isclose(result.iloc[4], 5.0)  # [4,5,6] → 5.0


def test_temporal_transform_width_1_is_identity(base_series):
    smoother = MovingAverageSmoother(method="trailing", width=1)
    result = apply_temporal_transform(base_series, smoother)
    np.testing.assert_allclose(result.values, base_series.values)


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


def test_execute_analyze_request_time_range_slice():
    payload = _make_request(time_range=TimeRange(gte="0103", lte="0106"))
    base = _make_base_payload()
    result = execute_analyze_request(payload, base, _make_extraction_metadata())
    assert len(result.series[0].values) == 4


def test_execute_analyze_request_empty_time_range_raises():
    payload = _make_request(time_range=TimeRange(gte="0200", lte="0300"))
    base = _make_base_payload()
    with pytest.raises(ValueError):
        execute_analyze_request(payload, base, _make_extraction_metadata())


def test_execute_analyze_request_zscore_and_smoother_combined():
    payload = _make_request(
        transform=ZScoreFixedInterval(time_range=None),
        requested_series_options=[
            SeriesOptions(
                name="smoothed",
                smoother=MovingAverageSmoother(method="trailing", width=3),
            )
        ],
    )
    base = _make_base_payload()
    result = execute_analyze_request(payload, base, _make_extraction_metadata())
    assert result is not None
    assert len(result.series) == 1


def test_execute_analyze_request_all_nan_summary_stats():
    # ZScoreMovingInterval(width=11) on a 10-element series → all NaN (window > series length)
    payload = _make_request(transform=ZScoreMovingInterval(width=11))
    base = _make_base_payload()
    result = execute_analyze_request(payload, base, _make_extraction_metadata())
    assert result.summary_stats[0].mean is None
    assert result.summary_stats[0].median is None
    assert result.summary_stats[0].stdev is None
