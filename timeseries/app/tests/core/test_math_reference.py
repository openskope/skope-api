"""The time-series math against direct computations of its definitions.

Each expected value is computed without the API's code: cells by testing cell
centers with shapely, zonal statistics by brute-force NumPy in float64, and
transforms and smoothers by plain loops over the definitions that
`experiments/backend-math-validation/validate.py` checks against the source
data. The API must agree with them to rounding.
"""

import math

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point, Polygon, box

from app.core.timeseries_processing import (
    apply_temporal_transform,
    apply_zscore_transform,
    calculate_spatial_coverage,
    extract_summarystat_timeseries,
)
from app.core.validation import resolve_spatial_window
from app.schemas.timeseries import (
    MovingAverageSmoother,
    TimeRange,
    ZScoreFixedInterval,
    ZScoreMovingInterval,
)

# A 1-degree grid whose top-left corner is (0, 10), so cell (row, col) spans
# x in [col, col + 1] and y in [10 - row - 1, 10 - row].
GRID = [1.0, 0.0, 0.0, 0.0, -1.0, 10.0]
NODATA = 4294967295

# ---------------------------------------------------------------------------
# Cells


def _expected_cells(area, window):
    """Cells of the window whose center is inside the area; touched ones if none."""
    rows, cols = int(window.height), int(window.width)
    centers = np.zeros((rows, cols), dtype=bool)
    touched = np.zeros((rows, cols), dtype=bool)
    for r in range(rows):
        for c in range(cols):
            x, y = window.col_off + c, 10 - (window.row_off + r)
            centers[r, c] = area.contains(Point(x + 0.5, y - 0.5))
            touched[r, c] = area.intersects(box(x, y - 1, x + 1, y))
    return centers if centers.any() else touched


def _cells(area):
    shapes, transform, window = resolve_spatial_window([area], GRID, "EPSG:4326")
    mask, n_cells, _ = calculate_spatial_coverage(
        shapes, transform, window, "EPSG:4326"
    )
    return window, mask, n_cells


def test_window_reaches_every_cell_the_bounds_touch():
    # Bounds start 0.6 into a cell on every side; cell centers 1.5 .. 5.5 are
    # inside, so 5 x 5 cells, in columns and rows 1 .. 5.
    area = box(0.6, 4.4, 5.6, 9.4)

    window, mask, n_cells = _cells(area)

    assert (window.col_off, window.row_off) == (0, 0)
    assert (window.width, window.height) == (6, 6)
    assert n_cells == 25
    assert (mask == _expected_cells(area, window)).all()


@pytest.mark.parametrize(
    "area",
    [
        Polygon([(0.3, 0.2), (7.9, 1.1), (3.6, 8.7)]),
        Polygon([(1.2, 1.2), (6.8, 1.4), (6.6, 6.9), (4.1, 3.3), (1.4, 6.7)]),
        box(2.25, 3.25, 6.75, 7.75),
    ],
    ids=["triangle", "concave", "quarter-cell-box"],
)
def test_cells_are_those_whose_center_is_inside(area):
    window, mask, n_cells = _cells(area)
    expected = _expected_cells(area, window)

    assert (mask == expected).all()
    assert n_cells == expected.sum()


def test_an_area_without_a_center_counts_the_cells_it_touches():
    # Inside column 4, between centers, across rows 2 .. 4.
    area = box(4.1, 5.2, 4.3, 7.8)

    window, mask, n_cells = _cells(area)

    assert n_cells == 3
    assert (mask == _expected_cells(area, window)).all()


# ---------------------------------------------------------------------------
# Zonal statistics


def _write_cube(path, bands):
    count, height, width = bands.shape
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=count,
        dtype="uint32",
        nodata=NODATA,
        crs="EPSG:4326",
        transform=from_origin(0.0, 10.0, 1.0, 1.0),
    ) as dst:
        dst.write(bands)
    return str(path)


def _expected_statistics(bands, mask):
    data = bands.astype(np.float64)
    data[bands == NODATA] = np.nan
    means, medians = [], []
    for band in data:
        values = band[mask]
        values = values[~np.isnan(values)]
        means.append(values.mean() if values.size else np.nan)
        medians.append(np.median(values) if values.size else np.nan)
    return np.array(means), np.array(medians)


@pytest.mark.parametrize("chunk_size", [1, 2, 5])
def test_zonal_statistics_per_timestep_in_order(tmp_path, chunk_size):
    rng = np.random.default_rng(7)
    bands = rng.integers(0, 60000, size=(5, 10, 10), dtype=np.uint32)
    bands[1, 2:5, 3:6] = NODATA  # some cells without a value
    bands[3] = NODATA  # a timestep without any value
    path = _write_cube(tmp_path / "cube.tif", bands)
    area = Polygon([(0.3, 0.2), (7.9, 1.1), (3.6, 8.7)])
    window, mask, _ = _cells(area)
    full_mask = np.zeros((10, 10), dtype=bool)
    r0, c0 = int(window.row_off), int(window.col_off)
    full_mask[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]] = mask

    _, means, medians = extract_summarystat_timeseries(
        path, [1, 2, 3, 4, 5], window, mask, chunk_size
    )

    expected_means, expected_medians = _expected_statistics(bands, full_mask)
    np.testing.assert_allclose(means, expected_means, rtol=1e-12)
    np.testing.assert_array_equal(medians, expected_medians)


def test_zonal_mean_keeps_float64_precision(tmp_path):
    # Float32 sums over this many cells drift by about 1e-7 of the mean.
    rng = np.random.default_rng(11)
    bands = rng.integers(0, 60000, size=(1, 300, 300), dtype=np.uint32)
    path = _write_cube(tmp_path / "large.tif", bands)
    mask = np.ones((300, 300), dtype=bool)
    window = rasterio.windows.Window(0, 0, 300, 300)

    _, means, _ = extract_summarystat_timeseries(path, [1], window, mask, 1)

    assert means[0] == pytest.approx(bands.astype(np.float64).mean(), rel=1e-12)


# ---------------------------------------------------------------------------
# Transforms and smoothers


def _series_with_gaps():
    rng = np.random.default_rng(3)
    values = rng.normal(500.0, 40.0, size=40)
    values[12:19] = np.nan  # a run of empty timesteps, like 0417-0589
    values[27] = np.nan
    return pd.Series(values, index=[f"{i:04d}" for i in range(1, 41)])


def _as_list(series):
    return [None if np.isnan(v) else float(v) for v in series]


def _population_std(values):
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))


def _expected_moving_zscore(values, width):
    expected = []
    for t, value in enumerate(values):
        window = values[t - width : t] if t >= width else None
        if value is None or window is None or None in window:
            expected.append(None)
            continue
        std = _population_std(window)
        expected.append(None if std == 0 else (value - sum(window) / width) / std)
    return expected


def _expected_average(values, before, after):
    expected = []
    for t in range(len(values)):
        window = values[t - before : t + after + 1] if t >= before else []
        complete = len(window) == before + after + 1 and None not in window
        expected.append(sum(window) / len(window) if complete else None)
    return expected


def _assert_same(actual, expected):
    assert [v is None for v in actual] == [v is None for v in expected]
    for a, e in zip(actual, expected):
        if e is not None:
            assert a == pytest.approx(e, rel=1e-12)


@pytest.mark.parametrize("width", [2, 3, 10])
def test_moving_zscore_matches_its_definition(width):
    series = _series_with_gaps()

    actual = apply_zscore_transform(series, ZScoreMovingInterval(width=width))

    _assert_same(_as_list(actual), _expected_moving_zscore(_as_list(series), width))


def test_fixed_zscore_matches_its_definition():
    series = _series_with_gaps()
    transform = ZScoreFixedInterval(time_range=TimeRange(gte="0005", lte="0030"))

    actual = apply_zscore_transform(series, transform)

    reference = [v for v in _as_list(series)[4:30] if v is not None]
    mean, std = sum(reference) / len(reference), _population_std(reference)
    expected = [None if v is None else (v - mean) / std for v in _as_list(series)]
    _assert_same(_as_list(actual), expected)


def test_fixed_zscore_of_a_constant_reference_is_null():
    series = pd.Series([2.0, np.nan, 2.0, 2.0], index=["0001", "0002", "0003", "0004"])

    actual = apply_zscore_transform(series, ZScoreFixedInterval(time_range=None))

    assert actual.isna().all()


def test_fixed_zscore_needs_two_reference_values():
    series = _series_with_gaps()
    transform = ZScoreFixedInterval(time_range=TimeRange(gte="0013", lte="0019"))

    with pytest.raises(ValueError, match="needs at least 2"):
        apply_zscore_transform(series, transform)


@pytest.mark.parametrize(
    "method, width, before, after",
    [
        ("trailing", 1, 0, 0),
        ("trailing", 5, 4, 0),
        ("centered", 3, 1, 1),
        ("centered", 7, 3, 3),
    ],
)
def test_moving_average_matches_its_definition(method, width, before, after):
    series = _series_with_gaps()

    actual = apply_temporal_transform(
        series, MovingAverageSmoother(method=method, width=width)
    )

    _assert_same(_as_list(actual), _expected_average(_as_list(series), before, after))
