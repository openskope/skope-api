import pytest
from shapely.geometry import box

from app.core.validation import (
    estimate_cell_count,
    validate_dataset_and_variable,
    validate_geom_size,
)

# ---------------------------------------------------------------------------
# validate_dataset_and_variable


def test_validate_dataset_and_variable_returns_the_release(served_registry):
    release = validate_dataset_and_variable(served_registry, "annual", "ppt")

    assert release.dataset_id == "annual"


def test_validate_dataset_and_variable_unknown_dataset(served_registry):
    with pytest.raises(ValueError, match="does-not-exist"):
        validate_dataset_and_variable(served_registry, "does-not-exist", "ppt")


def test_validate_dataset_and_variable_unknown_variable(served_registry):
    with pytest.raises(ValueError, match="no-such-var"):
        validate_dataset_and_variable(served_registry, "annual", "no-such-var")


# ---------------------------------------------------------------------------
# estimate_cell_count


def test_estimate_cell_count_geographic():
    # 1° × 1° box, 0.00833° pixels, EPSG:4326
    shapes = [box(-110.0, 37.0, -109.0, 38.0)]
    transform = [0.00833, 0.0, -115.0, 0.0, -0.00833, 43.0]
    result = estimate_cell_count(shapes, transform, "EPSG:4326")
    # The bounds span columns and rows 600.24 .. 720.29: cells 600 .. 720.
    assert result == 121 * 121


def test_estimate_cell_count_projected_uses_exact_crs_transform():
    shapes = [box(-110.0, 37.0, -109.0, 38.0)]
    transform = [800.0, 0.0, 200000.0, 0.0, -800.0, 4800000.0]
    result = estimate_cell_count(shapes, transform, "EPSG:32612")

    # The precise UTM footprint differs from the former square
    # meters-per-degree approximation.
    assert result == 114 * 141


def test_estimate_cell_count_zero_area():
    shapes = [box(0.0, 0.0, 0.0, 0.0)]
    transform = [0.00833, 0.0, -115.0, 0.0, -0.00833, 43.0]
    result = estimate_cell_count(shapes, transform, "EPSG:4326")
    assert result == 1


# ---------------------------------------------------------------------------
# validate_geom_size

GRID = [0.00833, 0.0, -115.0, 0.0, -0.00833, 43.0]


def test_validate_geom_size_within_limit(small_polygon_shape):
    validate_geom_size([small_polygon_shape], GRID, "EPSG:4326", max_cells=500_000)


def test_validate_geom_size_exceeds_limit(large_polygon_shape):
    with pytest.raises(ValueError, match="too large"):
        validate_geom_size([large_polygon_shape], GRID, "EPSG:4326", max_cells=500_000)


def test_validate_geom_size_multiple_shapes_uses_unary_union():
    # Two small polygons placed far apart: unary_union bbox spans a large area
    shape1 = box(-114.1, 37.9, -114.0, 38.0)
    shape2 = box(-80.1, 29.9, -80.0, 30.0)
    with pytest.raises(ValueError):
        validate_geom_size([shape1, shape2], GRID, "EPSG:4326", max_cells=10)
