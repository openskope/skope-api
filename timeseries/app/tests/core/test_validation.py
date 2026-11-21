import pytest
from shapely.geometry import box

from app.core.validation import validate_dataset_and_variable, validate_geom_size

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
# validate_geom_size

GRID = [0.00833, 0.0, -115.0, 0.0, -0.00833, 43.0]
GRID_SHAPE = (1440, 1560)  # rows, columns: to -102.0 and 31.0


def test_validate_geom_size_within_limit(small_polygon_shape):
    validate_geom_size(
        [small_polygon_shape], GRID, "EPSG:4326", GRID_SHAPE, max_cells=500_000
    )


def test_validate_geom_size_exceeds_limit(large_polygon_shape):
    with pytest.raises(ValueError, match="too large"):
        validate_geom_size(
            [large_polygon_shape], GRID, "EPSG:4326", GRID_SHAPE, max_cells=500_000
        )


def test_validate_geom_size_multiple_shapes_uses_unary_union():
    # Two small polygons far apart: their union's bounds span many cells, up to
    # the grid's east edge
    shape1 = box(-114.1, 37.9, -114.0, 38.0)
    shape2 = box(-80.1, 29.9, -80.0, 30.0)
    with pytest.raises(ValueError):
        validate_geom_size(
            [shape1, shape2], GRID, "EPSG:4326", GRID_SHAPE, max_cells=10
        )
