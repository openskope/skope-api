import pytest
from shapely.geometry import Point, box

from app.core.spatial import resolve_spatial_window
from app.core.validation import validate_dataset_and_variable, validate_window_size

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
# validate_window_size

GRID = [0.00833, 0.0, -115.0, 0.0, -0.00833, 43.0]
GRID_SHAPE = (1440, 1560)  # rows, columns: to -102.0 and 31.0


def _check(shapes, max_cells):
    selection = resolve_spatial_window(shapes, GRID, "EPSG:4326", GRID_SHAPE)
    validate_window_size(selection, max_cells)


def test_validate_window_size_within_limit(small_polygon_shape):
    _check([small_polygon_shape], max_cells=500_000)


def test_validate_window_size_exceeds_limit(large_polygon_shape):
    # Only the part on the grid counts: x -115 .. -110 reaches 601 columns,
    # and the grid's 1440 rows.
    with pytest.raises(ValueError, match="read window spans 865440 cells"):
        _check([large_polygon_shape], max_cells=500_000)


def test_validate_window_size_counts_the_window_between_shapes():
    # Two small polygons far apart on the grid: the window between them spans
    # many cells.
    shapes = [box(-114.1, 37.9, -114.0, 38.0), box(-103.1, 31.9, -103.0, 32.0)]
    with pytest.raises(ValueError, match="too large"):
        _check(shapes, max_cells=10)


def test_validate_window_size_ignores_a_shape_off_the_grid():
    # The far shape is off the grid, so it reads nothing: one small window.
    _check(
        [box(-114.1, 37.9, -114.0, 38.0), box(-80.1, 29.9, -80.0, 30.0)],
        max_cells=500,
    )


def test_validate_window_size_accepts_a_point():
    _check([Point(-110.0, 38.0)], max_cells=1)


def test_validate_window_size_limits_points_far_apart():
    with pytest.raises(ValueError, match="too large"):
        _check([Point(-114.5, 42.5), Point(-102.5, 31.5)], max_cells=10)
