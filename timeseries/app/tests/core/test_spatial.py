import pytest
from pyproj import Geod
from shapely.geometry import Point, box

from app.core.spatial import (
    calculate_spatial_coverage,
    estimate_cell_count,
    resolve_spatial_window,
)

LARGE_GRID = (2000, 2000)  # rows, columns: holds every area below

# ---------------------------------------------------------------------------
# estimate_cell_count


def test_estimate_cell_count_geographic():
    # 1° × 1° box, 0.00833° pixels, EPSG:4326
    shapes = [box(-110.0, 37.0, -109.0, 38.0)]
    transform = [0.00833, 0.0, -115.0, 0.0, -0.00833, 43.0]
    result = estimate_cell_count(shapes, transform, "EPSG:4326", LARGE_GRID)
    # The bounds span columns and rows 600.24 .. 720.29: cells 600 .. 720.
    assert result == 121 * 121


def test_estimate_cell_count_projected_uses_exact_crs_transform():
    shapes = [box(-110.0, 37.0, -109.0, 38.0)]
    transform = [800.0, 0.0, 200000.0, 0.0, -800.0, 4800000.0]
    result = estimate_cell_count(shapes, transform, "EPSG:32612", LARGE_GRID)

    # The precise UTM footprint differs from the former square
    # meters-per-degree approximation.
    assert result == 114 * 141


def test_estimate_cell_count_zero_area():
    shapes = [box(-110.0, 37.0, -110.0, 37.0)]
    transform = [0.00833, 0.0, -115.0, 0.0, -0.00833, 43.0]
    result = estimate_cell_count(shapes, transform, "EPSG:4326", LARGE_GRID)
    assert result == 1


# ---------------------------------------------------------------------------
# Areas partly outside the grid

# A 10 x 10 grid of 1-degree cells spanning x 0 .. 10 and y 0 .. 10.
GRID = [1.0, 0.0, 0.0, 0.0, -1.0, 10.0]
GRID_SHAPE = (10, 10)


def test_window_keeps_the_cells_on_the_grid():
    # x 5 .. 15 and y -3 .. 4: the grid holds columns 5 .. 9 and rows 6 .. 9.
    area = box(5.0, -3.0, 15.0, 4.0)

    _, _, window = resolve_spatial_window([area], GRID, "EPSG:4326", GRID_SHAPE)

    assert (window.col_off, window.row_off) == (5, 6)
    assert (window.width, window.height) == (5, 4)
    assert estimate_cell_count([area], GRID, "EPSG:4326", GRID_SHAPE) == 20


@pytest.mark.parametrize("x, y, col, row", [(10.0, 5.5, 9, 4), (5.5, 0.0, 5, 9)])
def test_a_point_on_the_outer_edge_reads_the_edge_cell(x, y, col, row):
    _, _, window = resolve_spatial_window([Point(x, y)], GRID, "EPSG:4326", GRID_SHAPE)

    assert (window.col_off, window.row_off, window.width, window.height) == (
        col,
        row,
        1,
        1,
    )


def test_coverage_counts_only_the_overlap():
    # x 5 .. 15 and y 2 .. 4: cell centers inside are columns 5 .. 9, rows 6 .. 7.
    area = box(5.0, 2.0, 15.0, 4.0)
    shapes, transform, window = resolve_spatial_window(
        [area], GRID, "EPSG:4326", GRID_SHAPE
    )

    mask, n_cells, total_area = calculate_spatial_coverage(
        shapes, transform, window, "EPSG:4326"
    )

    assert n_cells == 10 and mask.sum() == 10
    overlap, _ = Geod(ellps="WGS84").geometry_area_perimeter(box(5.0, 2.0, 10.0, 4.0))
    assert total_area == pytest.approx(abs(overlap), rel=1e-9)
