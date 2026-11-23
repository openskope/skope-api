import pytest
from pyproj import Geod
from shapely.geometry import Point, box

from app.core.spatial import calculate_spatial_coverage, resolve_spatial_window
from app.exceptions import SelectedAreaOutOfBoundsError
from app.schemas.geometry import (
    SkopeFeatureModel,
    SkopePointModel,
    SkopePolygonModel,
)

LARGE_GRID = (2000, 2000)  # rows, columns: holds every area below
FINE = [0.00833, 0.0, -115.0, 0.0, -0.00833, 43.0]

# ---------------------------------------------------------------------------
# The read window's size


def test_window_cells_geographic():
    # 1° × 1° box, 0.00833° pixels, EPSG:4326
    selection = resolve_spatial_window(
        [box(-110.0, 37.0, -109.0, 38.0)], FINE, "EPSG:4326", LARGE_GRID
    )
    # The bounds span columns and rows 600.24 .. 720.29: cells 600 .. 720.
    assert selection.window_cells == 121 * 121


def test_window_cells_projected_uses_exact_crs_transform():
    transform = [800.0, 0.0, 200000.0, 0.0, -800.0, 4800000.0]
    selection = resolve_spatial_window(
        [box(-110.0, 37.0, -109.0, 38.0)], transform, "EPSG:32612", LARGE_GRID
    )

    # The precise UTM footprint differs from the former square
    # meters-per-degree approximation.
    assert selection.window_cells == 114 * 141


def test_a_sub_pixel_area_reads_one_cell():
    # Inside the cell at column 600, row 720.
    selection = resolve_spatial_window(
        [box(-110.0, 36.996, -109.999, 36.997)], FINE, "EPSG:4326", LARGE_GRID
    )
    assert selection.window_cells == 1


# ---------------------------------------------------------------------------
# Areas on the grid's edges and beyond them: window, mask, cells and area

# A 10 x 10 grid of 1-degree cells spanning x 0 .. 10 and y 0 .. 10.
GRID = [1.0, 0.0, 0.0, 0.0, -1.0, 10.0]
GRID_SHAPE = (10, 10)


def _select(shapes):
    """The window, mask, cell count and area extraction uses for `shapes`."""
    selection = resolve_spatial_window(shapes, GRID, "EPSG:4326", GRID_SHAPE)
    mask, n_cells, area = calculate_spatial_coverage(selection, "EPSG:4326")
    return selection.window, mask, n_cells, area


def _geodesic_area(shape):
    area, _ = Geod(ellps="WGS84").geometry_area_perimeter(shape)
    return abs(area)


def _offsets(window):
    return (window.col_off, window.row_off, window.width, window.height)


def test_window_keeps_the_cells_on_the_grid():
    # x 5 .. 15 and y -3 .. 4: the grid holds columns 5 .. 9 and rows 6 .. 9.
    selection = resolve_spatial_window(
        [box(5.0, -3.0, 15.0, 4.0)], GRID, "EPSG:4326", GRID_SHAPE
    )

    assert _offsets(selection.window) == (5, 6, 5, 4)
    assert selection.window_cells == 20


def test_coverage_counts_only_the_overlap():
    # x 5 .. 15 and y 2 .. 4: cell centers inside are columns 5 .. 9, rows 6 .. 7.
    _, mask, n_cells, area = _select([box(5.0, 2.0, 15.0, 4.0)])

    assert n_cells == 10 and mask.sum() == 10
    assert area == pytest.approx(_geodesic_area(box(5.0, 2.0, 10.0, 4.0)), rel=1e-9)


@pytest.mark.parametrize(
    "x, y, col, row",
    [
        (0.0, 5.5, 0, 4),  # west
        (10.0, 5.5, 9, 4),  # east
        (5.5, 10.0, 5, 0),  # north
        (5.5, 0.0, 5, 9),  # south
        (10.0, 0.0, 9, 9),  # south-east corner
    ],
)
def test_a_point_on_the_outer_edge_selects_the_edge_cell(x, y, col, row):
    window, mask, n_cells, area = _select([Point(x, y)])

    assert _offsets(window) == (col, row, 1, 1)
    assert n_cells == 1 and mask.all()
    assert area == 0


def test_a_shape_off_the_grid_does_not_widen_the_window():
    # A one-cell box in the north-west corner, and one far off the grid.
    on_grid = box(0.2, 9.2, 0.8, 9.8)
    off_grid = box(30.0, -30.0, 40.0, -20.0)

    window, mask, n_cells, area = _select([on_grid, off_grid])

    assert _offsets(window) == (0, 0, 1, 1)
    assert n_cells == 1 and mask.all()
    assert area == pytest.approx(_geodesic_area(on_grid), rel=1e-9)


def test_a_polygon_touching_only_the_edge_is_dropped():
    # The second box shares only the grid's east edge: it has no cells.
    inside = box(0.2, 9.2, 0.8, 9.8)
    touching = box(10.0, 0.0, 11.0, 10.0)

    window, _, n_cells, _ = _select([inside, touching])

    assert _offsets(window) == (0, 0, 1, 1)
    assert n_cells == 1


def test_a_point_off_the_grid_is_dropped_beside_an_area_on_it():
    window, _, n_cells, _ = _select([Point(30.0, 30.0), box(0.2, 9.2, 0.8, 9.8)])

    assert _offsets(window) == (0, 0, 1, 1)
    assert n_cells == 1


def test_points_far_apart_read_every_cell_between_them():
    selection = resolve_spatial_window(
        [Point(0.5, 9.5), Point(9.5, 0.5)], GRID, "EPSG:4326", GRID_SHAPE
    )
    mask, n_cells, _ = calculate_spatial_coverage(selection, "EPSG:4326")

    assert selection.window_cells == 100
    assert n_cells == 2 and mask[0, 0] and mask[9, 9]


@pytest.mark.parametrize(
    "shapes, message",
    [
        ([box(30.0, 30.0, 31.0, 31.0)], "selected area"),
        ([box(10.0, 0.0, 11.0, 10.0)], "selected area"),  # touches the edge only
        ([Point(30.0, 30.0)], "selected point"),
        ([Point(10.0, 10.5)], "selected point"),
    ],
    ids=["area-outside", "area-on-edge", "point-outside", "point-past-edge"],
)
def test_an_area_with_nothing_on_the_grid_raises(shapes, message):
    with pytest.raises(SelectedAreaOutOfBoundsError, match=message):
        resolve_spatial_window(shapes, GRID, "EPSG:4326", GRID_SHAPE)


@pytest.mark.parametrize(
    "geometry",
    [
        {"type": "Point", "coordinates": [10.0, 5.5]},
        {
            "type": "Polygon",
            "coordinates": [[[8.0, 2.0], [12.0, 2.0], [12.0, 4.0], [8.0, 2.0]]],
        },
    ],
    ids=["point-on-edge", "polygon-across-edge"],
)
def test_a_shape_selects_the_same_bare_or_in_a_feature(geometry):
    model = SkopePointModel if geometry["type"] == "Point" else SkopePolygonModel
    bare = model(**geometry)
    wrapped = SkopeFeatureModel(type="Feature", geometry=geometry, properties={})

    window, mask, n_cells, area = _select(bare.shapes)
    wrapped_window, wrapped_mask, wrapped_cells, wrapped_area = _select(wrapped.shapes)

    assert _offsets(wrapped_window) == _offsets(window)
    assert (wrapped_mask == mask).all()
    assert (wrapped_cells, wrapped_area) == (n_cells, area)
    assert n_cells > 0
