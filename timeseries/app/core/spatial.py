"""Map a requested area onto a dataset's grid: its read window and its cells."""

import math
from typing import NamedTuple, Sequence

import numpy as np
import rasterio.windows
import shapely
from affine import Affine
from pyproj import CRS, Transformer
from rasterio.features import geometry_mask
from rasterio.windows import Window
from shapely.geometry import Point, box
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from app.exceptions import SelectedAreaOutOfBoundsError


class SpatialSelection(NamedTuple):
    """An area on a dataset's grid. Validation, masking and the area all use
    it, so they share one idea of where the grid ends."""

    shapes: list[BaseGeometry]  # in the dataset's CRS, clipped to the grid
    transform: Affine  # the grid's
    window: Window  # every cell the shapes' bounds reach

    @property
    def window_cells(self) -> int:
        """The cells extraction reads: the selected ones and any between them,
        so two points far apart read every cell in between."""
        return int(self.window.width * self.window.height)


def resolve_spatial_window(
    shapes: Sequence[BaseGeometry],
    transform: Sequence[float],
    dataset_crs: str,
    grid_shape: Sequence[int],
) -> SpatialSelection:
    """Project WGS84 request shapes onto the grid (`grid_shape` is rows,
    columns) and return them, clipped to it, with their read window.

    The part of an area off the grid neither widens the window nor counts
    toward the cells or the area. A point on the grid's outer edge stays; a
    polygon must share some area with the grid, so one that only touches its
    edge is dropped. Raise if nothing is left.
    """
    dataset_transform = Affine(*transform[:6])
    rows, cols = grid_shape
    grid = box(*rasterio.windows.bounds(Window(0, 0, cols, rows), dataset_transform))
    transformer = Transformer.from_crs("EPSG:4326", dataset_crs, always_xy=True)
    # One projection and one clip for all the shapes.
    projected = shapely.transform(
        list(shapes), transformer.transform, interleaved=False
    )
    clipped = [
        part
        for part in shapely.intersection(projected, grid)
        if not part.is_empty and (isinstance(part, Point) or part.area > 0)
    ]
    if not clipped:
        if all(isinstance(shape, Point) for shape in projected):
            raise SelectedAreaOutOfBoundsError(
                "The selected point is outside this dataset's region."
            )
        raise SelectedAreaOutOfBoundsError(
            "The selected area is outside this dataset's region; "
            "choose an area that overlaps it."
        )
    exact = rasterio.windows.from_bounds(
        *shapely.total_bounds(clipped), transform=dataset_transform
    )
    # Every cell the bounds reach: floor the start and ceil the end, so no cell
    # whose center is inside the area falls outside the window. Then keep the
    # cells on the grid, and at least one, for an area on the grid's outer edge.
    col_start, col_end = _on_grid(exact.col_off, exact.width, cols)
    row_start, row_end = _on_grid(exact.row_off, exact.height, rows)
    window = Window(col_start, row_start, col_end - col_start, row_end - row_start)
    return SpatialSelection(clipped, dataset_transform, window)


def _on_grid(offset: float, length: float, size: int) -> tuple[int, int]:
    start = min(max(math.floor(offset), 0), size - 1)
    end = max(min(math.ceil(offset + length), size), start + 1)
    return start, end


def calculate_spatial_coverage(
    selection: SpatialSelection, dataset_crs: str
) -> tuple[np.ndarray, int, float]:
    """Return the window's mask of selected cells, their count, and the area.

    A point selects the cell it is in, or the edge cell for a point on the
    grid's east or south edge, which no cell's half-open extent holds.
    """
    shapes, dataset_transform, window = selection
    window_transform = rasterio.windows.transform(window, dataset_transform)
    out_shape = (int(window.height), int(window.width))
    points = [shape for shape in shapes if isinstance(shape, Point)]
    areas = [shape for shape in shapes if not isinstance(shape, Point)]

    mask = np.zeros(out_shape, dtype=bool)
    if areas:
        mask = geometry_mask(
            areas, transform=window_transform, invert=True, out_shape=out_shape
        )
        if not mask.any():
            # A sub-pixel polygon covers no cell center: take the cells it touches.
            mask = geometry_mask(
                areas,
                transform=window_transform,
                invert=True,
                out_shape=out_shape,
                all_touched=True,
            )
    for point in points:
        col, row = ~window_transform @ (point.x, point.y)
        mask[
            min(max(math.floor(row), 0), out_shape[0] - 1),
            min(max(math.floor(col), 0), out_shape[1] - 1),
        ] = True
    n_cells = int(mask.sum())

    crs = CRS.from_string(dataset_crs)
    union = unary_union(shapes)
    if crs.is_geographic:
        area, _ = crs.get_geod().geometry_area_perimeter(union)
        total_area = abs(area)
    else:
        total_area = float(union.area)

    return mask, n_cells, total_area
