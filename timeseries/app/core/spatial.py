"""Map a requested area onto a dataset's grid: its read window and its cells."""

import math
from typing import Sequence

import numpy as np
import rasterio
import rasterio.windows
from affine import Affine
from pyproj import CRS, Transformer
from rasterio.features import geometry_mask
from rasterio.windows import Window
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as transform_geometry
from shapely.ops import unary_union


def resolve_spatial_window(
    shapes: Sequence[BaseGeometry],
    transform: Sequence[float],
    dataset_crs: str,
) -> tuple[list[BaseGeometry], Affine, Window]:
    """Project WGS84 request shapes and return their exact raster read window."""
    dataset_transform = Affine(*transform[:6])
    transformer = Transformer.from_crs("EPSG:4326", dataset_crs, always_xy=True)
    projected_shapes = [
        transform_geometry(transformer.transform, shape) for shape in shapes
    ]
    bounds = unary_union(projected_shapes).bounds
    exact = rasterio.windows.from_bounds(*bounds, transform=dataset_transform)
    # Every cell the bounds reach: floor the start and ceil the end, so no cell
    # whose center is inside the area falls outside the window.
    col_off, row_off = math.floor(exact.col_off), math.floor(exact.row_off)
    window = Window(
        col_off,
        row_off,
        max(1, math.ceil(exact.col_off + exact.width) - col_off),
        max(1, math.ceil(exact.row_off + exact.height) - row_off),
    )
    return projected_shapes, dataset_transform, window


def estimate_cell_count(
    shapes: Sequence[BaseGeometry], transform: Sequence[float], dataset_crs: str
) -> int:
    """Return the bounding raster-window size used by extraction."""
    _, _, window = resolve_spatial_window(shapes, transform, dataset_crs)
    return math.ceil(window.width) * math.ceil(window.height)


def calculate_spatial_coverage(
    shapes: list,
    dataset_transform: rasterio.Affine,
    window: Window,
    dataset_crs: str,
) -> tuple[np.ndarray, int, float]:
    window_transform = rasterio.windows.transform(window, dataset_transform)
    mask = geometry_mask(
        shapes,
        transform=window_transform,
        invert=True,
        out_shape=(window.height, window.width),
    )
    if np.sum(mask) == 0:
        # Fallback for point geometries or small, sub-pixel polygons that don't cover any pixel center
        mask = geometry_mask(
            shapes,
            transform=window_transform,
            invert=True,
            out_shape=(window.height, window.width),
            all_touched=True,
        )
    n_cells = int(np.sum(mask))

    crs = CRS.from_string(dataset_crs)
    union = unary_union(shapes)
    if crs.is_geographic:
        area, _ = crs.get_geod().geometry_area_perimeter(union)
        total_area = abs(area)
    else:
        total_area = float(union.area)

    return mask, n_cells, total_area
