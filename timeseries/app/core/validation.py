import math
from typing import Sequence

import rasterio.windows
from affine import Affine
from pyproj import Transformer
from rasterio.windows import Window
from shapely.ops import unary_union
from shapely.ops import transform as transform_geometry
from shapely.geometry import Point as ShapelyPoint
from shapely.geometry.base import BaseGeometry

from app.registry.compose import AppRegistry, ServedRelease

# ---------------------------------------------------------------------------
# Dataset and variable validation to prevent arbitrary or malicious queries


def validate_dataset_and_variable(
    registry: AppRegistry, dataset_id: str, variable_id: str
) -> ServedRelease:
    """
    Validates dataset and variable existence to prevent arbitrary or malicious queries.
    Returns the dataset's release; raises ValueError if either ID is not served.
    """
    release = registry.get(dataset_id)
    if release is None:
        raise ValueError(f"Dataset '{dataset_id}' not found.")
    if not release.has_variable(variable_id):
        raise ValueError(
            f"Variable '{variable_id}' not found in dataset '{dataset_id}'."
        )
    return release


# ---------------------------------------------------------------------------
# Geometry size validation to prevent excessively large queries


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


def validate_geom_size(
    shapes: list[BaseGeometry],
    transform: Sequence[float],
    dataset_crs: str,
    max_cells: int,
) -> None:
    """
    Validates that the geometry does not exceed a maximum number of cells when rasterized
    on the dataset's grid. Raises ValueError if the geometry is too large.
    """
    if all(isinstance(s, ShapelyPoint) for s in shapes):
        return  # A point is exactly 1 cell — always within limits
    estimated_cells = estimate_cell_count(shapes, transform, dataset_crs)

    if estimated_cells > max_cells:
        raise ValueError(
            f"Selected area is too large. Estimated cell count: {estimated_cells}, maximum allowed: {max_cells}."
        )
