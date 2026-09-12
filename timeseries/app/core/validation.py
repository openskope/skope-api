import math
import re
from typing import Sequence

import rasterio.windows
from affine import Affine
from pyproj import Transformer
from rasterio.windows import Window
from shapely.ops import unary_union
from shapely.ops import transform as transform_geometry
from shapely.geometry import Point as ShapelyPoint
from shapely.geometry.base import BaseGeometry

COLORMAP_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

# ---------------------------------------------------------------------------
# Dataset and variable validation to prevent arbitrary or malicious queries


def validate_dataset_and_variable(
    registry: dict, dataset_id: str, variable_id: str
) -> None:
    """
    Validates dataset and variable existence to prevent arbitrary or malicious queries.
    Raises ValueError if the IDs are not found in the registry.
    """
    dataset = registry.get(dataset_id)
    if not dataset:
        raise ValueError(f"Dataset '{dataset_id}' not found.")

    variables = dataset.get("variables", [])

    if not any(var.get("id") == variable_id for var in variables):
        raise ValueError(
            f"Variable '{variable_id}' not found in dataset '{dataset_id}'."
        )


def validate_tile_style(colormap: str, rescale: str) -> tuple[str, str]:
    if not COLORMAP_NAME_PATTERN.fullmatch(colormap):
        raise ValueError("Invalid colormap name.")

    try:
        lower_text, upper_text = rescale.split(",")
        lower = float(lower_text)
        upper = float(upper_text)
    except ValueError as exc:
        raise ValueError("Rescale must contain exactly two numeric values.") from exc

    if not math.isfinite(lower) or not math.isfinite(upper):
        raise ValueError("Rescale values must be finite.")
    if lower >= upper:
        raise ValueError("Rescale minimum must be less than maximum.")

    return colormap, f"{lower:g},{upper:g}"


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
    window = (
        rasterio.windows.from_bounds(*bounds, transform=dataset_transform)
        .round_lengths()
        .round_offsets()
    )
    window = Window(
        window.col_off,
        window.row_off,
        max(1, window.width),
        max(1, window.height),
    )
    return projected_shapes, dataset_transform, window


def estimate_cell_count(
    shapes: Sequence[BaseGeometry], transform: Sequence[float], dataset_crs: str
) -> int:
    """Return the bounding raster-window size used by extraction."""
    _, _, window = resolve_spatial_window(shapes, transform, dataset_crs)
    return math.ceil(window.width) * math.ceil(window.height)


def validate_geom_size(
    shapes: list[BaseGeometry], dataset_entry: dict, max_cells: int
) -> None:
    """
    Validates that the geometry does not exceed a maximum number of cells when rasterized.
    Accepts a list of Shapely geometries and a registry dataset entry (with 'crs' and 'transform').
    Raises ValueError if the geometry is too large.
    """
    if all(isinstance(s, ShapelyPoint) for s in shapes):
        return  # A point is exactly 1 cell — always within limits
    transform = dataset_entry["transform"]
    estimated_cells = estimate_cell_count(shapes, transform, dataset_entry["crs"])

    if estimated_cells > max_cells:
        raise ValueError(
            f"Selected area is too large. Estimated cell count: {estimated_cells}, maximum allowed: {max_cells}."
        )
