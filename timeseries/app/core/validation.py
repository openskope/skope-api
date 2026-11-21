from typing import Sequence

from shapely.geometry import Point as ShapelyPoint
from shapely.geometry.base import BaseGeometry

from app.core.spatial import estimate_cell_count
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


def validate_geom_size(
    shapes: list[BaseGeometry],
    transform: Sequence[float],
    dataset_crs: str,
    grid_shape: Sequence[int],
    max_cells: int,
) -> None:
    """
    Validates that the geometry does not exceed a maximum number of cells when rasterized
    on the dataset's grid. Raises ValueError if the geometry is too large.
    """
    if all(isinstance(s, ShapelyPoint) for s in shapes):
        return  # A point is exactly 1 cell — always within limits
    estimated_cells = estimate_cell_count(shapes, transform, dataset_crs, grid_shape)

    if estimated_cells > max_cells:
        raise ValueError(
            f"Selected area is too large. Estimated cell count: {estimated_cells}, maximum allowed: {max_cells}."
        )
