from app.core.spatial import SpatialSelection
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
# Read-window size validation to prevent excessively large queries


def validate_window_size(selection: SpatialSelection, max_cells: int) -> None:
    """Raise ValueError if the area's read window holds more than `max_cells`
    cells: shapes far apart read every cell between them, however few they
    select."""
    if selection.window_cells > max_cells:
        raise ValueError(
            "Selected area is too large: its read window spans "
            f"{selection.window_cells} cells, maximum allowed: {max_cells}."
        )
