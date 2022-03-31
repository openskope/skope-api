from fastapi.exceptions import RequestValidationError
from pydantic.error_wrappers import ErrorWrapper


class TimeseriesTimeoutError(Exception):
    def __init__(self, message: str, processing_time: float):
        super().__init__()
        self.message = message
        self.processing_time = processing_time


class TimeseriesValidationError(Exception):
    field = "__root__"

    def to_request_validation_error(self):
        return RequestValidationError([ErrorWrapper(self, ("body", self.field))])


class TimeRangeInvalid(TimeseriesValidationError):
    """Time range does not satisfy gte less than or equal to lte"""

    field = "__root__"
    template = "Start time step is greater than end time step"

    def __init__(self):
        super().__init__(self.template)


class SelectedAreaPolygonIsTooLarge(TimeseriesValidationError):
    """Selected area polygon contains more cells than analysis service is willing too load"""

    field = "selected_area"
    template = "Selected area polygon selects {n_cells} cells which is more the {max_cells} max"

    def __init__(self, n_cells, max_cells):
        super().__init__(self.template.format(n_cells=n_cells, max_cells=max_cells))


class SelectedAreaPolygonIsNotValid(TimeseriesValidationError):
    """Selected area polygon is not valid"""

    field = "selected_area"


class SelectedAreaOutOfBoundsError(TimeseriesValidationError):
    """Selected area was outside of the dataset boundaries"""


class DatasetNotFoundError(TimeseriesValidationError):
    """Could not find the dataset in the metadata store"""

    field = "dataset_id"


class VariableNotFoundError(TimeseriesValidationError):
    """Variable not found for dataset in the metadata store"""


class TimeRangeContainmentError(TimeseriesValidationError):
    """Time range not contained by selected dataset's time range"""
