from fastapi.exceptions import RequestValidationError
from pydantic.error_wrappers import ErrorWrapper


class TimeseriesTimeoutError(Exception):
    def __init__(self, message: str, processing_time: float):
        super().__init__()
        self.message = message
        self.processing_time = processing_time


class TimeseriesValidationError(Exception):
    field = '__root__'

    def to_request_validation_error(self):
        return RequestValidationError([ErrorWrapper(self, ("body", self.field))])


class SelectedAreaPolygonIsNotValid(TimeseriesValidationError):
    """Selected area polygon is not valid"""
    field = 'selected_area'


class SelectedAreaOutOfBoundsError(TimeseriesValidationError):
    """Selected area was outside of the dataset boundaries"""


class DatasetNotFoundError(TimeseriesValidationError):
    """Could not find the dataset in the metadata store"""
    field = 'dataset_id'


class VariableNotFoundError(TimeseriesValidationError):
    """Variable not found for dataset in the metadata store"""


class TimeRangeContainmentError(TimeseriesValidationError):
    """Time range not contained by selected dataset's time range"""
