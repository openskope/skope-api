from fastapi.exceptions import RequestValidationError
from pydantic.error_wrappers import ErrorWrapper


class TimeseriesError(Exception):
    field = '__root__'

    def to_request_validation_error(self):
        return RequestValidationError([ErrorWrapper(self, ("body", self.field))])


class SelectedAreaPolygonIsNotValid(TimeseriesError):
    """Selected area polygon is not valid"""


class SelectedAreaOutOfBoundsError(TimeseriesError):
    """Selected area was outside of the dataset boundaries"""


class DatasetNotFoundError(TimeseriesError):
    """Could not find the dataset in the metadata store"""
    field = 'dataset_id'


class VariableNotFoundError(TimeseriesError):
    """Variable not found for dataset in the metadata store"""


class TimeRangeContainmentError(TimeseriesError):
    """Time range not contained by selected dataset's time range"""
