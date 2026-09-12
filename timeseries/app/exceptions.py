from fastapi.exceptions import RequestValidationError


class TimeseriesValidationError(Exception):
    field = "__root__"

    def to_request_validation_error(self):
        return RequestValidationError(
            [
                {
                    "type": "value_error",
                    "loc": ("body", self.field),
                    "msg": str(self),
                    "input": None,
                    "ctx": {"error": str(self)},
                }
            ]
        )


class SelectedAreaPolygonIsTooLarge(TimeseriesValidationError):
    """Selected area polygon contains more cells than analysis service can load"""

    field = "selected_area"
    template = "The selected polygon exceeds the max allowable cells: {n_cells} > max {max_cells} "

    def __init__(self, n_cells, max_cells):
        super().__init__(self.template.format(n_cells=n_cells, max_cells=max_cells))


class SelectedAreaPolygonIsNotValid(TimeseriesValidationError):
    """Selected area polygon is not valid"""

    field = "selected_area"


class SelectedAreaOutOfBoundsError(TimeseriesValidationError):
    """Selected area was outside of the dataset boundaries"""
