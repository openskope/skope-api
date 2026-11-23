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
    """Selected area's read window holds more cells than one read may load"""

    field = "selected_area"
    template = "The selected area's read window spans {window_cells} cells; maximum is {max_cells}."

    def __init__(self, window_cells, max_cells):
        super().__init__(
            self.template.format(window_cells=window_cells, max_cells=max_cells)
        )


class SelectedAreaPolygonIsNotValid(TimeseriesValidationError):
    """Selected area polygon is not valid"""

    field = "selected_area"


class SelectedAreaOutOfBoundsError(TimeseriesValidationError):
    """Selected area was outside of the dataset boundaries"""
