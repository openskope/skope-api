import pytest
from pydantic import ValidationError

from app.schemas.timeseries import (
    TimeRange,
    MovingAverageSmoother,
    TimeseriesRequest,
)

# ---------------------------------------------------------------------------
# TimeRange


def test_time_range_valid():
    tr = TimeRange(gte="0001-01-01", lte="0005-12-31")
    assert tr.gte == "0001-01-01"
    assert tr.lte == "0005-12-31"


def test_time_range_equal_dates_valid():
    tr = TimeRange(gte="0003-06-15", lte="0003-06-15")
    assert tr.gte == tr.lte


def test_time_range_gte_after_lte_raises():
    with pytest.raises(ValidationError):
        TimeRange(gte="0010-01-01", lte="0001-01-01")


def test_time_range_invalid_format_raises():
    with pytest.raises(ValidationError):
        TimeRange(gte="not-a-date", lte="0005-01-01")


def test_time_range_bare_year_valid():
    # The regex allows a bare 4-digit year (YYYY)
    tr = TimeRange(gte="0100", lte="0200")
    assert tr.gte == "0100"


# ---------------------------------------------------------------------------
# MovingAverageSmoother


def test_moving_average_smoother_odd_centered_valid():
    s = MovingAverageSmoother(method="centered", width=3)
    assert s.width == 3


def test_moving_average_smoother_even_centered_raises():
    with pytest.raises(ValidationError, match="odd"):
        MovingAverageSmoother(method="centered", width=4)


def test_moving_average_smoother_even_trailing_valid():
    s = MovingAverageSmoother(method="trailing", width=4)
    assert s.width == 4


def test_moving_average_smoother_width_1_centered_valid():
    s = MovingAverageSmoother(method="centered", width=1)
    assert s.width == 1


def test_moving_average_smoother_width_200_valid():
    s = MovingAverageSmoother(method="trailing", width=200)
    assert s.width == 200


def test_moving_average_smoother_width_201_raises():
    with pytest.raises(ValidationError):
        MovingAverageSmoother(method="trailing", width=201)


def test_moving_average_smoother_width_0_raises():
    with pytest.raises(ValidationError):
        MovingAverageSmoother(method="trailing", width=0)


# ---------------------------------------------------------------------------
# TimeseriesRequest — pattern validation (security-relevant)


def _make_request_dict(**overrides):
    defaults = {
        "dataset_id": "paleocar-v3",
        "variable_id": "ppt-annual",
        "selected_area": {"type": "Point", "coordinates": [-110.0, 38.0]},
        "zonal_statistic": "mean",
        "transform": {"type": "NoTransform"},
        "requested_series_options": [
            {"name": "raw", "smoother": {"type": "NoSmoother"}}
        ],
        "time_range": None,
    }
    defaults.update(overrides)
    return defaults


def test_timeseries_request_valid_ids():
    req = TimeseriesRequest(**_make_request_dict())
    assert req.dataset_id == "paleocar-v3"
    assert req.variable_id == "ppt-annual"


def test_timeseries_request_dataset_id_space_raises():
    with pytest.raises(ValidationError):
        TimeseriesRequest(**_make_request_dict(dataset_id="invalid id"))


def test_timeseries_request_dataset_id_path_traversal_raises():
    with pytest.raises(ValidationError):
        TimeseriesRequest(**_make_request_dict(dataset_id="../../etc/passwd"))


def test_timeseries_request_dataset_id_script_injection_raises():
    with pytest.raises(ValidationError):
        TimeseriesRequest(**_make_request_dict(dataset_id="ds<script>"))


def test_timeseries_request_variable_id_dot_raises():
    with pytest.raises(ValidationError):
        TimeseriesRequest(**_make_request_dict(variable_id="var.name"))


def test_timeseries_request_requires_at_least_one_series():
    with pytest.raises(ValidationError):
        TimeseriesRequest(**_make_request_dict(requested_series_options=[]))


def test_timeseries_request_limits_series_count():
    options = [
        {"name": f"series {index}", "smoother": {"type": "NoSmoother"}}
        for index in range(11)
    ]
    with pytest.raises(ValidationError):
        TimeseriesRequest(**_make_request_dict(requested_series_options=options))


def test_timeseries_request_requires_unique_series_names():
    options = [
        {"name": "raw", "smoother": {"type": "NoSmoother"}},
        {"name": "raw", "smoother": {"type": "NoSmoother"}},
    ]
    with pytest.raises(ValidationError, match="unique"):
        TimeseriesRequest(**_make_request_dict(requested_series_options=options))


def test_timeseries_request_limits_geometry_coordinates(monkeypatch):
    monkeypatch.setattr("app.schemas.timeseries.settings.max_geometry_coordinates", 4)
    polygon = {
        "type": "Polygon",
        "coordinates": [
            [
                [-110.1, 37.9],
                [-110.0, 37.9],
                [-110.0, 38.0],
                [-110.1, 38.0],
                [-110.1, 37.9],
            ]
        ],
    }

    with pytest.raises(ValidationError, match="coordinates"):
        TimeseriesRequest(**_make_request_dict(selected_area=polygon))
