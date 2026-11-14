"""Temporal axes and the time-to-band rule (AT-007, AT-008; OBS-004, API-002 to API-004)."""

import pytest

from skope_release.timeaxis import Axis, AxisError, ChunkLayout, MalformedTimestep, UnknownTimestep, fname


def test_at_008_enumerated_axis():
    axis = Axis.enumerated(["0100", "0103", "0104", "0110", "0111"], "year")
    layout = ChunkLayout("ds", axis, 2)
    assert layout.resolve("v", "0100") == ("cogs/v/ds--0100--0103.tif", 1)
    assert layout.resolve("v", "0104") == ("cogs/v/ds--0104--0110.tif", 1)
    assert layout.resolve("v", "0110") == ("cogs/v/ds--0104--0110.tif", 2)
    assert layout.resolve("v", "0111") == ("cogs/v/ds--0111--0111.tif", 1)
    with pytest.raises(UnknownTimestep):
        layout.resolve("v", "0105")
    assert sum((layout.chunk_keys(c) for c in range(layout.chunk_count)), ()) == axis.values


def test_at_008_regular_axis_rejects_a_remainder():
    axis = Axis.regular("0100", "P5Y", "0120", "year")
    assert axis.count == 5 and axis.keys() == ("0100", "0105", "0110", "0115", "0120")
    with pytest.raises(UnknownTimestep):
        axis.index("0102")


def test_paleocar_shaped_axis():
    axis = Axis.regular("0103", "P1Y", "2000", "year")
    layout = ChunkLayout("paleocar_v3", axis, 100)
    assert axis.count == 1898 and layout.chunk_count == 19
    assert layout.item_id(0) == "paleocar_v3--0103--0202"
    assert layout.item_id(18) == "paleocar_v3--1903--2000"
    assert layout.resolve("ppt_annual", "0103") == ("cogs/ppt_annual/paleocar_v3--0103--0202.tif", 1)
    assert layout.resolve("ppt_annual", "1000") == ("cogs/ppt_annual/paleocar_v3--0903--1002.tif", 98)
    assert layout.resolve("ppt_annual", "2000") == ("cogs/ppt_annual/paleocar_v3--1903--2000.tif", 98)
    # API-004: unique, ordered resolutions over the whole axis.
    resolved = [layout.resolve("v", k) for k in axis.keys()]
    assert len(set(resolved)) == axis.count
    assert [layout.locate(i) for i in range(axis.count)] == sorted(layout.locate(i) for i in range(axis.count))


@pytest.mark.parametrize("key", ["103", "01030", "0103-01", "abcd", "0000"])
def test_malformed_keys(key):
    axis = Axis.regular("0103", "P1Y", "2000", "year")
    with pytest.raises(MalformedTimestep):
        axis.index(key)


@pytest.mark.parametrize("key", ["0102", "2001", "9999"])
def test_off_axis_keys(key):
    axis = Axis.regular("0103", "P1Y", "2000", "year")
    with pytest.raises(UnknownTimestep):
        axis.index(key)


def test_monthly_and_daily_axes():
    monthly = Axis.regular("2000-11", "P1M", "2001-02", "month")
    assert monthly.keys() == ("2000-11", "2000-12", "2001-01", "2001-02")
    daily = Axis.regular("2000-02-27", "P1D", "2000-03-01", "day")
    assert daily.keys() == ("2000-02-27", "2000-02-28", "2000-02-29", "2000-03-01")
    hourly = Axis.regular("2000-01-01T00:00:00Z", "PT6H", "2000-01-01T18:00:00Z", "datetime")
    assert hourly.count == 4
    assert fname("2000-01-01T06:00:00Z") == "2000-01-01T06-00-00Z"


@pytest.mark.parametrize(
    "origin,step,end,precision",
    [
        ("0100", "P2Y", "0103", "year"),  # never reaches end exactly (OBS-004)
        ("0100", "P0Y", "0100", "year"),  # step must be positive
        ("0103", "P1Y", "0100", "year"),  # end before origin
        ("2000", "P1M", "2001", "year"),  # step finer than precision
        ("0100", "1Y", "0101", "year"),  # not an ISO duration
    ],
)
def test_invalid_regular_axes(origin, step, end, precision):
    with pytest.raises(AxisError):
        Axis.regular(origin, step, end, precision)


@pytest.mark.parametrize("values", [["0100", "0100"], ["0101", "0100"], []])
def test_invalid_enumerated_axes(values):
    with pytest.raises(AxisError):
        Axis.enumerated(values, "year")


def test_range_selection():
    axis = Axis.regular("0103", "P1Y", "2000", "year")
    assert [axis.key(i) for i in axis.select("0105", "0107")] == ["0105", "0106", "0107"]
    assert axis.select("2001", "2005") == []
