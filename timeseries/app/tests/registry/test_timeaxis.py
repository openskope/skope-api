"""The API resolves timesteps with the same rule the release build used (API-002)."""

import pytest

from app.vendor.timeaxis import Axis, ChunkLayout, MalformedTimestep, UnknownTimestep

# The paleocar_v3 prototype release: 0103-2000, 100 timesteps per COG.
LAYOUT = ChunkLayout("paleocar_v3", Axis.regular("0103", "P1Y", "2000", "year"), 100)


@pytest.mark.parametrize(
    "key, path, band",
    [
        ("0103", "cogs/ppt_annual/paleocar_v3--0103--0202.tif", 1),
        ("0590", "cogs/ppt_annual/paleocar_v3--0503--0602.tif", 88),
        ("2000", "cogs/ppt_annual/paleocar_v3--1903--2000.tif", 98),
    ],
)
def test_resolves_first_middle_and_last_timesteps(key, path, band):
    assert LAYOUT.resolve("ppt_annual", key) == (path, band)


@pytest.mark.parametrize("key", ["103", "0103-01", "abcd", ""])
def test_malformed_key_is_rejected(key):
    with pytest.raises(MalformedTimestep):
        LAYOUT.resolve("ppt_annual", key)


@pytest.mark.parametrize("key", ["0102", "2001"])
def test_key_off_the_axis_is_unknown(key):
    with pytest.raises(UnknownTimestep):
        LAYOUT.resolve("ppt_annual", key)


def test_between_step_key_is_unknown():
    layout = ChunkLayout("ds", Axis.regular("0001", "P2Y", "0009", "year"), 3)

    with pytest.raises(UnknownTimestep):
        layout.resolve("v", "0002")
