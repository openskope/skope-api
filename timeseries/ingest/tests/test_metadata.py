from datetime import datetime, timezone

import pytest

from cog_stac_pipeline.metadata import validate_else_add_temporal_end


START_YEAR_ONE = datetime(1, 1, 1, tzinfo=timezone.utc)
START_PRISM = datetime(1895, 1, 1, tzinfo=timezone.utc)


def metadata_with_end(end):
    return {"timespan": {"period": {"gte": "0001", "lte": end}}}


def test_temporal_end_accepts_matching_yearly_coverage():
    dataset_metadata = metadata_with_end("2000")

    updated = validate_else_add_temporal_end(
        dataset_metadata,
        "paleocar_v2",
        {"gdd": 2000, "ppt": 2000},
        START_YEAR_ONE,
        {"years": 1},
    )

    assert updated is False


def test_temporal_end_accepts_matching_monthly_coverage():
    dataset_metadata = metadata_with_end("2013-07")

    updated = validate_else_add_temporal_end(
        dataset_metadata,
        "prism",
        {"ppt": 1423, "tmax": 1423},
        START_PRISM,
        {"months": 1},
    )

    assert updated is False


@pytest.mark.parametrize(
    ("band_count", "derived_end"),
    [(1420, "2013-04"), (1424, "2013-08")],
)
def test_temporal_end_rejects_short_or_long_raster(band_count, derived_end):
    dataset_metadata = metadata_with_end("2013-07")

    with pytest.raises(ValueError) as error:
        validate_else_add_temporal_end(
            dataset_metadata,
            "prism",
            {"tmax": band_count},
            START_PRISM,
            {"months": 1},
        )

    message = str(error.value)
    assert "dataset 'prism'" in message
    assert "variable 'tmax'" in message
    assert f"band count {band_count}" in message
    assert f"derived endpoint '{derived_end}'" in message
    assert "declared endpoint '2013-07'" in message


def test_temporal_end_rejects_inconsistent_variables_when_lte_is_missing():
    dataset_metadata = {"timespan": {"period": {"gte": "1895-01"}}}

    with pytest.raises(ValueError, match="Inconsistent temporal coverage") as error:
        validate_else_add_temporal_end(
            dataset_metadata,
            "prism",
            {"ppt": 1423, "tmax": 1420},
            START_PRISM,
            {"months": 1},
        )

    message = str(error.value)
    assert "variable 'ppt', band count 1423, derived endpoint '2013-07'" in message
    assert "variable 'tmax', band count 1420, derived endpoint '2013-04'" in message
    assert "declared endpoint '<missing>'" in message


def test_temporal_end_adds_missing_lte_for_consistent_variables():
    dataset_metadata = {"timespan": {"period": {"gte": "0001"}}}

    updated = validate_else_add_temporal_end(
        dataset_metadata,
        "paleocar_v2",
        {"gdd": 2000, "ppt": 2000},
        START_YEAR_ONE,
        {"years": 1},
    )

    assert updated is True
    assert dataset_metadata["timespan"]["period"]["lte"] == "2000"
