from datetime import datetime, timezone

from dateutil.relativedelta import relativedelta

from cog_stac_pipeline.datetime_utils import (
    format_stac_datetime,
    generate_date_range,
    get_iso_key,
    singular_to_plural_for_relativedelta,
)


def test_get_iso_key_matches_time_resolution():
    dt = datetime(103, 4, 5, 6, 7, 8, tzinfo=timezone.utc)

    assert get_iso_key(dt, {"years": 1}) == "0103"
    assert get_iso_key(dt, {"months": 1}) == "0103-04"
    assert get_iso_key(dt, {"days": 1}) == "0103-04-05"
    assert get_iso_key(dt, {"hours": 1}) == "0103-04-05T06:07:08Z"


def test_format_stac_datetime_zero_pads_early_years():
    dt = datetime(103, 1, 2, 3, 4, 5, tzinfo=timezone.utc)

    assert format_stac_datetime(dt) == "0103-01-02T03:04:05Z"


def test_generate_date_range_includes_end_date():
    dates = list(
        generate_date_range(
            datetime(1, 1, 1, tzinfo=timezone.utc),
            datetime(3, 1, 1, tzinfo=timezone.utc),
            relativedelta(years=1),
        )
    )

    assert [date.year for date in dates] == [1, 2, 3]


def test_singular_to_plural_for_relativedelta():
    assert singular_to_plural_for_relativedelta({"year": 1, "months": 2}) == {
        "years": 1,
        "months": 2,
    }
