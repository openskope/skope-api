import pytest

from ..stores import YearRange, YearMonthRange, YearMonth, BandRange, TimeRangeContainmentError

yearly_cat_ds_avail = YearRange(gte=7, lte=20)
monthly_cat_ds_avail = YearMonthRange(
    gte=YearMonth(year=13, month=5),
    lte=YearMonth(year=23, month=4)
)


def test_band_range_conversions():
    yr = YearRange(gte=7, lte=10)
    assert yearly_cat_ds_avail.find_band_range(yr) == BandRange(gte=1, lte=4)

    with pytest.raises(TimeRangeContainmentError):
        yr = YearRange(gte=15, lte=25)
        yearly_cat_ds_avail.find_band_range(yr)

    ymr = YearMonthRange(
        gte=YearMonth(year=15, month=5),
        lte=YearMonth(year=23, month=4)
    )
    assert monthly_cat_ds_avail.find_band_range(ymr) == BandRange(gte=25, lte=120)

    with pytest.raises(TimeRangeContainmentError):
        ymr = YearMonthRange(
            gte=YearMonth(year=10, month=5),
            lte=YearMonth(year=23, month=4)
        )
        monthly_cat_ds_avail.find_band_range(ymr)


def test_translate_band_range():
    assert yearly_cat_ds_avail.translate_band_range(BandRange(gte=1, lte=1)) == YearRange(gte=7, lte=7)
    assert monthly_cat_ds_avail.translate_band_range(BandRange(gte=1, lte=120)) == monthly_cat_ds_avail