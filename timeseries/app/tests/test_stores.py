import pytest

from ..stores import TimeRange, BandRange, Resolution
from ..exceptions import TimeRangeContainmentError

yearly_cat_ds_avail = TimeRange(gte='0007-01-01', lte='0020-01-01')
monthly_cat_ds_avail = TimeRange(gte='0013-05-01', lte='0023-04-01')


def test_band_range_conversions():
    yr = TimeRange(gte='0007-01-01', lte='0010-01-01')
    assert yearly_cat_ds_avail.find_band_range(yr) == BandRange(gte=1, lte=4, resolution=Resolution.year)

    with pytest.raises(TimeRangeContainmentError):
        yr = TimeRange(gte='0015-01-01', lte='0025-01-01')
        yearly_cat_ds_avail.find_band_range(yr)

    ymr = TimeRange(gte='0015-05-01', lte='0023-04-01')

    assert monthly_cat_ds_avail.find_band_range(ymr) == BandRange(gte=25, lte=120, resolution=Resolution.month)

    with pytest.raises(TimeRangeContainmentError):
        ymr = TimeRange(gte='0010-05-01', lte='0023-04-01')
        monthly_cat_ds_avail.find_band_range(ymr)


def test_translate_band_range():
    assert yearly_cat_ds_avail.translate_band_range(BandRange(gte=1, lte=1, resolution=Resolution.year)) == TimeRange(gte='0007-01-01', lte='0007-01-01')
    assert monthly_cat_ds_avail.translate_band_range(BandRange(gte=1, lte=120, resolution=Resolution.month)) == monthly_cat_ds_avail

"""
def test_year_month_round_trip():
    ym = YearMonth(year=0, month=1)
    assert ym == ym.from_index(ym.to_months_since_0ce())
    ym = YearMonth(year=1974, month=12)
    assert ym == ym.from_index(ym.to_months_since_0ce())
    assert YearMonth(year=1975, month=1) == ym.from_index(ym.to_months_since_0ce() + 1)
"""