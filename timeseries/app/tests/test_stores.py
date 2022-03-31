import pytest

from app.exceptions import TimeRangeContainmentError
from app.schemas.common import TimeRange, BandRange
from app.schemas.dataset import get_dataset_manager

yearly_cat_ds_avail = TimeRange(gte="0007-01-01", lte="0020-01-01")
monthly_cat_ds_avail = TimeRange(gte="0013-05-01", lte="0023-04-01")

dataset_manager = get_dataset_manager()

yearly_dataset = dataset_manager.get_variable_metadata(
    dataset_id="annual_5x5x5_dataset", variable_id="float32_variable"
)
yearly_dataset.time_range = yearly_cat_ds_avail
monthly_dataset = dataset_manager.get_variable_metadata(
    dataset_id="monthly_5x5x60_dataset", variable_id="float32_variable"
)
monthly_dataset.time_range = monthly_cat_ds_avail


def test_band_range_conversions():
    tr = TimeRange(gte="0007-01-01", lte="0010-01-01")

    assert yearly_dataset.find_band_range(tr) == BandRange(gte=1, lte=4)

    with pytest.raises(TimeRangeContainmentError):
        tr = TimeRange(gte="0015-01-01", lte="0025-01-01")
        yearly_dataset.find_band_range(tr)

    tr = TimeRange(gte="0015-05-01", lte="0023-04-01")

    assert monthly_dataset.find_band_range(tr) == BandRange(gte=25, lte=120)

    with pytest.raises(TimeRangeContainmentError):
        tr = TimeRange(gte="0010-05-01", lte="0023-04-01")
        monthly_dataset.find_band_range(tr)


def test_translate_band_range():
    assert yearly_dataset.translate_band_range(BandRange(gte=1, lte=1)) == TimeRange(
        gte="0007-01-01", lte="0007-01-01"
    )
    assert (
        monthly_dataset.translate_band_range(BandRange(gte=1, lte=120))
        == monthly_cat_ds_avail
    )


"""
def test_year_month_round_trip():
    ym = YearMonth(year=0, month=1)
    assert ym == ym.from_index(ym.to_months_since_0ce())
    ym = YearMonth(year=1974, month=12)
    assert ym == ym.from_index(ym.to_months_since_0ce())
    assert YearMonth(year=1975, month=1) == ym.from_index(ym.to_months_since_0ce() + 1)
"""
