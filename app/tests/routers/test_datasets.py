import copy

import numpy as np
import pytest
import rasterio
from fastapi.testclient import TestClient
from httpx import AsyncClient

from ...exceptions import SelectedAreaPolygonIsTooLarge
from ...main import app
from ...routers import datasets as ds
from ...routers.datasets import MonthAnalysisQuery, Point, OptionalYearMonthRange, ZonalStatistic, Polygon
from ...stores import YearMonth, dataset_repo, BandRange

client = TestClient(app)


def test_moving_average_smoother():
    xs = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
    mas = ds.MovingAverageSmoother(method='centered', width=2)
    smoothed_xs = mas.apply(xs)
    assert np.allclose(smoothed_xs, np.array([1, (4 + 2) / 5, (3 + 4) / 5, (2 + 6) / 5, (1 + 8) / 5, 2]))


ymrs = [
    OptionalYearMonthRange(
        gte=YearMonth(year=1, month=1),
        lte=YearMonth(year=1, month=12)
    ),
    OptionalYearMonthRange(
        gte=YearMonth(year=3, month=1),
        lte=YearMonth(year=3, month=12)
    ),
    OptionalYearMonthRange(
        gte=YearMonth(year=4, month=7),
        lte=YearMonth(year=5, month=6)
    )
]

MONTHLY_TIME_SERIES_URL = '/timeseries-service/api/v2/datasets/monthly'


@pytest.mark.asyncio
@pytest.mark.parametrize("variable_id", dataset_repo.get_dataset_variables('monthly_5x5x60_dataset', 'month'))
@pytest.mark.parametrize("time_range", ymrs)
async def test_monthly_first_year(variable_id, time_range):
    ds_meta = dataset_repo.get_dataset_meta(
        dataset_id='monthly_5x5x60_dataset',
        variable_id=variable_id,
        resolution='month'
    )
    time_avail = ds_meta.time_range
    tr = time_range.normalize(time_avail)
    br = time_avail.find_band_range(tr)
    maq = MonthAnalysisQuery(
        dataset_id='monthly_5x5x60_dataset',
        variable_id=variable_id,
        time_range=time_range,
        selected_area=Point(
            type='Point',
            coordinates=(-123, 45)
        ),
        transforms=[],
        zonal_statistic=ZonalStatistic.mean.value
    )
    async with AsyncClient(app=app, base_url='http://test') as ac:
        response = await ac.post(MONTHLY_TIME_SERIES_URL, data=maq.json())
    assert response.status_code == 200
    assert response.json()['values'] == [i * 100 for i in br]


@pytest.mark.asyncio
async def test_missing_property():
    time_range = OptionalYearMonthRange(
        gte=YearMonth(year=1, month=1),
        lte=YearMonth(year=1, month=12)
    )
    maq = MonthAnalysisQuery(
        dataset_id='monthly_5x5x60_dataset',
        variable_id='float32_variable',
        time_range=time_range,
        selected_area=Point(
            type='Point',
            coordinates=(-123, 45)
        ),
        transforms=[],
        zonal_statistic=ZonalStatistic.mean.value
    )

    async with AsyncClient(app=app, base_url='http://test') as ac:
        for key in set(maq.dict().keys()).difference({'resolution', 'max_processing_time'}):
            data = copy.deepcopy(maq)
            data.__dict__.pop(key)
            response = await ac.post(MONTHLY_TIME_SERIES_URL, data=data.json())
            assert response.status_code == 422
            assert response.json()['detail'][0]['loc'] == ['body', key]


@pytest.mark.asyncio
async def test_timeout():
    time_range = OptionalYearMonthRange(
        gte=YearMonth(year=1, month=1),
        lte=YearMonth(year=1, month=12)
    )
    maq = MonthAnalysisQuery(
        dataset_id='monthly_5x5x60_dataset',
        variable_id='float32_variable',
        time_range=time_range,
        selected_area=Point(
            type='Point',
            coordinates=(-123, 45)
        ),
        transforms=[],
        zonal_statistic=ZonalStatistic.mean.value,
        max_processing_time=0
    )

    async with AsyncClient(app=app, base_url='http://test') as ac:
        response = await ac.post(MONTHLY_TIME_SERIES_URL, data=maq.json())
        assert response.status_code == 504


def test_split_indices():
    with rasterio.open('data/monthly_5x5x60_dataset_float32_variable.tif') as ds:
        width = ds.shape[0]
        height= ds.shape[1]
        br = BandRange(1, 60)
        assert list(Polygon._make_band_range_groups(width=width, height=height, band_range=br, max_size=34)) == \
               [range(i * 1 + 1, (i + 1) * 1 + 1) for i in range(60)]
        assert list(Polygon._make_band_range_groups(width=width, height=height, band_range=br, max_size=57)) == \
               [range(i * 2 + 1, (i + 1) * 2 + 1) for i in range(30)]
        assert list(Polygon._make_band_range_groups(width=width, height=height, band_range=br, max_size=76)) == \
               [range(i * 3 + 1, (i + 1) * 3 + 1) for i in range(20)]
        assert list(Polygon._make_band_range_groups(width=width, height=height, band_range=br, max_size=100)) == \
               [range(i * 4 + 1, (i + 1) * 4 + 1) for i in range(15)]
        assert list(Polygon._make_band_range_groups(width=width, height=height, band_range=br, max_size=129)) == \
               [range(i * 5 + 1, (i + 1) * 5 + 1) for i in range(12)]
        assert list(Polygon._make_band_range_groups(width=width, height=height, band_range=br, max_size=163)) == \
               [range(i * 6 + 1, (i + 1) * 6 + 1) for i in range(10)]
        assert list(Polygon._make_band_range_groups(width=width, height=height, band_range=br, max_size=255)) == \
               [range(i * 10 + 1, (i + 1) * 10 + 1) for i in range(6)]
        assert list(Polygon._make_band_range_groups(width=width, height=height, band_range=br, max_size=923)) == \
               [range(1, 37), range(37, 61)]
        assert list(Polygon._make_band_range_groups(width=width, height=height, band_range=br, max_size=3264)) == [range(1, 61)]

        with pytest.raises(SelectedAreaPolygonIsTooLarge):
            list(Polygon._make_band_range_groups(width=width, height=height, band_range=br, max_size=13))

        br = BandRange(20, 45)
        assert list(Polygon._make_band_range_groups(width=width, height=height, band_range=br, max_size=34)) == \
               [range(i * 1 + 20, (i + 1) * 1 + 20) for i in range(26)]
        assert list(Polygon._make_band_range_groups(width=width, height=height, band_range=br, max_size=700)) == [range(20,46)]
        assert list(Polygon._make_band_range_groups(width=width, height=height, band_range=br, max_size=325)) == \
               [range(20, 33), range(33, 46)]
        assert list(Polygon._make_band_range_groups(width=width, height=height, band_range=br, max_size=300)) == \
               [range(20, 32), range(32, 44), range(44, 46)]
        assert list(Polygon._make_band_range_groups(width=width, height=height, band_range=br, max_size=627)) == \
               [range(20, 45), range(45, 46)]


