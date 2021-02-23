import copy
import numpy as np
import pytest

from fastapi.testclient import TestClient
from httpx import AsyncClient
from ...main import app
from ...routers import datasets as ds
from ...routers.datasets import MonthAnalysisQuery, Point, OptionalYearMonthRange, ZonalStatistic
from ...settings import settings
from ...stores import YearMonth, dataset_repo

client = TestClient(app)


def test_moving_average_smoother():
    xs = np.array([1,1,1,1,1,2,2,2,2,2])
    mas = ds.MovingAverageSmoother(method='centered', width=2)
    smoothed_xs = mas.apply(xs)
    assert np.allclose(smoothed_xs, np.array([1, (4 + 2)/5, (3 + 4)/5, (2 + 6)/5, (1 + 8)/5, 2]))


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
            coordinates=(-123,45)
        ),
        transforms=[],
        zonal_statistic=ZonalStatistic.mean.value
    )
    async with AsyncClient(app=app, base_url='http://test') as ac:
        response = await ac.post('datasets/monthly', data=maq.json())
    assert response.status_code == 200
    assert response.json()['values'] == [i*100 for i in br]


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
            response = await ac.post('datasets/monthly', data=data.json())
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
        response = await ac.post('datasets/monthly', data=maq.json())
        assert response.status_code == 504
