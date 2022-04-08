from fastapi.testclient import TestClient
from httpx import AsyncClient

import copy
import logging
import numpy as np
import pytest
import rasterio


from app.config import get_settings
from app.core.services import extract_timeseries
from app.exceptions import SelectedAreaPolygonIsTooLarge, TimeseriesTimeoutError
from app.main import app
from app.schemas.common import TimeRange, OptionalTimeRange, BandRange
from app.schemas.dataset import get_dataset_manager
from app.schemas.geometry import (
    SkopePointModel,
    SkopePolygonModel,
)
from app.schemas.timeseries import (
    ZonalStatistic,
    TimeseriesRequest,
    NoSmoother,
    MovingAverageSmoother,
    NoTransform,
)

settings = get_settings()
logger = logging.getLogger(__name__)

client = TestClient(app)

dataset_manager = get_dataset_manager()


def test_moving_average_smoother():
    xs = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
    mas = MovingAverageSmoother(method="centered", width=3)
    smoothed_xs = mas.apply(xs)
    assert np.allclose(smoothed_xs, np.array([1, 1, 1, 4 / 3, 5 / 3, 2, 2, 2]))
    assert len(smoothed_xs) == len(xs) - 2


def build_timeseries_query(**overrides):
    query_parameters = {
        "selected_area": SkopePointModel(type="Point", coordinates=(-123, 45)).dict(),
        "transform": NoTransform().dict(),
        "requested_series": [{"name": "original", "smoother": {"type": "NoSmoother"}}],
        "zonal_statistic": ZonalStatistic.mean.value,
    }
    query_parameters.update(overrides)
    return TimeseriesRequest(**query_parameters)


# FIXME: should assert / verify out of band errors when the timerange exceeds the dataset bounds
TIME_RANGES = [
    OptionalTimeRange(gte="0001-01-01", lte="0003-01-01"),
    OptionalTimeRange(gte="0001-01-01", lte="0005-01-01"),
    OptionalTimeRange(gte="0002-01-01", lte="0004-01-01"),
    OptionalTimeRange(gte="0003-01-01", lte="0004-01-01"),
    OptionalTimeRange(gte="0003-01-01", lte="0005-01-01"),
    OptionalTimeRange(gte="0003-01-01", lte="0003-01-01"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "variable_id", dataset_manager.get_variables("annual_5x5x5_dataset")
)
@pytest.mark.parametrize("time_range", TIME_RANGES)
async def test_annual_time_ranges(variable_id, time_range):
    ds_meta = dataset_manager.get_variable_metadata(
        dataset_id="annual_5x5x5_dataset", variable_id=variable_id
    )
    br = ds_meta.find_band_range(time_range)
    timeseries_query = build_timeseries_query(
        dataset_id="annual_5x5x5_dataset",
        variable_id=variable_id,
        time_range=time_range,
        zonal_statistic=ZonalStatistic.mean.value,
    )
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post(settings.base_uri, content=timeseries_query.json())
    assert response.status_code == 200
    logger.debug("response: %s", response)
    assert response.json()["series"][0]["values"] == [i * 100 for i in br]


@pytest.mark.asyncio
async def test_annual_different_smoothers():
    tsq = build_timeseries_query(
        dataset_id="annual_5x5x5_dataset",
        variable_id="float32_variable",
        requested_series=[
            {"name": "original", "smoother": NoSmoother().dict()},
            {
                "name": "trailing",
                "smoother": MovingAverageSmoother(method="trailing", width=2).dict(),
            },
            {
                "name": "centered",
                "smoother": MovingAverageSmoother(method="centered", width=3),
            },
        ],
        time_range=OptionalTimeRange(gte="0001-01-01", lte="0004-01-01").dict(),
    )
    response = await extract_timeseries(tsq, dataset_manager)
    original = response.series[0]
    trailing = response.series[1]
    centered = response.series[2]

    assert original.time_range == TimeRange(gte="0001-01-01", lte="0004-01-01")
    assert len(original.values) == 4
    # trailing average should only return data for years 3 and 4 because year 2 would go outside the range of data
    # (did not request year 0 data)
    assert trailing.time_range == TimeRange(gte="0003-01-01", lte="0004-01-01")
    assert len(trailing.values) == 2
    assert centered.time_range == TimeRange(gte="0002-01-01", lte="0004-01-01")
    assert len(centered.values) == 3


@pytest.mark.asyncio
async def test_missing_property():
    time_range = OptionalTimeRange(gte="0001-01-01", lte="0001-12-01")
    maq = build_timeseries_query(
        dataset_id="monthly_5x5x60_dataset",
        variable_id="float32_variable",
        time_range=time_range,
    )
    async with AsyncClient(app=app, base_url="http://test") as ac:
        for key in set(maq.dict().keys()).difference({"max_processing_time"}):
            data = copy.deepcopy(maq)
            data.__dict__.pop(key)
            response = await ac.post(settings.base_uri, content=data.json())
            assert response.status_code == 422
            assert response.json()["detail"][0]["loc"] == ["body", key]


@pytest.mark.asyncio
async def test_timeout():
    time_range = OptionalTimeRange(gte="0001-01-01", lte="0005-01-01")
    maq = build_timeseries_query(
        dataset_id="annual_5x5x5_dataset",
        variable_id="float32_variable",
        time_range=time_range,
        max_processing_time=0,
    )

    with pytest.raises(TimeseriesTimeoutError):
        response = await extract_timeseries(maq, dataset_manager)
        assert response is None


def test_split_indices():
    with rasterio.open("data/monthly_5x5x60_dataset_float32_variable.tif") as ds:
        width = ds.shape[0]
        height = ds.shape[1]
        br = BandRange(1, 60)
        assert list(
            SkopePolygonModel._make_band_range_groups(
                width=width, height=height, band_range=br, max_size=34
            )
        ) == [range(i * 1 + 1, (i + 1) * 1 + 1) for i in range(60)]
        assert list(
            SkopePolygonModel._make_band_range_groups(
                width=width, height=height, band_range=br, max_size=57
            )
        ) == [range(i * 2 + 1, (i + 1) * 2 + 1) for i in range(30)]
        assert list(
            SkopePolygonModel._make_band_range_groups(
                width=width, height=height, band_range=br, max_size=76
            )
        ) == [range(i * 3 + 1, (i + 1) * 3 + 1) for i in range(20)]
        assert list(
            SkopePolygonModel._make_band_range_groups(
                width=width, height=height, band_range=br, max_size=100
            )
        ) == [range(i * 4 + 1, (i + 1) * 4 + 1) for i in range(15)]
        assert list(
            SkopePolygonModel._make_band_range_groups(
                width=width, height=height, band_range=br, max_size=129
            )
        ) == [range(i * 5 + 1, (i + 1) * 5 + 1) for i in range(12)]
        assert list(
            SkopePolygonModel._make_band_range_groups(
                width=width, height=height, band_range=br, max_size=163
            )
        ) == [range(i * 6 + 1, (i + 1) * 6 + 1) for i in range(10)]
        assert list(
            SkopePolygonModel._make_band_range_groups(
                width=width, height=height, band_range=br, max_size=255
            )
        ) == [range(i * 10 + 1, (i + 1) * 10 + 1) for i in range(6)]
        assert list(
            SkopePolygonModel._make_band_range_groups(
                width=width, height=height, band_range=br, max_size=923
            )
        ) == [range(1, 37), range(37, 61)]
        assert list(
            SkopePolygonModel._make_band_range_groups(
                width=width, height=height, band_range=br, max_size=3264
            )
        ) == [range(1, 61)]

        with pytest.raises(SelectedAreaPolygonIsTooLarge):
            list(
                SkopePolygonModel._make_band_range_groups(
                    width=width, height=height, band_range=br, max_size=13
                )
            )

        br = BandRange(20, 45)
        assert list(
            SkopePolygonModel._make_band_range_groups(
                width=width, height=height, band_range=br, max_size=34
            )
        ) == [range(i * 1 + 20, (i + 1) * 1 + 20) for i in range(26)]
        assert list(
            SkopePolygonModel._make_band_range_groups(
                width=width, height=height, band_range=br, max_size=700
            )
        ) == [range(20, 46)]
        assert list(
            SkopePolygonModel._make_band_range_groups(
                width=width, height=height, band_range=br, max_size=325
            )
        ) == [range(20, 33), range(33, 46)]
        assert list(
            SkopePolygonModel._make_band_range_groups(
                width=width, height=height, band_range=br, max_size=300
            )
        ) == [range(20, 32), range(32, 44), range(44, 46)]
        assert list(
            SkopePolygonModel._make_band_range_groups(
                width=width, height=height, band_range=br, max_size=627
            )
        ) == [range(20, 45), range(45, 46)]
