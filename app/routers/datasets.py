import asyncio
import logging
import time

import numpy as np
import math
import rasterio

from enum import Enum
from fastapi import APIRouter
from geojson_pydantic import geometries as geompyd
from pydantic import BaseModel, Field
from rasterio.mask import raster_geometry_mask
from rasterio.windows import Window
from scipy import stats
from shapely import geometry as geom
from app.settings import settings
from typing import List, Optional, Tuple, Union, Literal, Sequence, Any

from shapely.validation import explain_validity

from ..exceptions import SelectedAreaOutOfBoundsError, SelectedAreaPolygonIsNotValid, TimeseriesTimeoutError
from ..stores import YearRange, YearMonthRange, dataset_repo, TimeRange, BandRange, YearMonth

logger = logging.getLogger(__name__)
router = APIRouter(tags=['datasets'])


class YearlySeries:
    def __init__(self, time_range: YearRange, values: np.array):
        self.time_range = time_range
        self.values = values


class ZonalStatistic(str, Enum):
    mean = 'mean'
    median = 'median'

    def to_numpy_call(self):
        if self == self.mean:
            return np.mean
        elif self == self.median:
            return np.median


def bounding_box(bounds) -> geom.Polygon:
    return geom.box(
        minx=bounds.left,
        miny=bounds.bottom,
        maxx=bounds.right,
        maxy=bounds.top)


class Point(geompyd.Point):
    def extract(self,
                dataset: rasterio.DatasetReader,
                zonal_statistic: ZonalStatistic,
                band_range: Sequence[int]):
        box = bounding_box(dataset.bounds)
        point = geom.Point(self.coordinates)
        if not box.covers(point):
            raise SelectedAreaOutOfBoundsError('selected area is not covered by the dataset region')
        logger.info('extracting point: %s', self)
        px, py = dataset.index(self.coordinates[0], self.coordinates[1])
        logging.info('indices: %s', (px, py))
        return dataset.read(list(band_range), window=Window(px, py, 1, 1)).flatten()

    class Config:
        schema_extra = {
            "example": {
                "type": "Point",
                "coordinates": [
                    -120,
                    42.5
                ]
            }
        }


class Polygon(geompyd.Polygon):
    def extract(self,
                dataset: rasterio.DatasetReader,
                zonal_statistic: ZonalStatistic,
                band_range: Sequence[int]):
        box = bounding_box(dataset.bounds)
        polygon = geom.Polygon(*self.coordinates)
        if not polygon.is_valid:
            raise SelectedAreaPolygonIsNotValid(
                f'selected area is not a valid polygon: {explain_validity(polygon).lower()}')
        # DE-9IM format
        # https://giswiki.hsr.ch/images/3/3d/9dem_springer.pdf
        # 'T********' means that the interior of the bounding box must intersect the interior of the selected area
        if not box.relate_pattern(polygon, 'T********'):
            raise SelectedAreaOutOfBoundsError(
                'no interior point of the selected area intersects an interior point of the dataset region')
        logger.info('extracting polygon: %s', self)
        zonal_func = zonal_statistic.to_numpy_call()
        masked, transform, window = raster_geometry_mask(dataset, [self], crop=True, all_touched=True)
        result = np.zeros(dataset.count, dtype=np.float64)
        for band in band_range:
            data = dataset.read(band, window=window)
            values = np.ma.array(data=data, mask=np.logical_or(np.equal(data, dataset.nodata), masked))
            result[band - 1] = zonal_func(values)
        logger.info('data: %s', result)
        return result


class Smoother(BaseModel):
    type: str


class WindowType(str, Enum):
    centered = 'centered'
    trailing = 'trailing'

    def get_time_range_required(self, br: BandRange, width: int):
        if self == self.centered:
            return BandRange(gte=br.gte - width, lte=br.lte + width)
        else:
            return BandRange(gte=br.gte - width, lte=br.lte)

    def get_window_size(self, width):
        if self == self.centered:
            return width * 2 + 1
        else:
            return width + 1


class MovingAverageSmoother(Smoother):
    type: Literal['MovingAverageSmoother'] = 'MovingAverageSmoother'
    method: WindowType
    width: int

    def get_desired_band_range_adjustment(self):
        if self.method == self.method.centered:
            return np.array([-self.width, self.width])
        else:
            return np.array([-self.width, 0])

    def apply(self, xs):
        window_size = self.method.get_window_size(self.width)
        return np.convolve(xs, np.ones(window_size) / window_size, 'valid')

    class Config:
        schema_extra = {
            "example": {
                "type": "MovingAverageSmoother",
                "method": WindowType.centered.value,
                "width": 1
            }
        }


class ZScoreRoller(BaseModel):
    type: Literal['ZScoreRoller'] = 'ZScoreRoller'
    width: int

    def get_desired_band_range_adjustment(self):
        return np.array([-self.width, 0])

    def apply(self, xs):
        n = len(xs) - self.width
        results = np.zeros(n)
        for i in range(n):
            results[i] = (xs[i + self.width] - np.mean(xs[i:(i + self.width)])) / np.std(xs[i:(i + self.width)])
        return results


class ZScoreScaler(BaseModel):
    type: Literal['ZScoreScaler'] = 'ZScoreScaler'

    def get_desired_band_range_adjustment(self):
        return np.array([0, 0])

    def apply(self, xs):
        return stats.zscore(xs)


class BaseAnalysisQuery(BaseModel):
    dataset_id: str = Field(..., regex='^\w+$')
    variable_id: str = Field(..., regex='^\w+$')
    selected_area: Union[Point, Polygon]
    zonal_statistic: ZonalStatistic
    max_processing_time: int = Field(settings.max_processing_time, ge=0, le=settings.max_processing_time)

    def extract_slice(self, dataset: rasterio.DatasetReader, band_range: Sequence[int]):
        return self.selected_area.extract(dataset, self.zonal_statistic, band_range=band_range)

    def get_band_range_to_extract(self, time_range_available: 'YearRange'):
        time_range = self.time_range.normalize(time_range_available)
        br_avail = time_range_available.find_band_range(time_range_available)
        desired_br = time_range_available.find_band_range(time_range).to_numpy_pair()
        for transform in self.transforms:
            desired_br += transform.get_desired_band_range_adjustment()
        compromise_br = br_avail.intersect(BandRange.from_numpy_pair(desired_br))
        return compromise_br

    def get_time_range_after_transforms(self, time_range_available: 'YearRange', extract_br: BandRange) -> 'YearRange':
        """Get the year range after values after applying transformations"""
        inds = extract_br.to_numpy_pair()
        for transform in self.transforms:
            inds += transform.get_desired_band_range_adjustment() * -1
        yr = time_range_available.translate_band_range(BandRange.from_numpy_pair(inds))
        return yr

    def transform_series(self, xs):
        for transform in self.transforms:
            xs = transform.apply(xs)
        return xs

    def extract_sync(self):
        dataset_meta = dataset_repo.get_dataset_meta(dataset_id=self.dataset_id, variable_id=self.variable_id)
        band_range = self.get_band_range_to_extract(dataset_meta.time_range)
        with rasterio.Env():
            with rasterio.open(dataset_meta.p) as ds:
                xs = self.extract_slice(ds, band_range=band_range)
        xs = self.transform_series(xs)
        yr = self.get_time_range_after_transforms(dataset_meta.time_range, band_range)
        values = [None if math.isnan(x) else x for x in xs.tolist()]
        return {'time_range': yr, 'values': values}

    async def extract(self):
        start_time = time.time()
        try:
            # may want to do something like
            # https://github.com/mapbox/rasterio/blob/master/examples/async-rasterio.py
            # to reduce request time
            loop = asyncio.get_event_loop()
            future = loop.run_in_executor(None, self.extract_sync)
            return await asyncio.wait_for(future, timeout=self.max_processing_time, loop=loop)
        except asyncio.TimeoutError as e:
            process_time = time.time() - start_time
            raise TimeseriesTimeoutError(
                message='Request processing time exceeded limit',
                processing_time=process_time
            ) from e


class OptionalYearRange(BaseModel):
    gte: Optional[int]
    lte: Optional[int]

    def normalize(self, time_range_available: YearRange) -> YearRange:
        return YearRange(
            gte=self.gte if self.gte is not None else time_range_available.gte,
            lte=self.lte if self.lte is not None else time_range_available.lte
        )

    class Config:
        schema_extra = {
            "example": YearRange.Config.schema_extra["example"]
        }


class OptionalYearMonthRange(BaseModel):
    gte: Optional[YearMonth]
    lte: Optional[YearMonth]

    def normalize(self, time_range_available: YearMonthRange) -> YearMonthRange:
        return YearMonthRange(
            gte=self.gte if self.gte is not None else time_range_available.gte,
            lte=self.lte if self.lte is not None else time_range_available.lte
        )

    class Config:
        schema_extra = {
            "example": YearMonthRange.Config.schema_extra["example"]
        }


class MonthAnalysisQuery(BaseAnalysisQuery):
    time_range: OptionalYearMonthRange
    transforms: List[Union[MovingAverageSmoother, ZScoreRoller]]

    class Config:
        schema_extra = {
            "example": {
                "dataset_id": "monthly_5x5x60_dataset",
                "variable_id": "float32_variable",
                "time_range": OptionalYearMonthRange.Config.schema_extra['example'],
                "selected_area": Point.Config.schema_extra['example'],
                "zonal_statistic": ZonalStatistic.mean.value,
                "transforms": [
                    MovingAverageSmoother.Config.schema_extra['example']
                ]
            }
        }


class YearAnalysisQuery(BaseAnalysisQuery):
    time_range: OptionalYearRange
    transforms: List[Union[MovingAverageSmoother, ZScoreRoller, ZScoreScaler]]


class YearAnalysisResponse(BaseModel):
    time_range: YearRange
    values: List[float]


class MonthAnalysisResponse(BaseModel):
    time_range: YearMonthRange
    values: List[float]


class TimeseriesV1Request(BaseModel):
    datasetId: str
    variableName: str
    boundaryGeometry: Union[Point, Polygon]
    start: Optional[str]
    end: Optional[str]
    timeout: int = settings.max_processing_time

    def try_to_year_range(self, available_time_range: YearRange) -> YearRange:
        return OptionalYearRange(gte=self.start, lte=self.end).normalize(available_time_range)

    def try_to_year_month_range(self, available_time_range: YearMonthRange) -> YearMonthRange:
        gte_year, gte_month = self.start.split('-', 1)
        lte_year, lte_month = self.end.split('-', 1)
        return OptionalYearMonthRange(
            gte=YearMonth(year=gte_year, month=gte_month),
            lte=YearMonth(year=lte_year, month=lte_month)
        ).normalize(available_time_range)

    def to_year_month_str(self, ym: YearMonth) -> str:
        return f'{ym.year:04}-{ym.month:02}'

    def to_year_str(self, y: int) -> str:
        return f'{y}'

    async def extract(self):
        dataset_meta = dataset_repo.get_dataset_meta(dataset_id=self.datasetId, variable_id=self.variableName)
        if dataset_meta.resolution == 'year':
            time_range = self.try_to_year_range(dataset_meta.time_range)
            query_cls = YearAnalysisQuery
            start = self.to_year_str(time_range.gte)
            end = self.to_year_str(time_range.lte)
        else:
            time_range = self.try_to_year_month_range(dataset_meta.time_range)
            query_cls = MonthAnalysisQuery
            start = self.to_year_month_str(time_range.gte)
            end = self.to_year_month_str(time_range.lte)
        query = query_cls(
            dataset_id=self.datasetId,
            variable_id=self.variableName,
            selected_area=self.boundaryGeometry,
            zonal_statistic=ZonalStatistic.mean,
            time_range=time_range,
            transforms=[],
            max_processing_time=self.timeout
        )
        data = await query.extract()
        return {
            'datasetId': self.datasetId,
            'variableName': self.variableName,
            'boundaryGeometry': self.boundaryGeometry,
            'start': start,
            'end': end,
            'values': data['values']
        }


@router.post("/datasets/monthly", response_model=MonthAnalysisResponse, operation_id='retrieveMonthlyTimeseries')
async def extract_monthly_timeseries(data: MonthAnalysisQuery) -> MonthAnalysisResponse:
    return MonthAnalysisResponse(**await data.extract())


@router.post("/datasets/yearly", response_model=YearAnalysisResponse, operation_id='retrieveYearlyTimeseries')
async def extract_yearly_timeseries(data: YearAnalysisQuery) -> YearAnalysisResponse:
    return YearAnalysisResponse(**await data.extract())


@router.post('/timeseries-service/api/v1/timeseries')
async def timeseries_v1(data: TimeseriesV1Request):
    return await data.extract()
