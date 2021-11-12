import asyncio
import logging
import time
from concurrent.futures.thread import ThreadPoolExecutor

import numba
import numpy as np
import math
import pandas as pd
import pyproj
import rasterio

from enum import Enum

from datetime import datetime
from fastapi import APIRouter
from geojson_pydantic import geometries as geompyd
from numba import prange
from pydantic import BaseModel, Field, validator
from rasterio.features import shapes
from rasterio.mask import raster_geometry_mask
from rasterio.windows import Window
from scipy import stats
from shapely import geometry as geom
from shapely.ops import orient
from typing import List, Optional, Union, Literal, Sequence
from datetime import date

from shapely.validation import explain_validity

from app.exceptions import (SelectedAreaOutOfBoundsError, SelectedAreaPolygonIsNotValid, TimeseriesTimeoutError, SelectedAreaPolygonIsTooLarge)
from app.settings import settings
from app.stores import dataset_repo, BandRange

from app.stores import DatasetVariableMeta, TimeRange, OptionalTimeRange

logger = logging.getLogger(__name__)
router = APIRouter(tags=['datasets'], prefix='/timeseries-service/api')


class ZonalStatistic(str, Enum):
    mean = 'mean'
    median = 'median'

    @staticmethod
    def _median(xs, axis, dtype):
        return np.median(xs, axis=axis)

    def to_numpy_call(self):
        if self == self.mean:
            return np.mean
        elif self == self.median:
            return self._median


def bounding_box(bounds) -> geom.Polygon:
    return geom.box(
        minx=bounds.left,
        miny=bounds.bottom,
        maxx=bounds.right,
        maxy=bounds.top)


class Point(geompyd.Point):

    @staticmethod
    def calculate_area(px: int, py: int, dataset: rasterio.DatasetReader):
        wgs84 = pyproj.Geod(ellps='WGS84')
        top_left = dataset.xy(row=py, col=px)
        bottom_right = dataset.xy(row=py + 1, col=px + 1)
        top_right = (bottom_right[0], top_left[1])
        bottom_left = (top_left[0], bottom_right[1])
        bbox = geom.Polygon([top_left, bottom_left, bottom_right, top_right, top_left])
        area, perimeter = wgs84.geometry_area_perimeter(bbox)
        return abs(area)

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
        data = dataset.read(list(band_range), window=Window(px, py, 1, 1), out_dtype=np.float64).flatten()
        data[np.equal(data, dataset.nodata)] = np.nan
        area = self.calculate_area(px=px, py=py, dataset=dataset)
        return {
            'n_cells': 1,
            'area': area,
            'data': data,
        }

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
    @staticmethod
    def _make_band_range_groups(*, width: int, height: int, band_range: BandRange, max_size=250000):
        n_cells_per_band = width * height  # 25
        n_cells_per_full_chunk = max_size - max_size % n_cells_per_band
        if n_cells_per_full_chunk == 0:
            raise SelectedAreaPolygonIsTooLarge(n_cells=n_cells_per_band, max_cells=max_size)
        n_bands = len(band_range)
        n = n_cells_per_band * n_bands  # 650
        n_full_chunks = (n // n_cells_per_full_chunk)  # 650 // 625 = 1
        n_bands_per_full_chunk = n_cells_per_full_chunk // n_cells_per_band
        offset = band_range.gte
        for i in range(n_full_chunks):
            band_indices = range(i*n_bands_per_full_chunk + offset, (i+1)*n_bands_per_full_chunk + offset)
            yield band_indices
        n_last_bands = n_bands % (n_cells_per_full_chunk // n_cells_per_band)  # 26 % (625 // 25) = 26 % 25 = 1
        if n_last_bands > 0:
            yield range(n_bands - n_last_bands + offset, n_bands + offset)

    @staticmethod
    def calculate_area(masked, transform):
        shape_iter = shapes(masked.astype('uint8'), mask=np.equal(masked, 0), transform=transform)
        area = 0.0
        wgs84 = pyproj.Geod(ellps='WGS84')
        # area is signed positive or negative based on clockwise or
        # counterclockwise traversal:
        # https://pyproj4.github.io/pyproj/stable/api/geod.html?highlight=counter%20clockwise#pyproj.Geod.geometry_area_perimeter
        for shp, val in shape_iter:
            shp = orient(shp)
            shp = geom.shape(shp)
            area += wgs84.geometry_area_perimeter(shp)[0]
        return abs(area)

    def extract(self,
                dataset: rasterio.DatasetReader,
                zonal_statistic: ZonalStatistic,
                band_range: BandRange):
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
        n_cells = masked.size - np.count_nonzero(masked)
        area = self.calculate_area(masked, transform=transform)
        result = np.zeros(len(band_range), dtype=np.float64)
        offset = -band_range.gte
        for band_group in self._make_band_range_groups(width=window.width, height=window.height, band_range=band_range):
            data = dataset.read(list(band_group), window=window)
            values = np.ma.array(data=data, mask=np.logical_or(np.equal(data, dataset.nodata), masked))
            lb = band_group.start + offset
            ub = band_group.stop + offset
            r = zonal_func(values, axis=(1, 2), dtype=np.float64)
            result[lb:ub] = r
        return {'n_cells': n_cells, 'area': area, 'data': result}


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
        return width


class MovingAverageSmoother(Smoother):
    type: Literal['MovingAverageSmoother'] = 'MovingAverageSmoother'
    method: WindowType
    width: int = Field(
        ...,
        description="number of years (or months) from current time to use in the moving window",
        ge=1,
        le=200
    )

    @validator('width')
    def width_is_valid_for_window_type(cls, value, values):
        if 'method' not in values:
            return value
        method = values['method']
        if method == WindowType.centered and value % 2 == 0:
            raise ValueError('window width must be odd for centered windows')
        return value

    def get_desired_band_range_adjustment(self):
        logger.info(f'width = {self.width}')
        band_range_adjustment = []
        if self.method == WindowType.centered:
            band_range_adjustment = np.array([-(self.width // 2), self.width // 2])
        else:
            band_range_adjustment = np.array([-self.width, 0])
        logger.debug("smoother band range adjustment: %s", band_range_adjustment)
        return band_range_adjustment

    def apply(self, xs: np.array) -> np.array:
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


class NoSmoother(Smoother):
    type: Literal['NoSmoother'] = 'NoSmoother'

    def apply(self, xs: np.array) -> np.array:
        return xs

    def get_desired_band_range_adjustment(self):
        return np.array([0, 0])

    class Config:
        schema_extra = {
            "example": {
                "type": "NoSmoother",
            }
        }


def values_to_period_range(name: str, values: np.array, time_range: TimeRange) -> pd.Series:
    # FIXME: use periods instead of end to avoid an off-by-one
    # between the number of values and the generated index
    return pd.Series(values, name=name, index=pd.period_range(start=time_range.gte, periods=len(values), freq='A'))


class SeriesOptions(BaseModel):
    name: str
    smoother: Union[MovingAverageSmoother, NoSmoother]

    def get_desired_band_range_adjustment(self):
        return self.smoother.get_desired_band_range_adjustment()

    def apply(self, xs: np.array, time_range: TimeRange) -> pd.Series:
        values = self.smoother.apply(xs)
        return values_to_period_range(self.name, values, time_range)

    class Config:
        schema_extra = {
            "example": {
                "name": "transformed",
                "smoother": MovingAverageSmoother.Config.schema_extra['example']
            }
        }


class Series(BaseModel):
    options: SeriesOptions
    time_range: TimeRange
    values: List[Optional[float]]

    @classmethod
    def summary_stat(cls, f, xs):
        """
        Summarize a series

        :param f: a numpy nan removing function like np.nanmean etc
        :param xs: a numpy array
        :return: the summary statistic in json serializable form (nans are replaced with None
        in the case where `xs` is all nan elements)
        """
        stat = f(xs)
        stat = None if math.isnan(stat) else stat
        return stat

    @classmethod
    def get_summary_stats(cls, xs, name):
        xs_mean = cls.summary_stat(np.nanmean, xs)
        xs_median = cls.summary_stat(np.nanmedian, xs)
        xs_stdev = cls.summary_stat(np.nanstd, xs)
        ss = SummaryStat(
            name=name,
            mean=xs_mean,
            median=xs_median,
            stdev=xs_stdev
        )
        return ss

    def to_summary_stat(self):
        return self.get_summary_stats(xs=self._s.to_numpy(), name=self.options.name)


@numba.jit(nopython=True, nogil=True)
def rolling_z_score(xs, width):
    n = len(xs) - width
    results = np.zeros(n)
    for i in prange(n):
        results[i] = (xs[i + width] - np.mean(xs[i:(i + width)])) / np.std(xs[i:(i + width)])
    return results


class NoTransform(BaseModel):
    """A no-op transform to the timeseries"""
    type: Literal['NoTransform'] = 'NoTransform'

    def get_desired_band_range(self, dataset_meta: DatasetVariableMeta) -> Optional[BandRange]:
        return None

    def get_desired_band_range_adjustment(self):
        return np.array([0, 0])

    def apply(self, xs, txs):
        return xs


class ZScoreMovingInterval(BaseModel):
    """A moving Z-Score transform to the timeseries"""
    type: Literal['ZScoreMovingInterval'] = 'ZScoreMovingInterval'
    width: int = Field(..., description='number of prior years (or months) to use in the moving window', ge=0, le=200)

    def get_desired_band_range(self, dataset_meta: DatasetVariableMeta) -> Optional[BandRange]:
        return None

    def get_desired_band_range_adjustment(self):
        return np.array([-self.width, 0])

    def apply(self, xs, txs):
        return rolling_z_score(xs, self.width)

    class Config:
        schema_extra = {
            'example': {
                'type': 'ZScoreMovingInterval',
                'width': 5
            }
        }


class ZScoreFixedInterval(BaseModel):
    type: Literal['ZScoreFixedInterval'] = 'ZScoreFixedInterval'
    time_range: Optional[TimeRange]

    def get_desired_band_range(self, dataset_meta: DatasetVariableMeta) -> Optional[BandRange]:
        return dataset_meta.find_band_range(self.time_range) if self.time_range else None

    def get_desired_band_range_adjustment(self):
        return np.array([0, 0])

    def apply(self, xs, txs):
        if self.time_range is None:
            return stats.zscore(xs)
        else:
            mean_txs = np.mean(txs)
            std_txs = np.std(txs)
            return (xs - mean_txs)/std_txs

    class Config:
        schema_extra = {
            'example': {
                'type': 'ZScoreFixedInterval'
            }
        }


class TimeseriesQuery(BaseModel):
    dataset_id: str = Field(..., regex=r'^[\w-]+$', description='Dataset ID')
    variable_id: str = Field(..., regex=r'^[\w-]+$', description='Variable ID (unique to a particular dataset)')
    selected_area: Union[Point, Polygon]
    zonal_statistic: ZonalStatistic
    max_processing_time: int = Field(settings.max_processing_time, ge=0, le=settings.max_processing_time)
    transform: Union[ZScoreMovingInterval, ZScoreFixedInterval, NoTransform]
    requested_series: List[SeriesOptions]
    time_range: OptionalTimeRange

    def transforms(self, series_options: SeriesOptions):
        return [self.transform, series_options]

    def extract_slice(self, dataset: rasterio.DatasetReader, band_range: Sequence[int]):
        return self.selected_area.extract(dataset, self.zonal_statistic, band_range=band_range)

    def get_band_ranges_for_transform(self, dataset_meta: DatasetVariableMeta) -> BandRange:
        """Get the band range range to extract from the raster file"""
        br_avail = dataset_meta.find_band_range(dataset_meta.time_range)
        br_query = self.transform.get_desired_band_range(dataset_meta)
        compromise_br = br_avail.intersect(br_query) if br_query else None
        return compromise_br

    def get_band_range_to_extract(self, dataset_meta: DatasetVariableMeta) -> BandRange:
        """Get the band range range to extract from the raster file"""
        br_avail = dataset_meta.find_band_range(dataset_meta.time_range)
        br_query = dataset_meta.find_band_range(self.time_range)
        transform_br = br_query + self.transform.get_desired_band_range_adjustment()
        desired_br = transform_br
        for series in self.requested_series:
            candidate_br = transform_br + series.get_desired_band_range_adjustment()
            logger.info(f'candidate_br = {candidate_br}')
            desired_br = desired_br.union(candidate_br)

        compromise_br = br_avail.intersect(
            BandRange.from_numpy_pair(desired_br))
        logger.info(f'compromise_br = {compromise_br}')
        return compromise_br

    def get_time_range_after_transforms(self, series_options: SeriesOptions, dataset_meta: DatasetVariableMeta, extract_br: BandRange) -> TimeRange:
        """Get the year range after values after applying transformations"""
        inds = extract_br + \
            self.transform.get_desired_band_range_adjustment() * -1 + \
            series_options.get_desired_band_range_adjustment() * -1
        print(f'inds = {inds}')
        yr = dataset_meta.translate_band_range(
            BandRange.from_numpy_pair(inds))
        return yr

    def apply_series(self, xs, dataset_meta, band_range):
        series_list = []
        pd_series_list = []
        gte = datetime.fromordinal(self.time_range.gte.toordinal())
        lte = datetime.fromordinal(self.time_range.lte.toordinal())
        for series_options in self.requested_series:
            tr = self.get_time_range_after_transforms(series_options, dataset_meta, band_range)
            pd_series = series_options.apply(xs, tr).loc[gte:lte]
            pd_series_list.append(pd_series)
            compromise_tr = tr.intersect(self.time_range)
            values = [None if math.isnan(x) else x for x in pd_series.tolist()]
            series = Series(
                options=series_options,
                time_range=compromise_tr,
                values=values,
            )
            series_list.append(series)
        return (series_list, pd_series_list)

    def get_summary_stats(self, series, xs):
        summary_stats = [Series.get_summary_stats(s, s.name) for s in series]
        if not isinstance(self.transform, NoTransform):
            # provide original summary stats for z-scores
            summary_stats.insert(0, Series.get_summary_stats(xs, 'Original'))
        return summary_stats

    def extract_sync(self):
        dataset_meta = dataset_repo.get_dataset_variable_meta(
            dataset_id=self.dataset_id,
            variable_id=self.variable_id,
        )
        band_range = self.get_band_range_to_extract(dataset_meta)
        band_range_transform = self.get_band_ranges_for_transform(dataset_meta)
        with rasterio.Env():
            with rasterio.open(dataset_meta.p) as ds:
                res = self.extract_slice(ds, band_range=band_range)
                xs = res['data']
                n_cells = res['n_cells']
                area = res['area']
                transform_xs = self.extract_slice(ds, band_range=band_range_transform)['data'] if band_range_transform else None
        txs = self.transform.apply(xs, transform_xs)

        series, pd_series = self.apply_series(
            txs,
            dataset_meta=dataset_meta,
            band_range=band_range
        )
        return TimeseriesResponse(
            dataset_id=self.dataset_id,
            variable_id=self.variable_id,
            area=area,
            n_cells=n_cells,
            series=series,
            transform=self.transform,
            zonal_statistic=self.zonal_statistic,
            summary_stats=self.get_summary_stats(pd_series, xs),
        )

    async def extract(self):
        start_time = time.time()
        try:
            # may want to do something like
            # https://github.com/mapbox/rasterio/blob/master/examples/async-rasterio.py
            # to reduce request time
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as pool:
                future = loop.run_in_executor(pool, self.extract_sync)
                return await asyncio.wait_for(future, timeout=self.max_processing_time)
        except asyncio.TimeoutError as e:
            process_time = time.time() - start_time
            raise TimeseriesTimeoutError(
                message='Request processing time exceeded limit',
                processing_time=process_time
            ) from e

    class Config:
        schema_extra = {
            "moving_interval_example": {
                "resolution": "month",
                "dataset_id": "monthly_5x5x60_dataset",
                "variable_id": "float32_variable",
                "time_range": OptionalTimeRange.Config.schema_extra['example'],
                "selected_area": Point.Config.schema_extra['example'],
                "zonal_statistic": ZonalStatistic.mean.value,
                "transform": ZScoreMovingInterval.Config.schema_extra['example'],
                "requested_series": [SeriesOptions.Config.schema_extra['example']]
            },
            "fixed_interval_example": {
                "resolution": "month",
                "dataset_id": "monthly_5x5x60_dataset",
                "variable_id": "float32_variable",
                "time_range": OptionalTimeRange.Config.schema_extra['example'],
                "selected_area": Point.Config.schema_extra['example'],
                "zonal_statistic": ZonalStatistic.mean.value,
                "transform": ZScoreFixedInterval.Config.schema_extra['example'],
                "requested_series": [SeriesOptions.Config.schema_extra['example']]
            }
        }


Transform = Union[ZScoreMovingInterval, ZScoreFixedInterval, NoTransform]


class SummaryStat(BaseModel):
    name: str
    mean: Optional[float]
    median: Optional[float]
    stdev: Optional[float]

    @classmethod
    def from_series(cls, series: List[Series]) -> List['SummaryStat']:
        summary_statistics = []
        for s in series:
            summary_statistics.append(cls(
                name=s.options.name,
                mean=s._s.mean(),
                median=s._s.median(),
                stdev=s._s.std()
            ))
        return summary_statistics


class TimeseriesResponse(BaseModel):
    dataset_id: str
    variable_id: str
    area: float = Field(..., description='area of cells in selected area in square meters')
    n_cells: int = Field(..., description='number of cells in selected area')
    summary_stats: List[SummaryStat]
    series: List[Series]
    transform: Transform
    zonal_statistic: ZonalStatistic


class TimeseriesV1Request(BaseModel):
    datasetId: str
    variableName: str
    boundaryGeometry: Union[Point, Polygon]
    start: Optional[str]
    end: Optional[str]
    timeout: int = settings.max_processing_time

    def _to_date_from_y(self, year) -> date:
        return date(year=int(year), month=1, day=1)

    def _to_date_from_ym(self, year, month) -> date:
        return date(year=int(year), month=int(month), day=1)

    def to_time_range(self, dataset_meta: DatasetVariableMeta) -> TimeRange:
        """
        converts start / end string inputs incoming from the request into OptionalTimeRange dates
        1 -> 0001-01-01
        4 -> 0004-01-01
        '0001' -> 0001-01-01
        '2000-01' -> '2000-01-01'
        '2000-04-03' -> '2000-04-03'
        :param dataset_meta:
        :return:
        """
        if self.start is None:
            gte = dataset_meta.time_range.gte
        else:
            split_start = self.start.split('-', 1)
            if len(split_start) == 1:
                gte = self._to_date_from_y(split_start[0])
            elif len(split_start) == 2:
                gte = self._to_date_from_ym(split_start[0], split_start[1])

        if self.end is None:
            lte = dataset_meta.time_range.lte
        else:
            split_end = self.end.split('-', 1)
            if len(split_end) == 1:
                lte = self._to_date_from_y(split_end[0])
            elif len(split_end) == 2:
                lte = self._to_date_from_ym(split_end[0], split_end[1])

        otr = OptionalTimeRange(
            gte=gte,
            lte=lte
        )
        return dataset_meta.normalize_time_range(otr)

    async def extract(self):
        dataset_meta = dataset_repo.get_dataset_variable_meta(
            dataset_id=self.datasetId,
            variable_id=self.variableName
        )
        time_range = self.to_time_range(dataset_meta)
        start = time_range.gte.isoformat()
        end = time_range.lte.isoformat()

        query = TimeseriesQuery(
            resolution=dataset_meta.resolution,
            dataset_id=self.datasetId,
            variable_id=self.variableName,
            selected_area=self.boundaryGeometry,
            zonal_statistic=ZonalStatistic.mean,
            time_range=time_range,
            transform=NoTransform(),
            requested_series=[
                SeriesOptions(
                    name='original',
                    smoother=NoSmoother()
                )
            ],
            max_processing_time=self.timeout
        )
        data = await query.extract()
        return {
            'datasetId': self.datasetId,
            'variableName': self.variableName,
            'boundaryGeometry': self.boundaryGeometry,
            'start': start,
            'end': end,
            'values': data.series[0].values
        }


@router.post(
    "/v2/timeseries",
    response_model=TimeseriesResponse,
    operation_id='retrieveTimeseries')
async def extract_timeseries(data: TimeseriesQuery) -> TimeseriesResponse:
    """ Retrieve dataset analysis """
    return await data.extract()


@router.post('/v1/timeseries')
async def timeseries_v1(data: TimeseriesV1Request):
    return await data.extract()
