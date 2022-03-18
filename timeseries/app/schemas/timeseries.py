from concurrent.futures.thread import ThreadPoolExecutor
from datetime import datetime, date
from enum import Enum
from geojson_pydantic import (Feature, FeatureCollection, geometries as geompyd)
from pydantic import BaseModel, Field, validator
from scipy import stats
from shapely import geometry as geom
from typing import Sequence, Optional, Union, Literal, List

from .common import ZonalStatistic, BandRange, TimeRange, OptionalTimeRange
from .dataset import VariableMetadata, get_dataset_manager
from .geometry import Point, Polygon

from app.exceptions import TimeseriesTimeoutError
from app.settings import settings


import asyncio
import math
import numba
import numpy as np
import pandas as pd
import rasterio
import time

import logging

logger = logging.getLogger(__name__)


@numba.jit(nopython=True, nogil=True)
def rolling_z_score(xs, width):
    n = len(xs) - width
    results = np.zeros(n)
    for i in numba.prange(n):
        results[i] = (xs[i + width] - np.nanmean(xs[i:(i + width)])) / np.nanstd(xs[i:(i + width)])
    return results


def values_to_period_range_series(name: str, values: np.array, time_range: TimeRange) -> pd.Series:
    """
    Converts a numpy array and TimeRange into a pandas series
    """
    # use periods instead of end to avoid an off-by-one
    # between the number of values and the generated index
    return pd.Series(values, name=name, index=pd.period_range(start=time_range.gte, periods=len(values), freq='A'))


class WindowType(str, Enum):
    centered = 'centered'
    trailing = 'trailing'

    def get_time_range_required(self, br: BandRange, width: int):
        if self == self.centered:
            return BandRange(gte=br.gte - width, lte=br.lte + width)
        else:
            return BandRange(gte=br.gte - width, lte=br.lte)


class NoSmoother(BaseModel):
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


class MovingAverageSmoother(BaseModel):
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
        window_size = self.width
        return np.convolve(xs, np.ones(window_size) / window_size, 'valid')

    class Config:
        schema_extra = {
            "example": {
                "type": "MovingAverageSmoother",
                "method": WindowType.centered.value,
                "width": 1
            }
        }


Smoother = Union[NoSmoother, MovingAverageSmoother]
    

class ZScoreMovingInterval(BaseModel):
    """A moving Z-Score transform to the timeseries"""
    type: Literal['ZScoreMovingInterval'] = 'ZScoreMovingInterval'
    width: int = Field(..., description='number of prior years (or months) to use in the moving window', ge=0, le=200)

    def get_desired_band_range(self, dataset_variable_metadata: VariableMetadata) -> Optional[BandRange]:
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

    def get_desired_band_range(self, metadata: VariableMetadata) -> Optional[BandRange]:
        return metadata.find_band_range(self.time_range) if self.time_range else None

    def get_desired_band_range_adjustment(self):
        return np.array([0, 0])

    def apply(self, xs, txs):
        if self.time_range is None:
            return stats.zscore(xs, nan_policy='omit')
        else:
            mean_txs = np.nanmean(txs)
            std_txs = np.nanstd(txs)
            return (xs - mean_txs) / std_txs

    class Config:
        schema_extra = {
            'example': {
                'type': 'ZScoreFixedInterval'
            }
        }


class NoTransform(BaseModel):
    """A no-op transform to the timeseries"""
    type: Literal['NoTransform'] = 'NoTransform'

    def get_desired_band_range(self, metadata: VariableMetadata) -> Optional[BandRange]:
        return None

    def get_desired_band_range_adjustment(self):
        return np.array([0, 0])

    def apply(self, xs, txs):
        return xs


Transform = Union[ZScoreMovingInterval, ZScoreFixedInterval, NoTransform]


class SummaryStat(BaseModel):
    name: str
    mean: Optional[float]
    median: Optional[float]
    stdev: Optional[float]


class SeriesOptions(BaseModel):
    name: str
    smoother: Smoother

    def get_desired_band_range_adjustment(self):
        return self.smoother.get_desired_band_range_adjustment()

    def apply(self, xs: np.array, time_range: TimeRange) -> pd.Series:
        values = self.smoother.apply(xs)
        return values_to_period_range_series(self.name, values, time_range)

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
        return SummaryStat(
            name=name,
            mean=xs_mean,
            median=xs_median,
            stdev=xs_stdev
        )

    def to_summary_stat(self):
        return self.get_summary_stats(xs=self._s.to_numpy(), name=self.options.name)



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
    """
    FIXME: refactor to decouple extraction logic from the query data class
    """
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

    def to_time_range(self, metadata: VariableMetadata) -> TimeRange:
        """
        converts start / end string inputs incoming from the request into OptionalTimeRange dates
        1 -> 0001-01-01
        4 -> 0004-01-01
        '0001' -> 0001-01-01
        '2000-01' -> '2000-01-01'
        '2000-04-03' -> '2000-04-03'
        :param metadata:
        :return:
        """
        if self.start is None:
            gte = metadata.time_range.gte
        else:
            split_start = self.start.split('-', 1)
            if len(split_start) == 1:
                gte = self._to_date_from_y(split_start[0])
            elif len(split_start) == 2:
                gte = self._to_date_from_ym(split_start[0], split_start[1])

        if self.end is None:
            lte = metadata.time_range.lte
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
        return metadata.normalize_time_range(otr)

    async def extract(self, metadata):
        """
        metadata = get_dataset_manager().get_dataset_variable_meta(
            dataset_id=self.datasetId,
            variable_id=self.variableName
        )
        """
        time_range = self.to_time_range(metadata)
        start = time_range.gte.isoformat()
        end = time_range.lte.isoformat()

        # delegate to TimeseriesV2Request
        query = TimeseriesRequest(
            resolution=metadata.resolution,
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


class TimeseriesRequest(BaseModel):
    dataset_id: str = Field(..., regex=r'^[\w-]+$', description='Dataset ID')
    variable_id: str = Field(..., regex=r'^[\w-]+$', description='Variable ID (unique to a particular dataset)')
    selected_area: Union[Point, Polygon, Feature, FeatureCollection]
    zonal_statistic: ZonalStatistic
    max_processing_time: int = Field(settings.max_processing_time, ge=0, le=settings.max_processing_time)
    transform: Transform
    requested_series: List[SeriesOptions]
    time_range: OptionalTimeRange

    def transforms(self, series_options: SeriesOptions):
        return [self.transform, series_options]

    def extract_slice(self, dataset: rasterio.DatasetReader, band_range: Sequence[int]):
        # FIXME: do things with the selected_area instead of giving it behavior
        return self.selected_area.extract(dataset, self.zonal_statistic, band_range=band_range)

    def get_band_ranges_for_transform(self, metadata: VariableMetadata) -> BandRange:
        """Get the band range range to extract from the raster file"""
        br_avail = metadata.find_band_range(metadata.time_range)
        br_query = self.transform.get_desired_band_range(metadata)
        compromise_br = br_avail.intersect(br_query) if br_query else None
        logger.debug("dataset band range %s, desired band range %s, final band range %s", br_avail, br_query, compromise_br)
        return compromise_br

    def get_band_range_to_extract(self, metadata: VariableMetadata) -> BandRange:
        """Get the band range range to extract from the raster file"""
        br_avail = metadata.find_band_range(metadata.time_range)
        br_query = metadata.find_band_range(self.time_range)
        transform_br = br_query + self.transform.get_desired_band_range_adjustment()
        desired_br = transform_br
        for series in self.requested_series:
            candidate_br = transform_br + series.get_desired_band_range_adjustment()
            desired_br = desired_br.union(candidate_br)
            logger.info('transform band range %s adjusted to candidate band range %s, resulting in desired br: %s', transform_br, candidate_br, desired_br)

        compromise_br = br_avail.intersect(
            BandRange.from_numpy_pair(desired_br))
        logger.info('final compromise_br, %s', compromise_br)
        return compromise_br

    def get_time_range_after_transforms(self, series_options: SeriesOptions, metadata: VariableMetadata, extract_br: BandRange) -> TimeRange:
        """Get the year range after values after applying transformations"""
        inds = extract_br + \
            self.transform.get_desired_band_range_adjustment() * -1 + \
            series_options.get_desired_band_range_adjustment() * -1
        print(f'inds = {inds}')
        yr = metadata.translate_band_range(BandRange.from_numpy_pair(inds))
        return yr

    def apply_series(self, xs, metadata, band_range):
        series_list = []
        pd_series_list = []
        gte = datetime.fromordinal(self.time_range.gte.toordinal())
        lte = datetime.fromordinal(self.time_range.lte.toordinal())
        for series_options in self.requested_series:
            tr = self.get_time_range_after_transforms(series_options, metadata, band_range)
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
        # Computes summary statistics over requested timeseries band ranges
        summary_stats = [Series.get_summary_stats(s, s.name) for s in series]
        if not isinstance(self.transform, NoTransform):
            # provide original summary stats for z-scores over the original
            # band range, not the adjusted one
            summary_stats.insert(0, Series.get_summary_stats(xs, 'Original'))
        return summary_stats

    def extract_sync(self):
        metadata = get_dataset_manager().get_variable_metadata(dataset_id=self.dataset_id, variable_id=self.variable_id)
        band_range = self.get_band_range_to_extract(metadata)
        band_range_transform = self.get_band_ranges_for_transform(metadata)
        logger.debug("extract band range %s, transform band range: %s", band_range, band_range_transform)
        with rasterio.Env():
            with rasterio.open(metadata.path) as ds:
                data_slice = self.extract_slice(ds, band_range=band_range)
                xs = data_slice['data']
                n_cells = data_slice['n_cells']
                area = data_slice['area']
                transform_xs = self.extract_slice(ds, band_range=band_range_transform)['data'] if band_range_transform else None

        txs = self.transform.apply(xs, transform_xs)

        series, pd_series = self.apply_series(
            txs,
            metadata=metadata,
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