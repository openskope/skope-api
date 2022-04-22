from datetime import datetime, date
from enum import Enum
from geojson_pydantic import (
    Point,
    Polygon,
)
from pydantic import BaseModel, Field, validator
from scipy import stats
from typing import Sequence, Optional, Union, Literal, List

import logging
import math
import numba
import numpy as np
import pandas as pd
import rasterio

from .common import ZonalStatistic, BandRange, TimeRange, OptionalTimeRange
from .dataset import VariableMetadata, DatasetManager
from .geometry import (
    SkopeFeatureCollectionModel,
    SkopeFeatureModel,
    SkopePointModel,
    SkopePolygonModel,
)

from app.config import get_settings


settings = get_settings()
logger = logging.getLogger(__name__)


@numba.jit(nopython=True, nogil=True)
def rolling_z_score(xs, width):
    n = len(xs) - width
    results = np.zeros(n)
    for i in numba.prange(n):
        results[i] = (xs[i + width] - np.nanmean(xs[i : (i + width)])) / np.nanstd(
            xs[i : (i + width)]
        )
    return results


class WindowType(str, Enum):
    centered = "centered"
    trailing = "trailing"

    def get_time_range_required(self, br: BandRange, width: int):
        if self == self.centered:
            return BandRange(gte=br.gte - width, lte=br.lte + width)
        else:
            return BandRange(gte=br.gte - width, lte=br.lte)


class NoSmoother(BaseModel):
    type: Literal["NoSmoother"] = "NoSmoother"

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
    type: Literal["MovingAverageSmoother"] = "MovingAverageSmoother"
    method: WindowType
    width: int = Field(
        ...,
        description="number of years (or months) from current time to use in the moving window",
        ge=1,
        le=200,
    )

    @validator("width")
    def width_is_valid_for_window_type(cls, value, values):
        if "method" not in values:
            return value
        method = values["method"]
        if method == WindowType.centered and value % 2 == 0:
            raise ValueError("window width must be odd for centered windows")
        return value

    def get_desired_band_range_adjustment(self):
        logger.info(f"width = {self.width}")
        if self.method == WindowType.centered:
            offset = self.width // 2
            band_range_adjustment = np.array([-offset, offset])
        else:
            band_range_adjustment = np.array([-self.width, 0])
        logger.debug("smoother band range adjustment: %s", band_range_adjustment)
        return band_range_adjustment

    def apply(self, xs: np.array) -> np.array:
        window_size = self.width
        return np.convolve(xs, np.ones(window_size) / window_size, "valid")

    class Config:
        schema_extra = {
            "example": {
                "type": "MovingAverageSmoother",
                "method": WindowType.centered.value,
                "width": 1,
            }
        }


Smoother = Union[NoSmoother, MovingAverageSmoother]


class ZScoreMovingInterval(BaseModel):
    """A moving Z-Score transform to the timeseries"""

    type: Literal["ZScoreMovingInterval"] = "ZScoreMovingInterval"
    width: int = Field(
        ...,
        description="number of prior years (or months) to use in the moving window",
        ge=0,
        le=200,
    )

    def get_desired_band_range(
        self, dataset_variable_metadata: VariableMetadata
    ) -> Optional[BandRange]:
        return None

    def get_desired_band_range_adjustment(self):
        return np.array([-self.width, 0])

    def apply(self, xs, txs):
        return rolling_z_score(xs, self.width)

    class Config:
        schema_extra = {"example": {"type": "ZScoreMovingInterval", "width": 5}}


class ZScoreFixedInterval(BaseModel):
    type: Literal["ZScoreFixedInterval"] = "ZScoreFixedInterval"
    time_range: Optional[TimeRange]

    def get_desired_band_range(self, metadata: VariableMetadata) -> Optional[BandRange]:
        return metadata.find_band_range(self.time_range) if self.time_range else None

    def get_desired_band_range_adjustment(self):
        return np.array([0, 0])

    def apply(self, xs, txs):
        if self.time_range is None:
            # z score with respect to the already selected interval reflected in the incoming xs
            return stats.zscore(xs, nan_policy="omit")
        else:
            # z score with respect to a fixed interval
            mean_txs = np.nanmean(txs)
            std_txs = np.nanstd(txs)
            return (xs - mean_txs) / std_txs

    class Config:
        schema_extra = {"example": {"type": "ZScoreFixedInterval"}}


class NoTransform(BaseModel):
    """A no-op transform to the timeseries"""

    type: Literal["NoTransform"] = "NoTransform"

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

    def values_to_period_range_series(
        self, values: np.array, time_range: TimeRange
    ) -> pd.Series:
        """
        Converts a numpy array and TimeRange into a pandas series
        """
        # use periods instead of end to avoid an off-by-one
        # between the number of values and the generated index
        return pd.Series(
            values,
            name=self.name,
            index=pd.period_range(start=time_range.gte, periods=len(values), freq="A"),
        )

    def apply(self, xs: np.array, time_range: TimeRange) -> pd.Series:
        values = self.smoother.apply(xs)
        return self.values_to_period_range_series(values, time_range)

    class Config:
        schema_extra = {
            "example": {
                "name": "transformed",
                "smoother": MovingAverageSmoother.Config.schema_extra["example"],
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
        return None if math.isnan(stat) else stat

    @classmethod
    def to_summary_stat(cls, xs, name):
        xs_mean = cls.summary_stat(np.nanmean, xs)
        xs_median = cls.summary_stat(np.nanmedian, xs)
        xs_stdev = cls.summary_stat(np.nanstd, xs)
        return SummaryStat(name=name, mean=xs_mean, median=xs_median, stdev=xs_stdev)


class TimeseriesResponse(BaseModel):
    dataset_id: str
    variable_id: str
    area: float = Field(
        ..., description="area of cells in selected area in square meters"
    )
    n_cells: int = Field(..., description="number of cells in selected area")
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
            split_start = self.start.split("-", 1)
            if len(split_start) == 1:
                gte = self._to_date_from_y(split_start[0])
            elif len(split_start) == 2:
                gte = self._to_date_from_ym(split_start[0], split_start[1])

        if self.end is None:
            lte = metadata.time_range.lte
        else:
            split_end = self.end.split("-", 1)
            if len(split_end) == 1:
                lte = self._to_date_from_y(split_end[0])
            elif len(split_end) == 2:
                lte = self._to_date_from_ym(split_end[0], split_end[1])

        otr = OptionalTimeRange(gte=gte, lte=lte)
        return metadata.normalize_time_range(otr)

    async def to_timeseries_request(self, metadata):
        time_range = self.to_time_range(metadata)

        # delegate to TimeseriesV2Request
        return TimeseriesRequest(
            resolution=metadata.resolution,
            dataset_id=self.datasetId,
            variable_id=self.variableName,
            selected_area=self.boundaryGeometry,
            zonal_statistic=ZonalStatistic.mean,
            time_range=time_range,
            transform=NoTransform(),
            requested_series_options=[
                SeriesOptions(name="original", smoother=NoSmoother())
            ],
            max_processing_time=self.timeout,
        )


class TimeseriesRequest(BaseModel):
    dataset_id: str = Field(..., regex=r"^[\w-]+$", description="Dataset ID")
    variable_id: str = Field(
        ...,
        regex=r"^[\w-]+$",
        description="Variable ID (unique to a particular dataset)",
    )
    selected_area: Union[
        SkopePointModel,
        SkopePolygonModel,
        SkopeFeatureModel,
        SkopeFeatureCollectionModel,
    ]
    zonal_statistic: ZonalStatistic
    max_processing_time: int = Field(
        settings.max_processing_time, ge=0, le=settings.max_processing_time
    )
    transform: Transform
    requested_series_options: List[SeriesOptions]
    time_range: OptionalTimeRange

    def transforms(self, series_options: SeriesOptions):
        return [self.transform, series_options]

    def apply_transform(self, xs, txs):
        # FIXME: only needs / uses txs in the case of fixed interval z score
        return self.transform.apply(xs, txs)

    def extract_slice(self, dataset: rasterio.DatasetReader, band_range: Sequence[int]):
        return self.selected_area.extract(
            dataset, self.zonal_statistic, band_range=band_range
        )

    def get_variable_metadata(self, dataset_manager: DatasetManager):
        return dataset_manager.get_variable_metadata(
            dataset_id=self.dataset_id, variable_id=self.variable_id
        )

    def get_transform_band_range(self, metadata: VariableMetadata) -> BandRange:
        """Get the band range to extract from the raster file"""
        br_avail = metadata.find_band_range(metadata.time_range)
        br_query = self.transform.get_desired_band_range(metadata)
        compromise_br = br_avail.intersect(br_query) if br_query else None
        logger.debug(
            "dataset band range %s, desired band range %s, final band range %s",
            br_avail,
            br_query,
            compromise_br,
        )
        return compromise_br

    def get_requested_band_range(self, metadata: VariableMetadata) -> BandRange:
        dataset_band_range = metadata.find_band_range(metadata.time_range)
        requested_band_range = metadata.find_band_range(self.time_range)
        return dataset_band_range.intersect(requested_band_range)

    def get_band_range_to_extract(self, metadata: VariableMetadata) -> BandRange:
        """Get the band range range to extract from the raster file"""
        br_avail = metadata.find_band_range(metadata.time_range)
        br_query = metadata.find_band_range(self.time_range)
        transform_br = br_query + self.transform.get_desired_band_range_adjustment()
        desired_br = transform_br
        # requested series options currently manages smoothing options only
        for series in self.requested_series_options:
            candidate_br = transform_br + series.get_desired_band_range_adjustment()
            desired_br = desired_br.union(candidate_br)

        compromise_br = br_avail.intersect(BandRange.from_numpy_pair(desired_br))
        logger.info("final band range to extract, %s", compromise_br)
        return compromise_br

    def get_time_range_after_transforms(
        self,
        series_options: SeriesOptions,
        metadata: VariableMetadata,
        extract_br: BandRange,
    ) -> TimeRange:
        """Get the year range from values after applying transformations"""
        inds = (
            extract_br
            + self.transform.get_desired_band_range_adjustment() * -1
            + series_options.get_desired_band_range_adjustment() * -1
        )
        print(f"inds = {inds}")
        yr = metadata.translate_band_range(BandRange.from_numpy_pair(inds))
        return yr

    def apply_smoothing(self, xs, metadata, band_range):
        series_list = []
        pd_series_list = []
        gte = datetime.fromordinal(self.time_range.gte.toordinal())
        lte = datetime.fromordinal(self.time_range.lte.toordinal())
        band_range_adjustments = []
        for series_options in self.requested_series_options:
            tr = self.get_time_range_after_transforms(
                series_options, metadata, band_range
            )
            pd_series = series_options.apply(xs, tr).loc[gte:lte]
            pd_series_list.append(pd_series)
            band_range_adjustments.append(
                series_options.smoother.get_desired_band_range_adjustment()
            )
            compromise_tr = tr.intersect(self.time_range)
            logger.debug("compromise time range: %s", compromise_tr)
            values = [None if math.isnan(x) else x for x in pd_series.tolist()]
            series = Series(
                options=series_options,
                time_range=compromise_tr,
                values=values,
            )
            series_list.append(series)
        return (series_list, pd_series_list, band_range_adjustments)

    def get_summary_stats(self, series, original_timeseries):
        # Computes summary statistics over requested timeseries band ranges
        summary_stats = [Series.to_summary_stat(s, s.name) for s in series]
        if not isinstance(self.transform, NoTransform):
            # provide original summary stats for z-scores over the original
            # band range, not the adjusted one
            summary_stats.insert(
                0, Series.to_summary_stat(original_timeseries, "Original")
            )
        return summary_stats

    class Config:
        schema_extra = {
            "moving_interval_example": {
                "resolution": "month",
                "dataset_id": "monthly_5x5x60_dataset",
                "variable_id": "float32_variable",
                "time_range": OptionalTimeRange.Config.schema_extra["example"],
                "selected_area": SkopePointModel.Config.schema_extra["example"],
                "zonal_statistic": ZonalStatistic.mean.value,
                "transform": ZScoreMovingInterval.Config.schema_extra["example"],
                "requested_series_options": [
                    SeriesOptions.Config.schema_extra["example"]
                ],
            },
            "fixed_interval_example": {
                "resolution": "month",
                "dataset_id": "monthly_5x5x60_dataset",
                "variable_id": "float32_variable",
                "time_range": OptionalTimeRange.Config.schema_extra["example"],
                "selected_area": SkopePointModel.Config.schema_extra["example"],
                "zonal_statistic": ZonalStatistic.mean.value,
                "transform": ZScoreFixedInterval.Config.schema_extra["example"],
                "requested_series_options": [
                    SeriesOptions.Config.schema_extra["example"]
                ],
            },
        }
