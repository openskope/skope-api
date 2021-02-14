import logging
import numpy as np
import rasterio

from enum import Enum
from fastapi import APIRouter
from pydantic import BaseModel
from rasterio.mask import raster_geometry_mask
from rasterio.windows import Window
from scipy import stats
from typing import List, Optional, Tuple, Union, Literal

from ..stores import YearRange, YearMonthRange, dataset_repo

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/datasets", tags=['datasets'])


class ZonalStatistic(str, Enum):
    mean = 'mean'
    median = 'median'

    def to_numpy_call(self):
        if self == self.mean:
            return np.mean
        elif self == self.median:
            return np.median


class Geometry(BaseModel):
    type: str
    bbox: Optional[Tuple[float, float, float, float]]


class Point(Geometry):
    type: Literal['Point']
    coordinates: Tuple[float, float]

    def extract(self, dataset: rasterio.DatasetReader, zonal_statistic: ZonalStatistic):
        logger.info('extracting point: %s', self)
        px, py = dataset.index(self.coordinates[0], self.coordinates[1])
        logging.info('indices: %s', (px, py))
        return dataset.read(window=Window(px, py, 1, 1)).flatten()


class Polygon(Geometry):
    type: Literal['Polygon']
    coordinates: List[Tuple[float, float]]

    def extract(self, dataset: rasterio.DatasetReader, zonal_statistic: ZonalStatistic):
        logger.info('extracting polygon: %s', self)
        zonal_func = zonal_statistic.to_numpy_call()
        masked, transform, window = raster_geometry_mask(dataset, [self], crop=True, all_touched=True)
        result = np.zeros(dataset.count, dtype=dataset.dtypes[0])
        for band in range(dataset.count):
            data = dataset.read(band + 1, window=window)
            values = np.ma.array(data=data, mask=np.logical_or(np.equal(data, dataset.nodata), masked))
            result[band] = zonal_func(values)

        return result


class Smoother(BaseModel):
    type: str


class NoSmoother(Smoother):
    type: Literal['NoSmoother']

    def smooth(self, xs):
        return xs


class WindowType(str, Enum):
    centered = 'centered'
    trailing = 'trailing'


class MovingAverageSmoother(Smoother):
    type: Literal['MovingAverageSmoother']
    method: WindowType
    width: int

    def apply(self, xs):
        return np.convolve(xs, np.ones(self.width) / self.width, 'valid')


class ZScoreRoller(BaseModel):
    type: Literal['ZScoreRoller']
    width: int

    def apply(self, xs):
        n = len(xs) - self.width
        results = np.zeros(n)
        for i in range(n):
            results[i] = (xs[i + self.width] - np.mean(xs[i:(i + self.width)])) / np.std(xs[i:(i + self.width)])
        return results


class NoRoller(BaseModel):
    type: Literal['NoneRoller']

    def roll(self, xs):
        return xs


class NoScaler(BaseModel):
    type: Literal['NoScaler']

    def scale(self, xs):
        return xs


class ZScoreScaler(BaseModel):
    type: Literal['ZScoreScaler']

    def aaply(self, xs):
        return stats.zscore(xs)


class YearAnalysisRequest(BaseModel):
    resolution: Literal['year']
    dataset_id: str
    variable_id: str
    selected_area: Union[Point, Polygon]
    zonal_statistic: ZonalStatistic
    transforms: List[Union[MovingAverageSmoother, ZScoreRoller, ZScoreScaler]]

    def extract_slice(self, dataset: rasterio.DatasetReader):
        return self.selected_area.extract(dataset, self.zonal_statistic)

    def transform_series(self, xs):
        for transform in self.transforms:
            xs = transform.apply(xs)
        return xs

    def extract(self):
        dataset_meta = dataset_repo.get_dataset_meta(dataset_id=self.dataset_id, variable_id=self.variable_id)
        with rasterio.open(dataset_meta.p) as ds:
            xs = self.extract_slice(ds)
        xs = self.transform_series(xs)
        return xs


class YearAnalysisResponse(BaseModel):
    time_range: YearRange
    values: List[float]


class MonthAnalysisResponse(BaseModel):
    timeRange: YearMonthRange
    values: List[float]


@router.post("/yearly", operation_id='retrieveYearlyTimeseries')
def extract_yearly_timeseries(data: YearAnalysisRequest):
    xs = data.extract()
    return YearAnalysisResponse(time_range=YearRange(gte=1500, lte=1800), values=list(xs))