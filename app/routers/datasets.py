import logging
import numpy as np
import rasterio

from enum import Enum
from fastapi import APIRouter
from pydantic import BaseModel
from rasterio.mask import raster_geometry_mask
from rasterio.windows import Window
from scipy import stats
from shapely import geometry as geom
from typing import List, Optional, Tuple, Union, Literal, Sequence, Any

from ..exceptions import SelectedAreaOutOfBoundsError
from ..stores import YearRange, YearMonthRange, dataset_repo, TimeRange, BandRange

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/datasets", tags=['datasets'])


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


class Geometry(BaseModel):
    type: str
    bbox: Optional[Tuple[float, float, float, float]]


def bounding_box(bounds) -> geom.Polygon:
    return geom.box(
        minx=bounds.left,
        miny=bounds.bottom,
        maxx=bounds.right,
        maxy=bounds.top)


class Point(Geometry):
    type: Literal['Point']
    coordinates: Tuple[float, float]

    def extract(self,
                dataset: rasterio.DatasetReader,
                zonal_statistic: ZonalStatistic,
                band_range: Sequence[int]):
        box = bounding_box(dataset.bounds)
        point = geom.Point(self.coordinates)
        if not box.contains(point):
            raise SelectedAreaOutOfBoundsError('selected area is outside the study area')
        logger.info('extracting point: %s', self)
        px, py = dataset.index(self.coordinates[0], self.coordinates[1])
        logging.info('indices: %s', (px, py))
        return dataset.read(list(band_range), window=Window(px, py, 1, 1)).flatten()


class Polygon(Geometry):
    type: Literal['Polygon']
    coordinates: List[Tuple[float, float]]

    def extract(self,
                dataset: rasterio.DatasetReader,
                zonal_statistic: ZonalStatistic,
                band_range: Sequence[int]):
        box = bounding_box(dataset.bounds)
        polygon = geom.Polygon(self.coordinates)
        if not box.contains(polygon):
            raise SelectedAreaOutOfBoundsError('selected area is outside the study area')
        logger.info('extracting polygon: %s', self)
        zonal_func = zonal_statistic.to_numpy_call()
        masked, transform, window = raster_geometry_mask(dataset, [self], crop=True, all_touched=True)
        result = np.zeros(dataset.count, dtype=dataset.dtypes[0])
        for band in band_range:
            data = dataset.read(band, window=window)
            values = np.ma.array(data=data, mask=np.logical_or(np.equal(data, dataset.nodata), masked))
            result[band] = zonal_func(values)

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
            return width*2 + 1
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


class BaseAnalysisRequest(BaseModel):
    dataset_id: str
    variable_id: str
    selected_area: Union[Point, Polygon]
    zonal_statistic: ZonalStatistic

    def extract_slice(self, dataset: rasterio.DatasetReader, band_range: Sequence[int]):
        return self.selected_area.extract(dataset, self.zonal_statistic, band_range=band_range)

    def get_band_range_to_extract(self, time_range_available: 'YearRange'):
        br_avail = time_range_available.find_band_range(time_range_available)
        desired_br = time_range_available.find_band_range(self.time_range).to_numpy_pair()
        for transform in self.transforms:
            desired_br += transform.get_desired_band_range_adjustment()
        compromise_br = br_avail.intersect(BandRange.from_numpy_pair(desired_br))
        return compromise_br

    def get_time_range_after_transforms(self, time_range_available: 'YearRange', extract_br: BandRange) -> 'YearRange':
        """Get the year range after values after applying transformations"""
        inds = extract_br.to_numpy_pair()
        for transform in self.transforms:
            inds += transform.get_desired_band_range_adjustment()*-1
        yr = time_range_available.translate_band_range(BandRange.from_numpy_pair(inds))
        return yr

    def transform_series(self, xs):
        for transform in self.transforms:
            xs = transform.apply(xs)
        return xs

    def extract(self):
        dataset_meta = dataset_repo.get_dataset_meta(dataset_id=self.dataset_id, variable_id=self.variable_id)
        band_range = self.get_band_range_to_extract(dataset_meta.time_range)
        with rasterio.open(dataset_meta.p) as ds:
            xs = self.extract_slice(ds, band_range=band_range)
        xs = self.transform_series(xs)
        yr = self.get_time_range_after_transforms(dataset_meta.time_range, band_range)
        values = xs.tolist()
        return {'time_range': yr, 'values': values}


class MonthAnalysisRequest(BaseAnalysisRequest):
    resolution: Literal['month']
    time_range: YearMonthRange
    transforms: List[Union[MovingAverageSmoother, ZScoreRoller]]


class YearAnalysisRequest(BaseAnalysisRequest):
    resolution: Literal['year']
    time_range: YearRange
    transforms: List[Union[MovingAverageSmoother, ZScoreRoller, ZScoreScaler]]


class YearAnalysisResponse(BaseModel):
    time_range: YearRange
    values: List[float]


class MonthAnalysisResponse(BaseModel):
    time_range: YearMonthRange
    values: List[float]


@router.post("/monthly", operation_id='retrieveMonthlyTimeseries')
def extract_monthly_timeseries(data: MonthAnalysisRequest):
    return data.extract()


@router.post("/yearly", operation_id='retrieveYearlyTimeseries')
def extract_yearly_timeseries(data: YearAnalysisRequest) -> YearAnalysisResponse:
    return YearAnalysisResponse(**data.extract())