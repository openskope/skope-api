from enum import Enum, unique
from typing import Optional, List, Tuple, Union
from fastapi import FastAPI
from pydantic import BaseModel
import rasterio
import numpy as np
import logging
import yaml
from rasterio.windows import Window
from rasterio.mask import raster_geometry_mask
from scipy import stats


logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')


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
    type = 'Point'
    coordinates: Tuple[float, float]

    def extract(self, dataset: rasterio.DatasetReader, zonal_statistic: ZonalStatistic):
        print(f'extracting point: {self.to_string()}')
        px, py = dataset.index(self.coordinates[0], self.coordinates[1])
        print(f'indices: ({px}, {py})')
        return dataset.read(window=Window(px, py, 1, 1)).flatten()


class Polygon(Geometry):
    type = 'Polygon'
    coordinates: List[Tuple[float, float]]

    def extract(self, dataset: rasterio.DatasetReader, zonal_statistic: ZonalStatistic):
        print(f'extracting polygon: {self.to_string()}')
        zonal_func = zonal_statistic.to_numpy_call()
        masked, transform, window = raster_geometry_mask(dataset, [self], crop=True, all_touched=True)
        result = np.zeros(dataset.count, dtype=dataset.dtypes[0])
        for band in range(dataset.count):
            data = dataset.read(band + 1, window=window)
            values = np.ma.array(data=data, mask=np.logical_or(np.equal(data, dataset.nodata), masked))
            result[band] = zonal_func(values)

        return result


class YearMonth(BaseModel):
    year: int
    month: Optional[int]


class Smoother(BaseModel):
    type: str


class NoSmoother(Smoother):
    type = 'NoSmoother'

    def smooth(self, xs):
        print('smoother: none')
        return xs


class WindowType(str, Enum):
    centered = 'centered'
    trailing = 'trailing'


class MovingAverageSmoother(Smoother):
    type = 'MovingAverageSmoother'
    method: WindowType
    width: int

    def smooth(self, xs):
        print(f'smoother: moving average {self.width}')
        return np.convolve(xs, np.ones(self.width) / self.width, 'valid')


class ZScoreRoller(BaseModel):
    type = 'ZScoreRoller'
    width: int

    def roll(self, xs):
        n = len(xs) - self.width
        results = np.zeros(n)
        for i in range(n):
            results[i] = (xs[i+self.width] - np.mean(xs[i:(i+self.width)]))/np.std(xs[i:(i+self.width)])
        return results


class NoRoller(BaseModel):
    type = 'NoneRoller'

    def roll(self, xs):
        return xs


class CoordinateTransform(str, Enum):
    none = 'none'
    zscore = 'zscore'

    def transform(self, xs):
        if self == self.none:
            return xs
        else:
            return stats.zscore(xs)


class NoCoordinateTransform(BaseModel):
    type = 'NoCoordinateTransform'

    def transform(self, xs):
        return xs


class ZScoreCoordinateTransform(BaseModel):
    type = 'ZScoreCoordinateTransform'
    time_range: Tuple[YearMonth, YearMonth]

    def transform(self, xs):
        return stats.zscore(xs)


class AnalysisRequest(BaseModel):
    selectedArea: Union[Point, Polygon]
    zonalStatistic: ZonalStatistic
    timeRange: Tuple[YearMonth, YearMonth]
    smoother: Union[MovingAverageSmoother, NoSmoother]
    roller: Union[ZScoreRoller, NoRoller]
    coordinateTransform: Union[ZScoreCoordinateTransform, NoCoordinateTransform]

    def extract(self, dataset_id: str):
        with rasterio.open(f'data/{dataset_id}.tif') as dataset:
            w = self.selectedArea.extract(dataset, self.zonalStatistic)
        w = self.smoother.smooth(w)
        w = self.roller.roll(w)
        w = self.coordinateTransform.transform(w)
        return w


class AnalysisResponse(BaseModel):
    timeRange: Tuple[YearMonth, YearMonth]
    values: List[float]


class OutOfBoundsError(ValueError):
    pass


app = FastAPI()

with open("metadata.yml") as f:
    _metadata = yaml.safe_load(f)

@app.get("/metadata")
def metadata():
    return _metadata


# add request timeout middleware https://github.com/tiangolo/fastapi/issues/1752
@app.post("/datasets/{dataset_id}", response_model=AnalysisResponse, operation_id='retrieveTimeseries')
def extract_timeseries(dataset_id: str, data: AnalysisRequest):
    w = data.extract(dataset_id)
    return AnalysisResponse(timeRange=(YearMonth(year=1500), YearMonth(year=1800)), values=list(w))