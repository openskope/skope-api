from enum import Enum, unique
from typing import Optional, List, Tuple, Union
from fastapi import FastAPI
from pydantic import BaseModel
import rasterio
import numpy as np
import logging
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


class Polygon(Geometry):
    type = 'Polygon'
    coordinates: List[Tuple[float, float]]


class YearMonth(BaseModel):
    year: int
    month: Optional[int]


class Smoother(BaseModel):
    type: str


class NoSmoother(Smoother):
    type = 'NoSmoother'


class WindowType(str, Enum):
    centered = 'centered'
    trailing = 'trailing'


class MovingAverageSmoother(Smoother):
    type = 'MovingAverageSmoother'
    method: WindowType
    width: float


class CoordinateTransform(str, Enum):
    none = 'none'
    zscore = 'zscore'


class AnalysisRequest(BaseModel):
    selectedArea: Union[Point, Polygon]
    zonalStatistic: ZonalStatistic
    timeRange: Tuple[YearMonth, YearMonth]
    smoother: Union[NoSmoother, MovingAverageSmoother]
    coordinateTransform: CoordinateTransform


class AnalysisResponse(BaseModel):
    timeRange: Tuple[YearMonth, YearMonth]
    values: List[float]


app = FastAPI()


def extract_point(dataset: rasterio.DatasetReader, point: Point) -> np.array:
    print(f'extracting point: {point.to_string()}')
    px, py = dataset.index(point.coordinates[0], point.coordinates[1])
    print(f'indices: ({px}, {py})')
    return dataset.read(window=Window(px, py, 1, 1)).flatten()


def extract_polygon(dataset: rasterio.DatasetReader, zonal_statistics: ZonalStatistic, polygon: Polygon) -> np.array:
    print(f'extracting polygon: {polygon.to_string()}')
    zonal_func = zonal_statistics.to_numpy_call()
    masked, transform, window = raster_geometry_mask(dataset, [polygon], crop=True, all_touched=True)
    result = np.zeros(dataset.count, dtype=dataset.dtypes[0])
    for band in range(dataset.count):
        data = dataset.read(band + 1, window=window)
        values = np.ma.array(data=data, mask=np.logical_or(np.equal(data, dataset.nodata), masked))
        result[band] = zonal_func(values)

    return result


# add request timeout middleware https://github.com/tiangolo/fastapi/issues/1752
@app.post("/datasets/{datasetId}", response_model=AnalysisResponse)
def read_root(datasetId: str, data: AnalysisRequest):
    with rasterio.open(f'data/{datasetId}.tif') as dataset:
        if data.selectedArea.type == 'Point':
            w = extract_point(dataset=dataset, point=data.selectedArea)
        else:
            w = extract_polygon(dataset=dataset, zonal_statistics=data.zonalStatistic, polygon=data.selectedArea)
    if data.coordinateTransform == CoordinateTransform.zscore:
        w = stats.zscore(w)
    return AnalysisResponse(timeRange=(YearMonth(year=1500), YearMonth(year=1800)), values=list(w))