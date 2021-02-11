from enum import Enum, unique
from typing import Optional, List, Tuple, Union
from fastapi import FastAPI
from pydantic import BaseModel
import rasterio
import numpy as np
from rasterio.windows import Window
from rasterio.mask import raster_geometry_mask


class ZonalStatistic(str, Enum):
    mean = 'mean'
    median = 'median'


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
    px, py = dataset.index(point.coordinates[0], point.coordinates[1])
    return dataset.read(window=Window(px, py, 1, 1))


def extract_polygon(dataset: rasterio.DatasetReader, polygon: Polygon) -> np.array:
    masked, transform, window = raster_geometry_mask(dataset, [polygon], crop=True, all_touched=True)
    result = np.zeros(dataset.count, dtype=dataset.dtypes[0])
    for band in range(dataset.count):
        data = dataset.read(band + 1, window=window)
        result[band] = np.mean(data)
    return result


# add request timeout middleware https://github.com/tiangolo/fastapi/issues/1752
@app.post("/datasets/{datasetId}", response_model=AnalysisResponse)
def read_root(datasetId: str, data: AnalysisRequest):
    with rasterio.open(f'data/{datasetId}.tif') as src:
        w = src.read(window=Window(1,1,1,1)).flatten()
    return AnalysisResponse(timeRange=(YearMonth(year=1500), YearMonth(year=1800)), values=list(w))