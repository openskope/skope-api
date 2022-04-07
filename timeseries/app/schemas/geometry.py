from abc import ABCMeta, abstractmethod
from geojson_pydantic import Feature, FeatureCollection, Point, Polygon
from rasterio import DatasetReader, features, windows, mask
from shapely import geometry as geom
from shapely.ops import orient
from shapely.validation import explain_validity
from typing import Sequence

import logging
import numpy as np
import pyproj


from .common import ZonalStatistic, BandRange
from app.exceptions import (
    SelectedAreaOutOfBoundsError,
    SelectedAreaPolygonIsTooLarge,
    SelectedAreaPolygonIsNotValid,
)

logger = logging.getLogger(__name__)


def bounding_box(bounds) -> geom.Polygon:
    return geom.box(
        minx=bounds.left, miny=bounds.bottom, maxx=bounds.right, maxy=bounds.top
    )


class SkopeGeometry(metaclass=ABCMeta):
    @property
    def shapes(self):
        return [geom.shape(self)]

    def extract(
        self,
        dataset: DatasetReader,
        zonal_statistic: ZonalStatistic,
        band_range: Sequence[int],
    ):
        self.validate_geometry(dataset)
        return self.extract_raster_slice(dataset, zonal_statistic, band_range)

    @abstractmethod
    def extract_raster_slice(
        self,
        dataset: DatasetReader,
        zonal_statistic: ZonalStatistic,
        band_range: Sequence[int],
    ):
        pass

    @abstractmethod
    def validate_geometry(self, dataset: DatasetReader):
        pass


class SkopePointModel(Point, SkopeGeometry):
    @staticmethod
    def calculate_area(column_index: int, row_index: int, dataset: DatasetReader):
        wgs84 = pyproj.Geod(ellps="WGS84")
        top_left = dataset.xy(row=row_index, col=column_index)
        bottom_right = dataset.xy(row=row_index + 1, col=column_index + 1)
        top_right = (bottom_right[0], top_left[1])
        bottom_left = (top_left[0], bottom_right[1])
        bbox = geom.Polygon([top_left, bottom_left, bottom_right, top_right, top_left])
        area, perimeter = wgs84.geometry_area_perimeter(bbox)
        return abs(area)

    def validate_geometry(self, dataset: DatasetReader):
        box = bounding_box(dataset.bounds)
        point = geom.Point(self.coordinates)
        if not box.covers(point):
            raise SelectedAreaOutOfBoundsError(
                "selected area is not covered by the dataset region"
            )

    def extract_raster_slice(
        self,
        dataset: DatasetReader,
        zonal_statistic: ZonalStatistic,
        band_range: Sequence[int],
    ):
        logger.info("extracting point: %s", self)
        row_index, column_index = dataset.index(
            self.coordinates[0], self.coordinates[1]
        )
        logging.info("indices: %s", (row_index, column_index))
        data = dataset.read(
            list(band_range),
            window=windows.Window(column_index, row_index, 1, 1),
            out_dtype=np.float64,
        ).flatten()
        data[np.equal(data, dataset.nodata)] = np.nan
        area = self.calculate_area(
            column_index=column_index, row_index=row_index, dataset=dataset
        )
        return {
            "n_cells": 1,
            "area": area,
            "data": data,
        }

    class Config:
        schema_extra = {"example": {"type": "Point", "coordinates": [-120, 42.5]}}


class BaseSkopePolygonModel(SkopeGeometry):
    @staticmethod
    def _make_band_range_groups(
        *, width: int, height: int, band_range: BandRange, max_size=250000
    ):
        n_cells_per_band = width * height  # 25
        n_cells_per_full_chunk = max_size - max_size % n_cells_per_band
        if n_cells_per_full_chunk == 0:
            raise SelectedAreaPolygonIsTooLarge(
                n_cells=n_cells_per_band, max_cells=max_size
            )
        n_bands = len(band_range)
        n = n_cells_per_band * n_bands  # 650
        n_full_chunks = n // n_cells_per_full_chunk  # 650 // 625 = 1
        n_bands_per_full_chunk = n_cells_per_full_chunk // n_cells_per_band
        offset = band_range.gte
        for i in range(n_full_chunks):
            band_indices = range(
                i * n_bands_per_full_chunk + offset,
                (i + 1) * n_bands_per_full_chunk + offset,
            )
            yield band_indices
        n_last_bands = n_bands % (
            n_cells_per_full_chunk // n_cells_per_band
        )  # 26 % (625 // 25) = 26 % 25 = 1
        if n_last_bands > 0:
            yield range(n_bands - n_last_bands + offset, n_bands + offset)

    @staticmethod
    def calculate_area(masked, transform):
        shape_iter = features.shapes(
            masked.astype("uint8"), mask=np.equal(masked, 0), transform=transform
        )
        area = 0.0
        wgs84 = pyproj.Geod(ellps="WGS84")
        for shp, val in shape_iter:
            shp = orient(shp)
            shp = geom.shape(shp)
            area += wgs84.geometry_area_perimeter(shp)[0]
        # area is signed positive or negative based on clockwise or
        # counterclockwise traversal:
        # https://pyproj4.github.io/pyproj/stable/api/geod.html?highlight=counter%20clockwise#pyproj.Geod.geometry_area_perimeter
        # return the absolute value of the area
        return abs(area)

    def validate_geometry(self, dataset: DatasetReader):
        box = bounding_box(dataset.bounds)
        for shape in self.shapes:
            if not shape.is_valid:
                raise SelectedAreaPolygonIsNotValid(
                    f"selected area is not a valid polygon: {explain_validity(polygon).lower()}"
                )
            # DE-9IM format
            # https://giswiki.hsr.ch/images/3/3d/9dem_springer.pdf
            # 'T********' means that the interior of the bounding box must intersect the interior of the selected area
            if not box.relate_pattern(shape, "T********"):
                raise SelectedAreaOutOfBoundsError(
                    "no interior point of the selected area intersects an interior point of the dataset region"
                )

    def extract_raster_slice(
        self,
        dataset: DatasetReader,
        zonal_statistic: ZonalStatistic,
        band_range: BandRange,
    ):
        zonal_func = zonal_statistic.to_numpy_call()
        masked, transform, window = mask.raster_geometry_mask(
            dataset, shapes=self.shapes, crop=True, all_touched=True
        )
        n_cells = masked.size - np.count_nonzero(masked)
        area = self.calculate_area(masked, transform=transform)
        result = np.empty(len(band_range), dtype=np.float64)
        result.fill(np.nan)
        offset = -band_range.gte
        for band_group in self._make_band_range_groups(
            width=window.width, height=window.height, band_range=band_range
        ):
            data = dataset.read(list(band_group), window=window)
            masked_values = np.ma.array(
                data=data, mask=np.logical_or(np.equal(data, dataset.nodata), masked)
            )
            lb = band_group.start + offset
            ub = band_group.stop + offset
            zonal_func_results = zonal_func(masked_values, axis=(1, 2))
            # result[lb:ub] = [np.nan if np.equal(v, dataset.nodata) else v for v in zonal_func_results]
            result[lb:ub] = zonal_func_results.filled(fill_value=np.nan)

        return {"n_cells": n_cells, "area": area, "data": result}


class SkopePolygonModel(Polygon, BaseSkopePolygonModel):
    pass


class SkopeFeatureModel(Feature, BaseSkopePolygonModel):
    @property
    def shapes(self):
        return [geom.shape(self.geometry)]

    @property
    def coordinates(self):
        return self.geometry.coordinates


class SkopeFeatureCollectionModel(FeatureCollection, BaseSkopePolygonModel):
    @property
    def shapes(self):
        return [geom.shape(feature.geometry) for feature in self.features]

    @property
    def coordinates(self):
        return [feature.geometry.coordinates for feature in self.features]
