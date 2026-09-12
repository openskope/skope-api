import logging
from abc import ABCMeta, abstractmethod
from typing import List

from pydantic import ConfigDict
from geojson_pydantic import Feature, FeatureCollection, Point, Polygon
from shapely import geometry as geom, get_num_coordinates
from shapely.validation import explain_validity

from app.config import get_settings
from app.exceptions import (
    SelectedAreaOutOfBoundsError,
    SelectedAreaPolygonIsNotValid,
)

logger = logging.getLogger(__name__)
settings = get_settings()


class SkopeGeometry(metaclass=ABCMeta):
    @property
    def shapes(self) -> List[geom.base.BaseGeometry]:
        """Converts the GeoJSON Pydantic model into a list of Shapely geometries."""
        return [geom.shape(self)]

    def validate_complexity(self, max_shapes: int, max_coordinates: int) -> None:
        shapes = self.shapes
        if len(shapes) > max_shapes:
            raise ValueError(
                f"Selected area has {len(shapes)} shapes; maximum is {max_shapes}."
            )

        coordinate_count = sum(get_num_coordinates(shape) for shape in shapes)
        if coordinate_count > max_coordinates:
            raise ValueError(
                "Selected area has "
                f"{coordinate_count} coordinates; maximum is {max_coordinates}."
            )

    @abstractmethod
    def validate_geometry(self, dataset_bbox: geom.Polygon):
        """Validates that the geometry intersects the dataset's bounding box."""
        pass


class SkopePointModel(Point, SkopeGeometry):
    def validate_geometry(self, dataset_bbox: geom.Polygon):
        point = geom.Point(self.coordinates)
        if not dataset_bbox.covers(point):
            raise SelectedAreaOutOfBoundsError(
                "Selected area is not covered by the dataset region."
            )

    model_config = ConfigDict(
        json_schema_extra={"example": {"type": "Point", "coordinates": [-120, 42.5]}}
    )


class BaseSkopePolygonModel(SkopeGeometry):
    def validate_geometry(self, dataset_bbox: geom.Polygon):
        for shape in self.shapes:
            if not shape.is_valid:
                raise SelectedAreaPolygonIsNotValid(
                    f"Selected area is not a valid polygon: {explain_validity(shape).lower()}"
                )

            # DE-9IM format: 'T********' indicates the interior of the bounding box
            # must intersect the interior of the selected area.
            if not dataset_bbox.relate_pattern(shape, "T********"):
                raise SelectedAreaOutOfBoundsError(
                    "No interior point of the selected area intersects the dataset region."
                )


class SkopePolygonModel(Polygon, BaseSkopePolygonModel):
    pass


class SkopeFeatureModel(Feature, BaseSkopePolygonModel):
    @property
    def shapes(self) -> List[geom.base.BaseGeometry]:
        return [geom.shape(self.geometry)]


class SkopeFeatureCollectionModel(FeatureCollection, BaseSkopePolygonModel):
    @property
    def shapes(self) -> List[geom.base.BaseGeometry]:
        return [geom.shape(feature.geometry) for feature in self.features]
