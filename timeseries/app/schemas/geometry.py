import logging
from typing import Annotated, Union

from pydantic import ConfigDict, Field
from geojson_pydantic import Feature, FeatureCollection, MultiPolygon, Point, Polygon
from shapely import geometry as geom, get_num_coordinates
from shapely.validation import explain_validity

from app.config import get_settings
from app.exceptions import SelectedAreaPolygonIsNotValid

logger = logging.getLogger(__name__)
settings = get_settings()

# The geometries a selected area may hold, bare or in a Feature: the ones the
# app draws and imports. The `type` tag picks the model, so a wrong shape gets
# one error naming it, not one per model tried.
AreaGeometry = Annotated[
    Union[Point, Polygon, MultiPolygon], Field(discriminator="type")
]


class SkopeGeometry:
    @property
    def shapes(self) -> list[geom.base.BaseGeometry]:
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

    def validate_polygons(self) -> None:
        """Raise if a polygon in the area is invalid. Whether the area overlaps
        the dataset is checked on its grid, by `resolve_spatial_window`."""
        for shape in self.shapes:
            if not shape.is_valid:
                raise SelectedAreaPolygonIsNotValid(
                    "The selected area is not a valid polygon "
                    f"({explain_validity(shape).lower()}): an outline must not "
                    "cross itself. Redraw it or fix the file."
                )


class SkopePointModel(Point, SkopeGeometry):
    model_config = ConfigDict(
        json_schema_extra={"example": {"type": "Point", "coordinates": [-120, 42.5]}}
    )


class SkopePolygonModel(Polygon, SkopeGeometry):
    pass


class SkopeMultiPolygonModel(MultiPolygon, SkopeGeometry):
    pass


class SkopeFeatureModel(Feature[AreaGeometry, dict | None], SkopeGeometry):
    geometry: AreaGeometry  # a selected area needs a shape: not null

    @property
    def shapes(self) -> list[geom.base.BaseGeometry]:
        return [geom.shape(self.geometry)]


class SkopeFeatureCollectionModel(FeatureCollection[SkopeFeatureModel], SkopeGeometry):
    @property
    def shapes(self) -> list[geom.base.BaseGeometry]:
        return [geom.shape(feature.geometry) for feature in self.features]
