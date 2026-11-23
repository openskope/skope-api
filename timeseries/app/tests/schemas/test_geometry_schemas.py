import pytest
from pydantic import ValidationError

from app.exceptions import SelectedAreaPolygonIsNotValid
from app.schemas.geometry import (
    SkopeFeatureCollectionModel,
    SkopeFeatureModel,
    SkopeMultiPolygonModel,
    SkopePointModel,
    SkopePolygonModel,
)

# Whether an area overlaps the dataset is checked on its grid: see
# tests/core/test_spatial.py.


def _box_coords(minx, miny, maxx, maxy):
    """Returns GeoJSON coordinate ring for a box polygon."""
    return [
        [minx, miny],
        [maxx, miny],
        [maxx, maxy],
        [minx, maxy],
        [minx, miny],
    ]


# A figure-8: valid GeoJSON, but its outline crosses itself.
BOWTIE = [
    [-110.0, 38.0],
    [-109.0, 39.0],
    [-109.0, 38.0],
    [-110.0, 39.0],
    [-110.0, 38.0],
]


def _feature(geometry):
    return {"type": "Feature", "geometry": geometry, "properties": {}}


# ---------------------------------------------------------------------------
# validate_polygons


@pytest.mark.parametrize(
    "area",
    [
        SkopePointModel(type="Point", coordinates=[-110.0, 38.0]),
        SkopePolygonModel(
            type="Polygon", coordinates=[_box_coords(-113, 36, -111, 38)]
        ),
        SkopeMultiPolygonModel(
            type="MultiPolygon", coordinates=[[_box_coords(-113, 36, -111, 38)]]
        ),
    ],
    ids=["point", "polygon", "multipolygon"],
)
def test_a_valid_area_passes(area):
    area.validate_polygons()


@pytest.mark.parametrize(
    "area",
    [
        SkopePolygonModel(type="Polygon", coordinates=[BOWTIE]),
        SkopeFeatureModel(**_feature({"type": "Polygon", "coordinates": [BOWTIE]})),
        SkopeFeatureCollectionModel(
            type="FeatureCollection",
            features=[
                _feature({"type": "Point", "coordinates": [-110.0, 38.0]}),
                _feature({"type": "Polygon", "coordinates": [BOWTIE]}),
            ],
        ),
    ],
    ids=["bare", "feature", "collection"],
)
def test_a_polygon_that_crosses_itself_is_rejected(area):
    with pytest.raises(SelectedAreaPolygonIsNotValid, match="cross itself"):
        area.validate_polygons()


# ---------------------------------------------------------------------------
# The geometries a Feature may hold


@pytest.mark.parametrize("model", [SkopeFeatureModel, SkopeFeatureCollectionModel])
@pytest.mark.parametrize(
    "geometry",
    [{"type": "LineString", "coordinates": [[-110, 38], [-109, 39]]}, None],
    ids=["line", "null"],
)
def test_a_feature_without_a_point_or_polygon_is_not_a_selected_area(model, geometry):
    feature = _feature(geometry)
    body = (
        feature
        if model is SkopeFeatureModel
        else {"type": "FeatureCollection", "features": [feature]}
    )
    with pytest.raises(ValidationError):
        model(**body)


def test_a_point_has_the_same_shape_bare_or_in_a_feature():
    point = {"type": "Point", "coordinates": [-102.0, 37.0]}

    bare = SkopePointModel(**point)
    wrapped = SkopeFeatureModel(**_feature(point))

    assert [s.wkt for s in wrapped.shapes] == [s.wkt for s in bare.shapes]


# ---------------------------------------------------------------------------
# SkopeFeatureCollectionModel.shapes


def test_feature_collection_shapes_returns_all():
    fc = SkopeFeatureCollectionModel(
        type="FeatureCollection",
        features=[
            _feature({"type": "Polygon", "coordinates": [_box_coords(*bounds)]})
            for bounds in ((-113.0, 36.0, -112.0, 37.0), (-111.0, 35.0, -110.0, 36.0))
        ],
    )
    shapes = fc.shapes
    assert len(shapes) == 2
    assert all(hasattr(s, "geom_type") for s in shapes)
