import pytest

from app.exceptions import SelectedAreaOutOfBoundsError, SelectedAreaPolygonIsNotValid
from app.schemas.geometry import (
    SkopeFeatureCollectionModel,
    SkopePointModel,
    SkopePolygonModel,
)


def _box_coords(minx, miny, maxx, maxy):
    """Returns GeoJSON coordinate ring for a box polygon."""
    return [
        [minx, miny],
        [maxx, miny],
        [maxx, maxy],
        [minx, maxy],
        [minx, miny],
    ]


# ---------------------------------------------------------------------------
# SkopePointModel.validate_geometry


def test_point_inside_bbox_no_error(dataset_bbox):
    point = SkopePointModel(type="Point", coordinates=[-110.0, 38.0])
    point.validate_geometry(dataset_bbox)  # no exception


def test_point_outside_bbox_raises(dataset_bbox):
    point = SkopePointModel(type="Point", coordinates=[-130.0, 38.0])
    with pytest.raises(SelectedAreaOutOfBoundsError):
        point.validate_geometry(dataset_bbox)


def test_point_on_bbox_boundary_no_error(dataset_bbox):
    # Shapely `covers` includes boundary points
    point = SkopePointModel(type="Point", coordinates=[-115.0, 31.0])
    point.validate_geometry(dataset_bbox)


# ---------------------------------------------------------------------------
# SkopePolygonModel.validate_geometry


def test_polygon_inside_bbox_no_error(dataset_bbox):
    coords = [_box_coords(-113.0, 36.0, -111.0, 38.0)]
    poly = SkopePolygonModel(type="Polygon", coordinates=coords)
    poly.validate_geometry(dataset_bbox)


def test_polygon_partially_overlapping_bbox_no_error(dataset_bbox):
    # Extends east beyond the bbox boundary but has interior intersection
    coords = [_box_coords(-110.0, 37.0, -98.0, 40.0)]
    poly = SkopePolygonModel(type="Polygon", coordinates=coords)
    poly.validate_geometry(dataset_bbox)


def test_polygon_outside_bbox_raises(dataset_bbox):
    coords = [_box_coords(-130.0, 45.0, -120.0, 50.0)]
    poly = SkopePolygonModel(type="Polygon", coordinates=coords)
    with pytest.raises(SelectedAreaOutOfBoundsError):
        poly.validate_geometry(dataset_bbox)


def test_polygon_invalid_self_intersecting_raises(dataset_bbox):
    # Bowtie (figure-8) — valid GeoJSON ring but not a simple polygon
    bowtie_coords = [
        [-110.0, 38.0],
        [-109.0, 39.0],
        [-109.0, 38.0],
        [-110.0, 39.0],
        [-110.0, 38.0],
    ]
    poly = SkopePolygonModel(type="Polygon", coordinates=[bowtie_coords])
    with pytest.raises(SelectedAreaPolygonIsNotValid):
        poly.validate_geometry(dataset_bbox)


def test_polygon_touching_bbox_edge_only_raises(dataset_bbox):
    # Polygon immediately to the right of bbox, sharing only the right edge (x=-102)
    coords = [_box_coords(-102.0, 31.0, -101.0, 43.0)]
    poly = SkopePolygonModel(type="Polygon", coordinates=coords)
    with pytest.raises(SelectedAreaOutOfBoundsError):
        poly.validate_geometry(dataset_bbox)


# ---------------------------------------------------------------------------
# SkopeFeatureCollectionModel.shapes


def test_feature_collection_shapes_returns_all():
    feat1 = {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [_box_coords(-113.0, 36.0, -112.0, 37.0)],
        },
        "properties": {},
    }
    feat2 = {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [_box_coords(-111.0, 35.0, -110.0, 36.0)],
        },
        "properties": {},
    }
    fc = SkopeFeatureCollectionModel(type="FeatureCollection", features=[feat1, feat2])
    shapes = fc.shapes
    assert len(shapes) == 2
    assert all(hasattr(s, "geom_type") for s in shapes)
