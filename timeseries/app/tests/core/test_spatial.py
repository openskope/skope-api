from shapely.geometry import box

from app.core.spatial import estimate_cell_count

# ---------------------------------------------------------------------------
# estimate_cell_count


def test_estimate_cell_count_geographic():
    # 1° × 1° box, 0.00833° pixels, EPSG:4326
    shapes = [box(-110.0, 37.0, -109.0, 38.0)]
    transform = [0.00833, 0.0, -115.0, 0.0, -0.00833, 43.0]
    result = estimate_cell_count(shapes, transform, "EPSG:4326")
    # The bounds span columns and rows 600.24 .. 720.29: cells 600 .. 720.
    assert result == 121 * 121


def test_estimate_cell_count_projected_uses_exact_crs_transform():
    shapes = [box(-110.0, 37.0, -109.0, 38.0)]
    transform = [800.0, 0.0, 200000.0, 0.0, -800.0, 4800000.0]
    result = estimate_cell_count(shapes, transform, "EPSG:32612")

    # The precise UTM footprint differs from the former square
    # meters-per-degree approximation.
    assert result == 114 * 141


def test_estimate_cell_count_zero_area():
    shapes = [box(0.0, 0.0, 0.0, 0.0)]
    transform = [0.00833, 0.0, -115.0, 0.0, -0.00833, 43.0]
    result = estimate_cell_count(shapes, transform, "EPSG:4326")
    assert result == 1
