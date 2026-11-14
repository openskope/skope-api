import numpy as np
import pytest

from fastapi.testclient import TestClient

import app.main
from app.main import app as fastapi_app
from app.store.jobs import FileSystemJobStore, get_job_store
from app.core.job_control import get_job_controller
from app.tests.release_builder import build_release, write_pin

# Two small releases on a 5x5 grid of 1-degree pixels with its top-left corner
# at (-123, 45). Every pixel of timestep i holds the value i + 1.
GRID = (1.0, 0.0, -123.0, 0.0, -1.0, 45.0)


def _cube(count):
    return np.stack([np.full((5, 5), i + 1, dtype=np.float32) for i in range(count)])


# 1°×1° polygon covering exactly one pixel of the 1°/pixel test rasters.
# Pixel column 1, row 1 in the 5×5 grid (0-indexed from top-left at -123, 45).
SINGLE_CELL_POLYGON = {
    "type": "Polygon",
    "coordinates": [
        [
            [-122.0, 43.0],
            [-121.0, 43.0],
            [-121.0, 44.0],
            [-122.0, 44.0],
            [-122.0, 43.0],
        ]
    ],
}


DISPLAY = """
datasets:
  test_annual:
    order: 10
    variables:
      ppt: {order: 10, palette: skope-precip, range: [0, 4.5]}
  test_monthly:
    order: 20
    default_variable: ppt
    variables:
      ppt: {order: 10, palette: skope-precip, range: [1, 60]}
"""
PALETTES = """
skope-precip:
  kind: ramp
  colors: ["#B5834A", "#358C87"]
"""


def write_display(tmp_path, monkeypatch, preferences=DISPLAY):
    (tmp_path / "preferences.yml").write_text(preferences)
    (tmp_path / "palettes.yml").write_text(PALETTES)
    monkeypatch.setattr(
        app.main.settings, "display_preferences_path", str(tmp_path / "preferences.yml")
    )
    monkeypatch.setattr(
        app.main.settings, "display_palettes_path", str(tmp_path / "palettes.yml")
    )


@pytest.fixture
def job_store(tmp_path):
    store_dir = tmp_path / "jobs"
    store_dir.mkdir()
    return FileSystemJobStore(directory=str(store_dir))


@pytest.fixture
def pipeline_client(monkeypatch, tmp_path, job_store):
    # Serve two synthetic releases through the real startup composition.
    root = tmp_path / "releases"
    releases = [
        build_release(root, "test_annual", {"ppt": _cube(5)}, transform=GRID),
        build_release(
            root,
            "test_monthly",
            {"ppt": _cube(60)},
            origin="0001-01",
            step="P1M",
            precision="month",
            chunk_size=12,
            transform=GRID,
        ),
    ]
    pin = write_pin(tmp_path / "releases.yml", root, releases)
    monkeypatch.setattr(app.main.settings, "release_pin_path", str(pin))
    monkeypatch.setattr(app.main.settings, "release_root", str(root))
    write_display(tmp_path, monkeypatch)

    fastapi_app.dependency_overrides[get_job_store] = lambda: job_store

    try:
        with TestClient(fastapi_app, raise_server_exceptions=True) as client:
            yield client
    finally:
        fastapi_app.dependency_overrides.clear()
        get_job_controller.cache_clear()
