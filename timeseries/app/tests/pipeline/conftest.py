import os
import pytest
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.main import app
from app.store.data_reader import LocalDataReader
from app.store.jobs import FileSystemJobStore, get_job_store
from app.core.job_control import get_job_controller

# Absolute path to the local test rasters and lookup subdirectories.
# fetch_lookup_dict constructs: {DATA_DIR}/{dataset_id}/lookup.json
# resolve_temporal_slice constructs: {DATA_DIR}/{entry["file"]}
DATA_DIR = Path(__file__).parent / "data"

TEST_REGISTRY = {
    "test-annual": {
        "id": "test-annual",
        "crs": "EPSG:4326",
        "transform": [1.0, 0.0, -123.0, 0.0, -1.0, 45.0],
        "timespan": {"period": {"gte": "0001", "lte": "0005"}},
        "variables": [{"id": "ppt"}],
    },
    "test-monthly": {
        "id": "test-monthly",
        "crs": "EPSG:4326",
        "transform": [1.0, 0.0, -123.0, 0.0, -1.0, 45.0],
        "timespan": {"period": {"gte": "0001-01", "lte": "0005-12"}},
        "variables": [{"id": "ppt"}],
    },
}

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


@pytest.fixture
def job_store(tmp_path):
    store_dir = tmp_path / "jobs"
    store_dir.mkdir()
    return FileSystemJobStore(directory=str(store_dir))


@pytest.fixture
def pipeline_client(monkeypatch, tmp_path, job_store):
    # Redirect lookup dict cache so each test gets a clean slate
    cache_dir = str(tmp_path / "lookup_cache")
    os.makedirs(cache_dir, exist_ok=True)
    monkeypatch.setattr("app.store.index_loaders._CACHE_DIR", cache_dir)

    # Patch the lifespan initializers so TestClient doesn't need a real
    # metadata.yml or cloud storage configuration
    monkeypatch.setattr("app.main.load_registry", lambda _: TEST_REGISTRY)
    monkeypatch.setattr("app.main.get_data_reader", lambda _: LocalDataReader())

    # Point timeseries_tasks at DATA_DIR:
    #   - fetch_lookup_dict reads  {DATA_DIR}/{dataset_id}/lookup.json  via LocalDataReader
    #   - resolve_temporal_slice builds  {DATA_DIR}/{entry["file"]}  as the raster URI
    monkeypatch.setattr(
        "app.core.timeseries_tasks.settings",
        SimpleNamespace(storage_base_url=str(DATA_DIR), default_max_cells=500_000),
    )

    app.dependency_overrides[get_job_store] = lambda: job_store

    try:
        with TestClient(app, raise_server_exceptions=True) as client:
            yield client
    finally:
        app.dependency_overrides.clear()
        get_job_controller.cache_clear()
