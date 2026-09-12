import json
import os
import tempfile
from pathlib import Path

import pytest
import pandas as pd
from unittest.mock import AsyncMock
from shapely.geometry import box

# ---------------------------------------------------------------------------
# Working directory fix
#
# Settings requires config/app_settings.yml relative to CWD. In the container
# the config/ directory lives above the app/ directory (e.g. /code/config while
# pytest runs from /code/app). Walk upward from this file until we find a
# directory that contains config/app_settings.yml, then switch to it so that
# Settings() can locate its YAML file at import time.


def _find_config_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "config" / "app_settings.yml").exists():
            return parent
    return Path(__file__).resolve().parent  # fallback: stay put


os.chdir(_find_config_root())

from app.store.data_reader import DataReader
from app.store.jobs import FileSystemJobStore, RedisJobStore

# ---------------------------------------------------------------------------
# Registry / lookup fixtures


@pytest.fixture
def minimal_registry():
    return {
        "valid-ds": {
            "id": "valid-ds",
            "crs": "EPSG:4326",
            "transform": [0.00833, 0.0, -115.0, 0.0, -0.00833, 43.0],
            "variables": [{"id": "ppt"}],
        },
        "projected-ds": {
            "id": "projected-ds",
            "crs": "EPSG:32612",
            "transform": [800.0, 0.0, 200000.0, 0.0, -800.0, 4800000.0],
            "variables": [{"id": "temp"}],
        },
    }


@pytest.fixture
def minimal_lookup_data():
    return {
        "ppt": {
            "0100": {"file": "file_a.tif", "bidx": 1},
            "0101": {"file": "file_a.tif", "bidx": 2},
            "0102": {"file": "file_b.tif", "bidx": 1},
            "0103": {"file": "file_b.tif", "bidx": 2},
            "0104": {"file": "file_b.tif", "bidx": 3},
            "0105": {"file": "file_c.tif", "bidx": 1},
        },
        "temp": {
            "0100": {"file": "temp_a.tif", "bidx": 1},
        },
    }


# ---------------------------------------------------------------------------
# Transform / spatial fixtures


@pytest.fixture
def geo_transform_6():
    return [0.00833, 0.0, -115.0, 0.0, -0.00833, 43.0]


@pytest.fixture
def proj_transform_6():
    return [800.0, 0.0, 200000.0, 0.0, -800.0, 4800000.0]


@pytest.fixture
def small_polygon_shape():
    # ~0.1° × 0.1° box near (-110, 38)
    return box(-110.1, 37.9, -110.0, 38.0)


@pytest.fixture
def large_polygon_shape():
    # 20° × 20° — guaranteed to exceed any reasonable max_cells
    return box(-130.0, 25.0, -110.0, 45.0)


@pytest.fixture
def dataset_bbox():
    return box(-115.0, 31.0, -102.0, 43.0)


# ---------------------------------------------------------------------------
# Job store fixtures


@pytest.fixture
def tmp_jobs_dir(tmp_path):
    d = tmp_path / "jobs"
    d.mkdir()
    return str(d)


@pytest.fixture
def fs_job_store(tmp_jobs_dir):
    return FileSystemJobStore(directory=tmp_jobs_dir)


@pytest.fixture
async def redis_job_store():
    store = RedisJobStore(os.environ.get("REDIS_URL", "redis://redis:6379"))
    yield store
    await store._client.flushdb()
    await store.close()


# ---------------------------------------------------------------------------
# Data reader mock


@pytest.fixture
def mock_data_reader(minimal_lookup_data):
    reader = AsyncMock(spec=DataReader)
    reader.read_json = AsyncMock(return_value=minimal_lookup_data)
    return reader


# ---------------------------------------------------------------------------
# Time series fixtures


@pytest.fixture
def base_series():
    index = [
        "0100",
        "0101",
        "0102",
        "0103",
        "0104",
        "0105",
        "0106",
        "0107",
        "0108",
        "0109",
    ]
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    return pd.Series(values, index=index)


@pytest.fixture
def constant_series():
    index = [
        "0100",
        "0101",
        "0102",
        "0103",
        "0104",
        "0105",
        "0106",
        "0107",
        "0108",
        "0109",
    ]
    return pd.Series([5.0] * 10, index=index)
