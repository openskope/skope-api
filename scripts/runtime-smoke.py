"""Synthetic raster checks inside the real API/TiTiler/Redis containers."""

import importlib.util
import json
from pathlib import Path
import sys
import time

import httpx
import numpy as np
from rasterio.io import MemoryFile

sys.path.insert(0, "/code")
VALUES = [0, 2, None, 4, 6]
OPTIONS = [{"name": "raw", "smoother": {"type": "NoSmoother"}}]


def prepare():
    spec = importlib.util.spec_from_file_location(
        "release_builder", "/release_builder.py"
    )
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    root = Path("/fixtures")
    cube = np.stack(
        [np.full((5, 5), 65535 if v is None else v, dtype="uint16") for v in VALUES]
    )
    pinned = builder.build_release(
        root / "releases",
        "runtime_test",
        {"ppt": cube},
        nodata=65535,
        creation_options={
            "driver": "COG",
            "BLOCKSIZE": 128,
            "INTERLEAVE": "TILE",
            "COMPRESS": "ZSTD",
        },
    )
    builder.write_pin(root / "pin.json", Path("/releases"), [pinned])
    (root / "preferences.json").write_text(
        json.dumps(
            {
                "datasets": {
                    "runtime_test": {
                        "order": 10,
                        "variables": {
                            "ppt": {
                                "order": 10,
                                "palette": "skope-precip",
                                "range": [0, 6],
                            }
                        },
                    }
                }
            }
        )
    )
    for path in root.rglob("*"):
        path.chmod(0o755 if path.is_dir() else 0o644)


def check():
    with httpx.Client(base_url="http://server:8000", timeout=10) as client:
        deadline = time.monotonic() + 90
        while True:
            try:
                response = client.get("/metadata")
                response.raise_for_status()
                break
            except httpx.HTTPError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(1)

        # The polygon covers exactly one pixel in the test release.
        polygon = {
            "type": "Polygon",
            "coordinates": [
                [
                    [-122, 43],
                    [-121, 43],
                    [-121, 44],
                    [-122, 44],
                    [-122, 43],
                ]
            ],
        }
        payload = {
            "dataset_id": "runtime_test",
            "variable_id": "ppt",
            "selected_area": polygon,
            "zonal_statistic": "mean",
            "transform": {"type": "NoTransform"},
            "requested_series_options": OPTIONS,
            "time_range": {"gte": "0001", "lte": "0005"},
        }
        response = client.post("/timeseries/extract", json=payload)
        assert response.status_code == 202, response.text
        job_id = response.json()["job_id"]
        deadline = time.monotonic() + 30
        while True:
            response = client.get(f"/timeseries/status/{job_id}")
            response.raise_for_status()
            job = response.json()
            if job["status"] == "SUCCESS":
                break
            assert job["status"] in ("PENDING", "PROCESSING"), job
            assert time.monotonic() < deadline, job
            time.sleep(0.2)
        assert "base_series" not in job
        assert job["result"]["series"][0]["values"] == VALUES, job
        response = client.post(
            "/timeseries/analyze",
            json={
                "extraction_id": job_id,
                "transform": {"type": "NoTransform"},
                "requested_series_options": OPTIONS,
                "time_range": {"gte": "0002", "lte": "0004"},
            },
        )
        response.raise_for_status()
        assert response.json()["series"][0]["values"] == [2, None, 4], response.text

        # Real API -> TiTiler -> one-band VRT reads, including an empty year.
        import yaml

        palettes = yaml.safe_load(Path("/code/config/display/palettes.yml").read_text())
        stops = palettes["skope-precip"]["colors"]
        for timestep, color in (
            ("0001", stops[0]),
            ("0002", stops[5]),
            ("0003", None),
            ("0004", stops[10]),
            ("0005", stops[-1]),
        ):
            response = client.get(f"/tiles/runtime_test/ppt/{timestep}/6/10/23")
            response.raise_for_status()
            with MemoryFile(response.content) as memory, memory.open() as raster:
                pixels = raster.read()
            assert pixels.shape[0] == 4, response.headers
            opaque = pixels[3] == 255
            if color is None:
                assert not pixels[3].any(), "nodata must remain transparent"
            else:
                expected = [int(color[i : i + 2], 16) for i in (1, 3, 5)]
                assert opaque.any(), "the tile must contain the test raster"
                assert np.all(
                    pixels[:3, opaque] == np.array(expected)[:, None]
                ), timestep
        print(
            "Real runtime passed: custom palette, VRT bands, nodata, extraction and analysis."
        )


if __name__ == "__main__":
    {"prepare": prepare, "check": check}[sys.argv[1]]()
