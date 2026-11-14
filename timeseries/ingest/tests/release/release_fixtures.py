"""Synthetic datasets for release tests.

`make_temporal` and `make_static` write a small source raster plus the
dataset's `curated.yml` and `source-manifest.yml` under `tmp_path/datasets/`.
Keyword overrides let a test break exactly one thing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
from osgeo import gdal, osr

gdal.UseExceptions()

UINT32_NODATA = 4294967295
FLOAT32_NODATA = -9999.0
GT = (-115.0, 0.01, 0.0, 43.0, 0.0, -0.01)
CREATED = "2026-10-07T12:00:00Z"


def write_source(path: Path, keys: list[str], *, width=40, height=30, dtype=gdal.GDT_UInt32, nodata=UINT32_NODATA,
                 seed=0, categories=None, geotransform=GT, epsg=4269, empty=()) -> np.ndarray:
    """Bands whose key is in `empty` hold only nodata."""
    rng = np.random.default_rng(seed)
    ds = gdal.GetDriverByName("GTiff").Create(str(path), width, height, len(keys), dtype, options=["INTERLEAVE=PIXEL", "TILED=YES", "BLOCKXSIZE=16", "BLOCKYSIZE=16"])
    ds.SetGeoTransform(geotransform)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    ds.SetProjection(srs.ExportToWkt())
    ds.SetMetadataItem("AREA_OR_POINT", "Area")
    cube = []
    for i, key in enumerate(keys, start=1):
        if categories:
            data = rng.choice(list(categories), size=(height, width)).astype(np.uint8)
        elif dtype == gdal.GDT_Float32:
            data = rng.normal(0, 2, size=(height, width)).astype(np.float32)
        else:
            data = rng.integers(0, 4000, size=(height, width)).astype(np.uint32)
        if nodata is not None:
            data[:3, :3] = nodata
            if key in empty:
                data[:] = nodata
        band = ds.GetRasterBand(i)
        band.WriteArray(data)
        if nodata is not None:
            band.SetNoDataValue(nodata)
        band.SetDescription(key)
        cube.append(data)
    ds = None
    return np.stack(cube)


def years(start: int, count: int) -> list[str]:
    return [f"{start + i:04d}" for i in range(count)]


def curated_doc(dataset_id: str, variables: list[str], *, profile="TemporalCubeDataset", origin="0100", end="0104", **overrides) -> dict:
    doc = {
        "schema_version": "0.1.0",
        "id": dataset_id,
        "version": "1",
        "profile": profile,
        "title": f"Synthetic {dataset_id}",
        "description": "A synthetic dataset for tests.",
        "license": "CC-BY-4.0",
        "region_name": "Testland",
        "providers": [{"name": "Test Lab", "roles": ["producer", "licensor"]}],
        "publications": [{"citation": "Author, A. 2014. A paper.", "doi": "10.1038/ncomms6618"}],
        "lineage": "Made up for tests.",
        "uncertainty": {"summary": "None; synthetic.", "methodology_href": None},
        "links": [{"rel": "via", "href": "https://example.org/source"}],
        "variables": [
            {"id": v, "title": f"Variable {v}", "description": f"Synthetic {v}.", "unit": "mm", "category": "precipitation"}
            for v in variables
        ],
    }
    if profile == "TemporalCubeDataset":
        doc["temporal"] = {
            "calendar": "proleptic_gregorian",
            "precision": "year",
            "origin": origin,
            "end": end,
            "step": "P1Y",
            "endpoint_inclusion": "inclusive",
            "timestep_meaning": "aggregation_period",
            "description": "Calendar-year totals.",
        }
    else:
        doc["temporal_extent"] = ["2000-02-11T00:00:00Z", "2000-02-22T23:59:59Z"]
    doc.update(overrides)
    return doc


def manifest_doc(dataset_id: str, sources: dict[str, str], *, chunk_size=2, data_type="uint32", nodata=UINT32_NODATA,
                 static=False, resampling="AVERAGE", predictor=2, **overrides) -> dict:
    doc = {
        "schema_version": "0.1.0",
        "dataset": dataset_id,
        "release": {
            "created": CREATED,
            "cog": {"blocksize": 256, "interleave": "BAND", "compress": "ZSTD", "level": 9, "bigtiff": "IF_SAFER"},
        },
        "variables": [
            {
                "id": vid,
                "source": {"uri": uri, **({"band": 1} if static else {"bands": "descriptions"})},
                "encoding": {
                    "data_type": data_type,
                    "nodata": nodata,
                    "scale": 1.0,
                    "offset": 0.0,
                    "predictor": predictor,
                    "overview_resampling": resampling,
                },
            }
            for vid, uri in sources.items()
        ],
    }
    if not static:
        doc["release"]["chunk_size"] = chunk_size
    doc.update(overrides)
    return doc


def write_dataset(root: Path, dataset_id: str, curated: dict, manifest: dict) -> Path:
    directory = root / "datasets" / dataset_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "curated.yml").write_text(yaml.safe_dump(curated, sort_keys=False, allow_unicode=True))
    (directory / "source-manifest.yml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    return directory


