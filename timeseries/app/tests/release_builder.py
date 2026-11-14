"""Build small synthetic releases for tests.

A release here has the layout the release build writes (COGs named by the
time-to-band rule, overview.yml, and release-manifest.json last, listing every
other file with its size and checksum), without STAC: the API never reads it.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import rasterio
import yaml
from affine import Affine

from app.registry.compose import multihash
from app.registry.models import PinnedRelease
from app.vendor.timeaxis import Axis, ChunkLayout

DIGEST = "d" * 64


def build_release(
    root: Path,
    dataset_id: str,
    data: dict[str, np.ndarray],
    *,
    origin: str = "0001",
    step: str = "P1Y",
    precision: str = "year",
    chunk_size: int = 2,
    transform: tuple = (1.0, 0.0, -123.0, 0.0, -1.0, 45.0),
    crs: str = "EPSG:4326",
    nodata=None,
    date: str = "2026.01.01",
    declaration_digest: str = DIGEST,
    variable_fields: dict | None = None,
) -> PinnedRelease:
    """Write a release of `data` ({variable: array[time, row, col]}) under `root`."""
    release_id = f"{dataset_id}-r-{date}"
    path = root / release_id
    count, height, width = next(iter(data.values())).shape
    provisional = Axis("regular", precision, count, origin, step)
    axis = Axis.regular(origin, step, provisional.key(count - 1), precision)
    layout = ChunkLayout(dataset_id, axis, chunk_size)

    for variable_id, cube in data.items():
        for chunk in range(layout.chunk_count):
            first, last = layout.bounds(chunk)
            cog = path / layout.cog_path(variable_id, chunk)
            cog.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(
                cog,
                "w",
                driver="GTiff",
                width=width,
                height=height,
                count=last - first + 1,
                dtype=cube.dtype.name,
                crs=crs,
                transform=Affine(*transform),
                nodata=nodata,
            ) as dst:
                dst.write(cube[first : last + 1])
                for band, key in enumerate(layout.chunk_keys(chunk), start=1):
                    dst.set_band_description(band, key)

    west, south = transform[2], transform[5] + transform[4] * height
    east, north = transform[2] + transform[0] * width, transform[5]
    variables = {
        variable_id: {
            "title": f"{variable_id} title",
            "description": f"{variable_id} description",
            "unit": "mm",
            "category": "Precipitation",
            "data_type": cube.dtype.name,
            "nodata": nodata,
            "scale": 1.0,
            "offset": 0.0,
            **(variable_fields or {}).get(variable_id, {}),
        }
        for variable_id, cube in data.items()
    }
    overview = {
        "schema_version": "1.0.0",
        "release_id": release_id,
        "declaration_digest": declaration_digest,
        "dataset": {
            "id": dataset_id,
            "version": "1",
            "title": f"{dataset_id} title",
            "description": f"{dataset_id} description",
            "profile": "TemporalCubeDataset",
            "license": "CC-BY-4.0",
            "region_name": "Test region",
            "extent": {"bbox": [west, south, east, north]},
            "grid": {
                "code": crs,
                "shape": [height, width],
                "transform": list(transform),
            },
            "time": {
                "kind": "regular",
                "origin": origin,
                "step": step,
                "count": count,
                "chunk_size": chunk_size,
                "precision": precision,
                "calendar": "proleptic_gregorian",
                "timestep_meaning": "aggregation_period",
                "endpoint_inclusion": "inclusive",
            },
            "providers": [{"name": "Test producer", "roles": ["producer"]}],
        },
        "variables": variables,
    }
    (path / "overview.yml").write_text(yaml.safe_dump(overview, sort_keys=False))

    files = []
    for file in sorted(p for p in path.rglob("*") if p.is_file()):
        rel = file.relative_to(path).as_posix()
        roles = ["derived"] if rel == "overview.yml" else ["data"]
        files.append(
            {
                "path": rel,
                "roles": roles,
                "size": file.stat().st_size,
                "checksum": multihash(file),
            }
        )
    manifest = {
        "schema_version": "1.0.0",
        "release_id": release_id,
        "created": f"{date.replace('.', '-')}T00:00:00Z",
        "status": "complete",
        "dataset": {
            "id": dataset_id,
            "version": "1",
            "stac_entrypoint": "collection.json",
        },
        "producer": {"name": "test", "version": "0", "revision": "0"},
        "declaration": {
            "identity_profile": "openskope-release-declaration-v1",
            "digest_algorithm": "sha256",
            "digest": declaration_digest,
        },
        "sources": [
            {"id": v, "href": f"s3://test/{v}.tif", "checksum": "1220" + "0" * 64}
            for v in sorted(data)
        ],
        "files": files,
    }
    content = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
    (path / "release-manifest.json").write_bytes(content)
    return PinnedRelease(
        dataset=dataset_id,
        release_id=release_id,
        declaration_digest=declaration_digest,
        manifest_sha256=hashlib.sha256(content).hexdigest(),
    )


def write_pin(path: Path, release_root: Path, releases: list[PinnedRelease]) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "release_root": str(release_root),
                "releases": [r.model_dump() for r in releases],
            },
            sort_keys=False,
        )
    )
    return path
