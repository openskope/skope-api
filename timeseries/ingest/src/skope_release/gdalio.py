"""GDAL access shared by preflight, the COG writer, and byte inspection."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from urllib.parse import urlparse

from osgeo import gdal, osr

gdal.UseExceptions()
osr.UseExceptions()

GDAL_TO_STAC_TYPE = {
    gdal.GDT_Byte: "uint8",
    gdal.GDT_Int8: "int8",
    gdal.GDT_UInt16: "uint16",
    gdal.GDT_Int16: "int16",
    gdal.GDT_UInt32: "uint32",
    gdal.GDT_Int32: "int32",
    gdal.GDT_Float32: "float32",
    gdal.GDT_Float64: "float64",
}
STAC_TO_GDAL_TYPE = {v: k for k, v in GDAL_TO_STAC_TYPE.items()}
INTEGER_RANGES = {
    "uint8": (0, 2**8 - 1),
    "int8": (-(2**7), 2**7 - 1),
    "uint16": (0, 2**16 - 1),
    "int16": (-(2**15), 2**15 - 1),
    "uint32": (0, 2**32 - 1),
    "int32": (-(2**31), 2**31 - 1),
}

_CHUNK = 8 * 1024 * 1024
SWATH_BUDGET_BYTES = 1536 * 1024 * 1024


def swath_rows(width: int, bands: int, itemsize: int, block_height: int, height: int) -> int:
    """Rows per full-width read of every band within SWATH_BUDGET_BYTES.

    Reading all bands per swath decodes each tile of a pixel-interleaved
    source once, rather than once per band.
    """
    per_row = width * bands * itemsize
    rows = max(1, SWATH_BUDGET_BYTES // per_row)
    if rows >= block_height:
        rows -= rows % block_height  # whole source blocks: each tile is decoded once
    return min(rows, height)


def configure() -> None:
    """Process-wide GDAL settings for release builds."""
    # No PAM sidecars: statistics must live in the COG itself (COG-005).
    gdal.SetConfigOption("GDAL_PAM_ENABLED", "NO")
    gdal.SetConfigOption("GDAL_CACHEMAX", "512")
    gdal.SetConfigOption("GDAL_NUM_THREADS", "ALL_CPUS")


def gdal_version() -> str:
    return gdal.__version__


def read_path(uri: str, mirror: Path | None) -> str:
    """Where GDAL reads a source URI. A mirror holds `<bucket>/<key>` copies.

    The read path is operational and never part of the declaration; the
    source checksum proves the mirror copy is the declared object.
    """
    parsed = urlparse(uri)
    if parsed.scheme == "s3":
        if mirror is not None:
            local = mirror_path(uri, mirror)
            if local.is_file():
                return str(local)
        return f"/vsis3/{parsed.netloc}{parsed.path}"
    if parsed.scheme in {"http", "https"}:
        return f"/vsicurl/{uri}"
    if parsed.scheme in {"", "file"}:
        return parsed.path if parsed.scheme == "file" else uri
    raise ValueError(f"unsupported source URI scheme: {uri}")


def mirror_path(uri: str, mirror: Path) -> Path | None:
    """Where a mirror holds a source URI's copy, or None for a URI it cannot hold."""
    parsed = urlparse(uri)
    if parsed.scheme == "s3":
        return mirror / parsed.netloc / parsed.path.lstrip("/")
    return None


def _blocks(path: str | os.PathLike):
    """Stream a local or `/vsi` path in blocks."""
    path = str(path)
    if path.startswith("/vsi"):
        handle = gdal.VSIFOpenL(path, "rb")
        if handle is None:
            raise OSError(f"cannot open {path}")
        try:
            while True:
                block = gdal.VSIFReadL(1, _CHUNK, handle)
                if not block:
                    break
                yield block
        finally:
            gdal.VSIFCloseL(handle)
    else:
        with open(path, "rb") as handle:
            yield from iter(lambda: handle.read(_CHUNK), b"")


def sha256_multihash(path: str | os.PathLike) -> str:
    """Stream a local or `/vsi` path and return its SHA-256 multihash (OBS-011)."""
    digest = hashlib.sha256()
    for block in _blocks(path):
        digest.update(block)
    return "1220" + digest.hexdigest()


def copy_with_multihash(source: str | os.PathLike, target: Path) -> str:
    """Stream `source` into the local file `target`; return the copy's SHA-256 multihash."""
    digest = hashlib.sha256()
    with open(target, "wb") as out:
        for block in _blocks(source):
            digest.update(block)
            out.write(block)
    return "1220" + digest.hexdigest()


def crs_code(srs: osr.SpatialReference) -> str | None:
    srs = srs.Clone()
    srs.AutoIdentifyEPSG()
    authority, code = srs.GetAuthorityName(None), srs.GetAuthorityCode(None)
    return f"{authority}:{code}" if authority and code else None


def wgs84_bbox(srs_wkt: str, geotransform, width: int, height: int) -> tuple[float, float, float, float]:
    """WGS 84 bbox of the grid's outer edge, from its corners (Section 20.3)."""
    src = osr.SpatialReference()
    src.ImportFromWkt(srs_wkt)
    src.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    dst = osr.SpatialReference()
    dst.ImportFromEPSG(4326)
    dst.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    transform = osr.CoordinateTransformation(src, dst)
    gt = geotransform
    corners = [
        (gt[0] + gt[1] * px + gt[2] * py, gt[3] + gt[4] * px + gt[5] * py)
        for px, py in ((0, 0), (width, 0), (0, height), (width, height))
    ]
    points = [transform.TransformPoint(x, y)[:2] for x, y in corners]
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    return (min(xs), min(ys), max(xs), max(ys))


def same_crs(wkt_a: str, wkt_b: str) -> bool:
    a, b = osr.SpatialReference(), osr.SpatialReference()
    a.ImportFromWkt(wkt_a)
    b.ImportFromWkt(wkt_b)
    return bool(a.IsSame(b))
