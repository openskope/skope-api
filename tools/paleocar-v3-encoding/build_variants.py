"""Build the encoding-experiment variants (EXP-002, EXP-003, EXP-004).

Runs in the release build's image, with the build's GDAL and its COG creation
options (`cog_writer.cog_creation_options`), so a variant is what a release with
that declaration would write. Variants are disposable evidence (EXP-001).

Usage (see run.sh):
    python build_variants.py SOURCE OUT_DIR --scratch SCRATCH_DIR
        [--first-year 1401 --last-year 1900] [--blocks 64,128,256] [--only 'b128-*']

Each variant is a directory of chunk COGs plus `meta.json`, written last; a
directory without `meta.json` is incomplete and is rebuilt.
"""

from __future__ import annotations

import argparse
import fnmatch
import itertools
import json
import os
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from osgeo import gdal

from skope_release.cog_writer import cog_creation_options

gdal.UseExceptions()

UINT16_NODATA = 65535
BASELINE = {"block": 512, "interleave": "BAND", "chunk": 100, "dtype": "uint32"}


def variant_name(block: int, interleave: str, chunk: int, dtype: str) -> str:
    return f"b{block}-{interleave.lower()}-c{chunk}-{dtype}"


def matrix(blocks, interleaves, chunks, dtypes, baseline: bool) -> list[dict]:
    variants = [
        {"block": b, "interleave": i, "chunk": c, "dtype": d, "baseline": False}
        for b, i, c, d in itertools.product(blocks, interleaves, chunks, dtypes)
    ]
    if baseline:
        variants = [v for v in variants if any(v[k] != BASELINE[k] for k in BASELINE)]
        variants.insert(0, {**BASELINE, "baseline": True})
    for v in variants:
        v["name"] = variant_name(v["block"], v["interleave"], v["chunk"], v["dtype"])
    return variants


def selected_bands(source: gdal.Dataset, first: str | None, last: str | None) -> tuple[list[int], list[str]]:
    keys = [source.GetRasterBand(i).GetDescription() for i in range(1, source.RasterCount + 1)]
    lo = keys.index(first) if first else 0
    hi = keys.index(last) if last else len(keys) - 1
    return list(range(lo + 1, hi + 2)), keys[lo : hi + 1]


def write_scratch(source_path: str, path: Path, dtype: str, bands: list[int], keys: list[str], swath_mb: int) -> float:
    """One band-interleaved, stripped GeoTIFF of the selected bands, like the build's scratch files."""
    started = time.perf_counter()
    gdal.SetConfigOption("GDAL_NUM_THREADS", "1")  # bounds decode memory on the pixel-interleaved source
    src = gdal.Open(source_path)
    first = src.GetRasterBand(1)
    nodata = first.GetNoDataValue()
    width, height = src.RasterXSize, src.RasterYSize
    itemsize = gdal.GetDataTypeSizeBytes(first.DataType)
    rows = max(1, (swath_mb * 1024 * 1024) // (width * len(bands) * itemsize))
    block_height = first.GetBlockSize()[1]
    if rows >= block_height:
        rows -= rows % block_height
    rows = min(rows, height)
    gdal_type = gdal.GDT_UInt16 if dtype == "uint16" else first.DataType
    out_nodata = UINT16_NODATA if dtype == "uint16" else nodata
    tmp = path.with_suffix(".partial.tif")
    dst = gdal.GetDriverByName("GTiff").Create(
        str(tmp), width, height, len(bands), gdal_type,
        options=["INTERLEAVE=BAND", "TILED=NO", f"BLOCKYSIZE={rows}", "COMPRESS=ZSTD", "ZSTD_LEVEL=1",
                 "PREDICTOR=2", "BIGTIFF=YES"],
    )
    dst.SetGeoTransform(src.GetGeoTransform())
    dst.SetProjection(src.GetProjection())
    if src.GetMetadataItem("AREA_OR_POINT"):
        dst.SetMetadataItem("AREA_OR_POINT", src.GetMetadataItem("AREA_OR_POINT"))
    for i, key in enumerate(keys, start=1):
        band = dst.GetRasterBand(i)
        band.SetDescription(key)
        band.SetNoDataValue(out_nodata)
    for y in range(0, height, rows):
        n = min(rows, height - y)
        swath = src.ReadAsArray(0, y, width, n, band_list=bands).reshape(len(bands), n, width)
        if dtype == "uint16":
            valid = swath != nodata
            if valid.any() and swath[valid].max() >= UINT16_NODATA:
                raise SystemExit(f"a valid value at rows {y}..{y + n} does not fit uint16 below its nodata")
            swath = np.where(valid, swath, UINT16_NODATA).astype(np.uint16)
        for i in range(len(bands)):
            dst.GetRasterBand(i + 1).WriteArray(swath[i], 0, y)
    dst.Close()
    src = None
    tmp.rename(path)
    gdal.SetConfigOption("GDAL_NUM_THREADS", None)
    return time.perf_counter() - started


def creation_options(v: dict) -> list[str]:
    plan = SimpleNamespace(resolved_manifest=SimpleNamespace(release=SimpleNamespace(cog=SimpleNamespace(
        blocksize=v["block"], compress="ZSTD", interleave=v["interleave"], bigtiff="IF_SAFER", level=9))))
    variable = SimpleNamespace(encoding=SimpleNamespace(predictor=2, overview_resampling="AVERAGE"))
    return cog_creation_options(plan, variable)


def build_variant(v: dict, scratch: Path, keys: list[str], out: Path, threads: int) -> dict:
    target_dir = out / v["name"]
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True)
    options = creation_options(v)
    gdal.SetConfigOption("GDAL_NUM_THREADS", "ALL_CPUS")  # as the build's gdalio.configure()
    started = time.perf_counter()
    files = []
    for first in range(0, len(keys), v["chunk"]):
        chunk_keys = keys[first : first + v["chunk"]]
        name = f"ppt_annual--{chunk_keys[0]}--{chunk_keys[-1]}.tif"
        gdal.Translate(str(target_dir / name), str(scratch), format="COG",
                       bandList=list(range(first + 1, first + len(chunk_keys) + 1)),
                       creationOptions=[*options, f"NUM_THREADS={threads}"])
        files.append({"path": name, "keys": chunk_keys})
    seconds = time.perf_counter() - started
    gdal.SetConfigOption("GDAL_NUM_THREADS", None)
    check = gdal.Open(str(target_dir / files[0]["path"]))
    interleave = check.GetMetadataItem("INTERLEAVE", "IMAGE_STRUCTURE")
    block = check.GetRasterBand(1).GetBlockSize()
    if interleave != v["interleave"] or block != [v["block"], v["block"]]:
        raise SystemExit(f"{v['name']}: wrote INTERLEAVE={interleave}, block {block}")
    size = sum((target_dir / f["path"]).stat().st_size for f in files)
    meta = {**v, "files": files, "keys": len(keys), "first_key": keys[0], "last_key": keys[-1],
            "size_bytes": size, "build_seconds": seconds, "creation_options": options,
            "gdal": gdal.__version__, "threads": threads}
    (target_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source")
    p.add_argument("out", type=Path)
    p.add_argument("--scratch", type=Path, required=True)
    p.add_argument("--first-year", default="1401")
    p.add_argument("--last-year", default="1900")
    p.add_argument("--full-range", action="store_true", help="every band of the source (stage 2)")
    p.add_argument("--blocks", default="64,128,256")
    p.add_argument("--interleaves", default="BAND,TILE")
    p.add_argument("--chunks", default="25,100,250")
    p.add_argument("--dtypes", default="uint32,uint16")
    p.add_argument("--no-baseline", action="store_true")
    p.add_argument("--only", nargs="*", help="build only variants whose name matches one of these globs")
    p.add_argument("--swath-mb", type=int, default=512)
    p.add_argument("--threads", type=int, default=os.cpu_count())
    p.add_argument("--rebuild", action="store_true", help="rebuild variants that already have meta.json")
    args = p.parse_args()

    variants = matrix([int(b) for b in args.blocks.split(",")], args.interleaves.split(","),
                      [int(c) for c in args.chunks.split(",")], args.dtypes.split(","), not args.no_baseline)
    if args.only:
        variants = [v for v in variants if any(fnmatch.fnmatch(v["name"], g) for g in args.only)]

    source = gdal.Open(args.source)
    first, last = (None, None) if args.full_range else (args.first_year, args.last_year)
    bands, keys = selected_bands(source, first, last)
    source = None
    if not args.full_range:
        uneven = sorted({v["chunk"] for v in variants if len(keys) % v["chunk"]})
        if uneven:
            sys.exit(f"{len(keys)} timesteps don't divide into whole files of {uneven}; pick another window")
    window = f"{keys[0]}-{keys[-1]}"
    args.out.mkdir(parents=True, exist_ok=True)
    args.scratch.mkdir(parents=True, exist_ok=True)

    for v in variants:
        if (args.out / v["name"] / "meta.json").exists() and not args.rebuild:
            print(f"{v['name']}: done already", flush=True)
            continue
        scratch = args.scratch / f"ppt_annual-{window}-{v['dtype']}.tif"
        if not scratch.exists():
            print(f"scratch {scratch.name} ...", flush=True)
            print(f"  {write_scratch(args.source, scratch, v['dtype'], bands, keys, args.swath_mb):.0f} s", flush=True)
        meta = build_variant(v, scratch, keys, args.out, args.threads)
        print(f"{v['name']}: {len(meta['files'])} files, {meta['size_bytes'] / 1e9:.2f} GB, "
              f"{meta['build_seconds']:.0f} s", flush=True)


if __name__ == "__main__":
    main()
