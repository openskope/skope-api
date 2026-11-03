"""Benchmark the encoding variants: map views and extraction.

Map views: every tile of a map view, at the map client's concurrency, over
slider sequences of adjacent years. Two views show the whole region (`fit` and
`native`); one per extraction area shows the visualize map framed on that area
(`area-<name>`), the views users step through. Tiles are read the way
TiTiler reads them (rio-tiler, one open per tile), so this runs where rio-tiler
is installed: the TiTiler image, or a local virtualenv. `vrt` tiles open a
one-band `vrt://` view of the file; `direct` tiles open the file itself, as the
API asks TiTiler today, and rio-tiler's warp then reads every band in the file.

Extraction: the API's read loop over a polygon (mirrored from
`app.core.timeseries_processing`; `parity` checks the mirror against the real
function inside the API image). Needs only numpy and rasterio.

Every measurement runs in a fresh process, so peak memory is per operation and
no GDAL state carries over. Results are appended to a JSON-lines file.

Usage (see run.sh):
    python bench.py run --variants DIR --out FILE --setting B --ops extract --mode warm --rounds 3
    python bench.py parity --variants DIR        # inside the API image, cwd /code
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import math
import os
import platform
import random
import resource
import subprocess
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import rasterio
import rasterio.warp
import rasterio.windows
from rasterio.features import geometry_mask
from rasterio.warp import transform_geom
from rasterio.windows import Window

HERE = Path(__file__).resolve().parent
AREAS = HERE / "areas.geojson"

# The API's settings (deploy/settings, app/config.py, execute_timeseries_job).
API_MAX_CELLS = 1_000_000
API_THREADS = 10
API_ENV = {"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR", "VSI_CACHE": "TRUE"}

# The map client (skopeui MapLibre.client.vue): COG layer tileSize 128 inside the
# dataset bbox. MapLibre sends up to 16 requests at once; TiTiler serves them with
# as many processes as it has workers (deploy/Dockerfile.titiler; --tile-workers).
VIEWPORT = (1280, 800)
FIT_PADDING = 20
CLIENT_TILE_SIZE = 128
TILE_ZOOM_SHIFT = math.log2(512 / CLIENT_TILE_SIZE)  # MapLibre's covering zoom for this tileSize
# Visualize (pages/dataset/[id]/visualize): the map takes half the page width and the
# height minus 250 px, and `fitToGeoJson` frames the study area in it.
VISUALIZE_VIEWPORT = (640, 550)
AREA_FIT_PADDING = 30
MAX_FIT_ZOOM = 10
TILE_WORKERS = 2
DISPLAY_RANGE = (0.0, 1807.5)  # deploy/display/preferences.yml, ppt_annual

EXTRACT_RANGES = (10, 100, 500)
SEQUENCE = 12


# --- measurement helpers ------------------------------------------------------


def variant_files(variant: Path) -> list[Path]:
    meta = json.loads((variant / "meta.json").read_text())
    return [variant / f["path"] for f in meta["files"]]


def evict(paths: list[Path]) -> None:
    """Drop the files' pages from the page cache (Linux; no privileges needed)."""
    if not hasattr(os, "posix_fadvise"):
        raise SystemExit("cold reads need posix_fadvise (Linux); run warm elsewhere")
    for path in paths:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


def peak_rss_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


def proc_io() -> dict | None:
    try:
        text = Path("/proc/self/io").read_text()
    except OSError:
        return None
    return {k: int(v) for k, v in (line.split(": ") for line in text.splitlines())}


def disk_reads(path: Path) -> dict | None:
    """Reads completed and sectors read on the block device holding `path`, from /proc/diskstats."""
    try:
        dev = path.stat().st_dev
        lines = Path("/proc/diskstats").read_text().splitlines()
    except OSError:
        return None
    for line in lines:
        f = line.split()
        if int(f[0]) == os.major(dev) and int(f[1]) == os.minor(dev):
            return {"reads": int(f[3]), "sectors": int(f[5])}
    return None


def delta(after: dict | None, before: dict | None) -> dict | None:
    if after is None or before is None:
        return None
    return {k: after[k] - before[k] for k in after}


def digest(*arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:16]


# --- extraction ------------------------------------------------------


def load_areas() -> dict[str, dict]:
    collection = json.loads(AREAS.read_text())
    return {f["properties"]["name"]: f["geometry"] for f in collection["features"]}


def extraction_window(geometry: dict, src) -> tuple[Window, np.ndarray]:
    """resolve_spatial_window and calculate_spatial_coverage, as the API computes them."""
    shape = transform_geom("EPSG:4326", src.crs, geometry)
    xs, ys = zip(*_coords(shape))
    window = rasterio.windows.from_bounds(min(xs), min(ys), max(xs), max(ys), transform=src.transform)
    window = window.round_lengths().round_offsets()
    window = Window(window.col_off, window.row_off, max(1, window.width), max(1, window.height))
    wt = rasterio.windows.transform(window, src.transform)
    out_shape = (int(window.height), int(window.width))
    mask = geometry_mask([shape], transform=wt, invert=True, out_shape=out_shape)
    if mask.sum() == 0:
        mask = geometry_mask([shape], transform=wt, invert=True, out_shape=out_shape, all_touched=True)
    return window, mask


def _coords(geometry: dict):
    def walk(c):
        if isinstance(c[0], (int, float)):
            yield c
        else:
            for x in c:
                yield from walk(x)
    yield from walk(geometry["coordinates"])


def extract_file(uri: str, bands: list[int], window: Window, mask: np.ndarray, chunk_size: int):
    """Mirror of `extract_summarystat_timeseries` (app/core/timeseries_processing.py)."""
    means, medians = [], []
    with rasterio.Env(**API_ENV), rasterio.open(uri) as src:
        for i in range(0, len(bands), chunk_size):
            data = src.read(bands[i : i + chunk_size], window=window).astype(np.float32)
            if src.nodata is not None:
                data[data == src.nodata] = np.nan
            data[:, ~mask] = np.nan
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                means.extend(np.nanmean(data, axis=(1, 2)))
                medians.extend(np.nanmedian(data, axis=(1, 2)))
    return uri, np.array(means), np.array(medians)


def file_mapping(variant: Path, n_years: int | None) -> list[tuple[str, list[int]]]:
    """The first `n_years` timesteps (all when None), grouped by file as the API's lookup groups them."""
    meta = json.loads((variant / "meta.json").read_text())
    mapping, remaining = [], n_years if n_years is not None else meta["keys"]
    for f in meta["files"]:
        if remaining <= 0:
            break
        take = min(remaining, len(f["keys"]))
        mapping.append((str(variant / f["path"]), list(range(1, take + 1))))
        remaining -= take
    return mapping


def run_extraction(task: dict) -> dict:
    variant = Path(task["variant"])
    geometry = load_areas()[task["area"]]
    mapping = file_mapping(variant, task["years"])
    with rasterio.open(mapping[0][0]) as src:
        window, mask = extraction_window(geometry, src)
    chunk = API_MAX_CELLS // (int(window.width) * int(window.height))

    def once():
        with ThreadPoolExecutor(API_THREADS) as pool:
            results = list(pool.map(lambda fb: extract_file(fb[0], fb[1], window, mask, chunk), mapping))
        return np.concatenate([r[1] for r in results]), np.concatenate([r[2] for r in results])

    if task["mode"] == "warm":
        once()
    else:
        evict(variant_files(variant))
    io0, disk0 = proc_io(), disk_reads(variant / "meta.json")
    started = time.perf_counter()
    means, medians = once()
    seconds = time.perf_counter() - started
    return {"seconds": seconds, "io": delta(proc_io(), io0), "disk": delta(disk_reads(variant / "meta.json"), disk0),
            "window": [int(window.width), int(window.height)], "cells": int(mask.sum()), "files": len(mapping),
            "concurrency": min(API_THREADS, len(mapping)), "values": len(means),
            "digest": digest(np.round(means, 3), np.round(medians, 3))}


# --- map views -------------------------------------------------------


def _mercator_y(lat: float) -> float:
    s = math.sin(math.radians(lat))
    return 0.5 - math.log((1 + s) / (1 - s)) / (4 * math.pi)


def _fit(bbox: tuple[float, float, float, float], viewport: tuple[int, int], padding: int,
         max_zoom: float = math.inf) -> dict:
    """The covering tile zoom and centre of MapLibre's fitBounds. A point's zero-size bbox gets `max_zoom`."""
    west, south, east, north = bbox
    w, h = viewport
    dx, dy = (east - west) / 360, _mercator_y(south) - _mercator_y(north)
    zooms = [max_zoom]
    if dx:
        zooms.append(math.log2((w - 2 * padding) / (dx * 512)))
    if dy:
        zooms.append(math.log2((h - 2 * padding) / (dy * 512)))
    return {"zoom": round(min(zooms) + TILE_ZOOM_SHIFT), "center": ((west + east) / 2, (south + north) / 2),
            "viewport": viewport}


def map_views(bounds: tuple[float, float, float, float], native_res_deg: float) -> dict[str, dict]:
    """The tiles MapLibre requests for the initial (fit-to-bbox) view, a native-resolution view, and the visualize
    map framed on each extraction area."""
    import morecantile

    tms = morecantile.tms.get("WebMercatorQuad")
    west, south, east, north = bounds
    views = {"fit": _fit(bounds, VIEWPORT, FIT_PADDING)}

    # Native: the first zoom whose 256-pixel tiles are finer than the grid, where reads hit full resolution.
    lat = (south + north) / 2
    native_m = native_res_deg * 111_320 * math.cos(math.radians(lat))
    z = 0
    while 40_075_016.7 / 2**z / 256 > native_m:
        z += 1
    views["native"] = {"zoom": z, "center": ((west + east) / 2, lat), "viewport": VIEWPORT}

    for name, geometry in load_areas().items():
        xs, ys = zip(*_coords(geometry))
        views[f"area-{name}"] = _fit((min(xs), min(ys), max(xs), max(ys)), VISUALIZE_VIEWPORT, AREA_FIT_PADDING,
                                     MAX_FIT_ZOOM)

    for view in views.values():
        z, (lon, lat) = view["zoom"], view["center"]
        n = 2**z
        cx, cy = (lon + 180) / 360 * n, _mercator_y(lat) * n
        w, h = view["viewport"]
        half_x, half_y = w / CLIENT_TILE_SIZE / 2, h / CLIENT_TILE_SIZE / 2
        in_bbox = {(t.x, t.y) for t in tms.tiles(west, south, east, north, [z])}
        view["tiles"] = sorted(
            (x, y, z)
            for x in range(math.floor(cx - half_x), math.floor(cx + half_x) + 1)
            for y in range(math.floor(cy - half_y), math.floor(cy + half_y) + 1)
            if (x, y) in in_bbox
        )
    return views


def _tile(args) -> bytes:
    """One tile request as TiTiler serves it: open, read, rescale, render."""
    from rio_tiler.errors import TileOutsideBounds
    from rio_tiler.io import Reader

    (x, y, z), path, band, tile_path = args
    if tile_path == "vrt":  # a one-band view of the file: the warp reads only that band
        path, band = f"vrt://{path}?bands={band}", 1
    try:
        with Reader(str(path)) as reader:
            img = reader.tile(x, y, z, indexes=band)
    except TileOutsideBounds:
        return b""
    img.rescale(in_range=(DISPLAY_RANGE,))
    img.render(img_format="PNG")
    return img.array.data.tobytes() + np.ma.getmaskarray(img.array).tobytes()


def _worker_ready(_) -> int:
    import rio_tiler.io  # noqa: F401 - imported before timing starts

    return os.getpid()


def run_view(task: dict) -> dict:
    """Tile reads don't run in parallel on threads (measured: 16 threads, no speedup), so
    each TiTiler worker process is one process here."""
    variant = Path(task["variant"])
    meta = json.loads((variant / "meta.json").read_text())
    where = {key: (variant / f["path"], i + 1) for f in meta["files"] for i, key in enumerate(f["keys"])}
    with rasterio.open(variant / meta["files"][0]["path"]) as src:
        bounds = tuple(rasterio.warp.transform_bounds(src.crs, "EPSG:4326", *src.bounds))
        res = src.res[0]
    tiles = map_views(bounds, res)[task["view"]]["tiles"]
    all_keys = [k for f in meta["files"] for k in f["keys"]]
    start = len(all_keys) // 2 - task["steps"] // 2
    steps = all_keys[start : start + task["steps"]]

    def sequence(pool) -> tuple[list[float], str]:
        times, h = [], hashlib.sha256()
        for key in steps:
            path, band = where[key]
            started = time.perf_counter()
            for blob in pool.map(_tile, [(t, path, band, task["tile_path"]) for t in tiles]):
                h.update(blob)
            times.append(time.perf_counter() - started)
        return times, h.hexdigest()[:16]

    workers = task["tile_workers"]
    with ProcessPoolExecutor(workers) as pool:
        list(pool.map(_worker_ready, range(workers * 4)))
        if task["mode"] == "warm":
            sequence(pool)
        else:
            evict(variant_files(variant))
        disk0 = disk_reads(variant / "meta.json")
        times, seq_digest = sequence(pool)
        disk1 = disk_reads(variant / "meta.json")
    return {"seconds": sum(times) / len(times), "step_seconds": times, "io": None, "disk": delta(disk1, disk0),
            "tiles": len(tiles), "zoom": tiles[0][2] if tiles else None, "concurrency": workers,
            "digest": seq_digest}


# --- orchestration ------------------------------------------------------------


def one(task: dict) -> None:
    base = peak_rss_bytes()
    result = run_extraction(task) if task["op"] == "extract" else run_view(task)
    result["peak_rss_delta"] = peak_rss_bytes() - base
    children = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    result["peak_rss_worker"] = children if sys.platform == "darwin" else children * 1024
    print(json.dumps(result))


def environment() -> dict:
    env = {"python": platform.python_version(), "platform": platform.platform(), "cpus": os.cpu_count(),
           "rasterio": rasterio.__version__, "gdal": rasterio.__gdal_version__,
           "gdal_cachemax": os.environ.get("GDAL_CACHEMAX")}
    try:
        import rio_tiler

        env["rio_tiler"] = rio_tiler.__version__
    except ImportError:
        pass
    return env


def tasks_for(variant: Path, ops: list[str], mode: str, first_round: bool, tile_workers: int) -> list[dict]:
    meta = json.loads((variant / "meta.json").read_text())
    tasks = []
    if "extract" in ops:
        ranges = [r for r in EXTRACT_RANGES if r < meta["keys"]] + [None]  # None = every timestep
        for area in load_areas():
            for years in ranges:
                tasks.append({"op": "extract", "area": area, "years": years})
    if "views" in ops:
        for view in ("fit", "native", *(f"area-{area}" for area in load_areas())):
            tasks.append({"op": "view", "view": view, "steps": SEQUENCE, "tile_path": "vrt",
                          "tile_workers": tile_workers})
            if first_round and meta["baseline"] and view in ("fit", "native"):  # the release's tiles before the one-band fix, for the record
                tasks.append({"op": "view", "view": view, "steps": SEQUENCE, "tile_path": "direct",
                              "tile_workers": tile_workers})
    for t in tasks:
        t.update(variant=str(variant), name=variant.name, mode=mode)
    return tasks


def run(args) -> None:
    variants = sorted(p.parent for p in Path(args.variants).glob("*/meta.json"))
    if args.only:
        variants = [v for v in variants if any(fnmatch.fnmatch(v.name, g) for g in args.only)]
    if not variants:
        sys.exit(f"no built variants under {args.variants}")
    env = environment()
    rng = random.Random(args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    metas = {}
    for v in variants:
        m = json.loads((v / "meta.json").read_text())
        metas[v.name] = {k: m[k] for k in ("block", "interleave", "chunk", "dtype", "baseline", "keys", "first_key",
                                           "last_key", "size_bytes", "build_seconds", "gdal")}
        metas[v.name]["file_count"] = len(m["files"])
    for round_ in range(1, args.rounds + 1):
        order = variants[:]
        rng.shuffle(order)
        for variant in order:
            for task in tasks_for(variant, args.ops, args.mode, round_ == 1, args.tile_workers):
                proc = subprocess.run([sys.executable, __file__, "one", json.dumps(task)],
                                      capture_output=True, text=True, timeout=args.timeout)
                if proc.returncode != 0:
                    record = {"error": proc.stderr.strip().splitlines()[-1:]}
                else:
                    record = json.loads(proc.stdout.strip().splitlines()[-1])
                record.update(task=task, variant_meta=metas[variant.name], setting=args.setting, round=round_,
                              env=env, at=time.time())
                with out.open("a") as fh:
                    fh.write(json.dumps(record) + "\n")
                label = task.get("area") or f"{task.get('view')}x{task.get('steps')}"
                shown = f"{record['seconds']:.3f} s" if "seconds" in record else record["error"]
                print(f"[{args.setting}/{args.mode} r{round_}] {variant.name} {task['op']} {label} "
                      f"{task.get('years', '')} {shown}", flush=True)


def parity(args) -> None:
    """The mirror against the API's own function, on one variant (run inside the API image, cwd /code)."""
    sys.path.insert(0, os.getcwd())
    from app.core.timeseries_processing import extract_summarystat_timeseries

    variant = sorted(p.parent for p in Path(args.variants).glob("*/meta.json"))[0]
    uri, bands = file_mapping(variant, None)[0]
    with rasterio.open(uri) as src:
        window, mask = extraction_window(load_areas()["medium"], src)
    chunk = API_MAX_CELLS // (int(window.width) * int(window.height))
    _, m1, d1 = extract_file(uri, bands, window, mask, chunk)
    _, m2, d2 = extract_summarystat_timeseries(uri, bands, window, mask, chunk)
    same = np.array_equal(m1, m2, equal_nan=True) and np.array_equal(d1, d2, equal_nan=True)
    print(f"{variant.name}: mirror {'matches' if same else 'DIFFERS FROM'} the API's extraction")
    sys.exit(0 if same else 1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--variants", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--setting", required=True, help="A (native), B (Docker, bind mount), C (staging simulation)")
    r.add_argument("--ops", nargs="+", choices=["extract", "views"], required=True)
    r.add_argument("--mode", choices=["warm", "cold"], required=True)
    r.add_argument("--rounds", type=int, default=3)
    r.add_argument("--only", nargs="*")
    r.add_argument("--seed", type=int, default=1)
    r.add_argument("--tile-workers", type=int, default=TILE_WORKERS, help="TiTiler worker processes to simulate")
    r.add_argument("--timeout", type=int, default=900)
    sub.add_parser("one").add_argument("task")
    sub.add_parser("parity").add_argument("--variants", required=True)
    args = p.parse_args()
    if args.command == "one":
        one(json.loads(args.task))
    elif args.command == "run":
        run(args)
    else:
        parity(args)


if __name__ == "__main__":
    main()
