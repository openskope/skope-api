"""Reopen written COGs and freeze a `FinalObservation` (COG-008, OBS-012).

Every fact published in STAC comes from here, measured from the bytes:
structure, grid, encoding, band descriptions, embedded statistics, size, and
checksum. Statistics are recomputed from pixels and compared with the embedded
values (COG-004, AT-011). A COG is checksummed only after it validates (COG-009).
"""

from __future__ import annotations

import logging
import math
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from osgeo import gdal, gdal_array
from osgeo_utils.samples.validate_cloud_optimized_geotiff import validate as validate_cog

from . import gdalio
from .cog_writer import workers
from .findings import Report
from .plan import (
    _OBSERVATION_ISSUER,
    AssetObservation,
    BandObservation,
    FinalObservation,
    PlannedVariable,
    ValidatedBuildPlan,
    require_plan,
)

log = logging.getLogger(__name__)

# Each inspection worker holds one COG in memory; current COGs are under 1 GB.
IN_MEMORY_LIMIT = 1 << 30
STAT_RELATIVE_TOLERANCE = 1e-10  # GDAL serializes ~13 significant digits (Section 20.3)
VALID_PERCENT_TOLERANCE = 0.005
LOSSLESS = {"ZSTD", "DEFLATE", "LZW"}


def _close(a: float, b: float, rel: float) -> bool:
    return a == b or abs(a - b) <= rel * max(abs(a), abs(b), 1e-300)


def _recompute(band, nodata) -> tuple[float, float, float, float, float] | None:
    values = band.ReadAsArray()
    total = values.size
    if nodata is not None:
        values = values[values != nodata]
    if np.issubdtype(values.dtype, np.floating):
        values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    as64 = values.astype(np.float64)
    return (float(as64.min()), float(as64.max()), float(as64.mean()), float(as64.std()), 100.0 * values.size / total)


def inspect_asset(
    path: Path,
    rel_path: str,
    *,
    plan: ValidatedBuildPlan,
    variable: PlannedVariable,
    chunk: int | None,
    expected_names: list[str],
    report: Report,
) -> AssetObservation | None:
    """Inspect one written COG from an in-memory copy of its bytes.

    The validator and the statistics read each band's blocks in turn, which on a
    tile-interleaved COG means scattered reads across the whole file. Reading the
    file once, in order, into memory makes those reads cheap, and the checksum is
    then of exactly the bytes validated. A file above IN_MEMORY_LIMIT is read in
    place.
    """
    if path.stat().st_size > IN_MEMORY_LIMIT:
        return _inspect(str(path), path, rel_path, plan=plan, variable=variable, chunk=chunk,
                        expected_names=expected_names, report=report)
    copy = f"/vsimem/inspect-{os.getpid()}-{rel_path.replace('/', '_')}"
    if gdal.CopyFile(str(path), copy) != 0:
        raise OSError(f"cannot copy {path} into memory")
    try:
        return _inspect(copy, path, rel_path, plan=plan, variable=variable, chunk=chunk,
                        expected_names=expected_names, report=report)
    finally:
        gdal.Unlink(copy)


def _inspect(
    source: str,
    path: Path,
    rel_path: str,
    *,
    plan: ValidatedBuildPlan,
    variable: PlannedVariable,
    chunk: int | None,
    expected_names: list[str],
    report: Report,
) -> AssetObservation | None:
    """Inspect the COG bytes at `source`, a copy of `path` or `path` itself."""
    ctx = {"dataset": plan.dataset_id, "variable": variable.id, "path": rel_path}
    if chunk is not None:
        ctx["chunk"] = plan.layout.item_id(chunk)
    start = len(report.errors)

    errors, warnings, _ = validate_cog(source, full_check=True)
    for message in errors:
        report.add("COG-001", message, **ctx)
    for message in warnings:
        report.warn("COG-001", message, **ctx)
    # A sidecar next to the file to be published fails COG-005; it never sits next to the copy.
    for sidecar in (path.with_suffix(path.suffix + ".aux.xml"), path.with_suffix(path.suffix + ".ovr")):
        if sidecar.exists():
            report.add("COG-005", f"sidecar {sidecar.name} must not exist", **ctx)

    enc = variable.encoding
    with gdal.Open(source) as ds:
        structure = ds.GetMetadata("IMAGE_STRUCTURE")
        compression = structure.get("COMPRESSION", "NONE")
        if compression not in LOSSLESS:
            report.add("COG-003", f"compression {compression} is not an approved lossless method", **ctx)
        interleave = structure.get("INTERLEAVE", "")
        resampling = structure.get("OVERVIEW_RESAMPLING", "")
        if first_overviews(ds) and resampling.upper() != enc.overview_resampling:
            report.add("COG-007", f"overview resampling {resampling!r} differs from the declared {enc.overview_resampling}", **ctx)

        if tuple(ds.GetGeoTransform()) != plan.grid.geotransform:
            report.add("COG-006", f"geotransform {ds.GetGeoTransform()} differs from the canonical grid", **ctx)
        srs = ds.GetSpatialRef()
        code = gdalio.crs_code(srs) if srs is not None else None
        if code != plan.grid.crs_code:
            report.add("COG-006", f"CRS {code} differs from {plan.grid.crs_code}", **ctx)
        if (ds.RasterYSize, ds.RasterXSize) != plan.grid.shape:
            report.add("OBS-003", "shape differs from the canonical grid", **ctx)
        if ds.RasterCount != len(expected_names):
            report.add("STAC-011", f"{ds.RasterCount} bands; expected {len(expected_names)}", **ctx)

        first = ds.GetRasterBand(1)
        blocksize = first.GetBlockSize()[0]
        overview_count = first.GetOverviewCount()
        if max(ds.RasterXSize, ds.RasterYSize) > blocksize:
            smallest = first.GetOverview(overview_count - 1) if overview_count else first
            if max(smallest.XSize, smallest.YSize) > blocksize:
                report.add("COG-002", "overviews do not reach a level no larger than one block", **ctx)

        bands: list[BandObservation] = []
        data_type = gdalio.GDAL_TO_STAC_TYPE.get(first.DataType, gdal.GetDataTypeName(first.DataType))
        if data_type != enc.data_type:
            report.add("COG-016", f"data type {data_type} differs from the declared {enc.data_type}", **ctx)
        unit = first.GetUnitType()
        for i, name in enumerate(expected_names, start=1):
            if i > ds.RasterCount:
                break
            band = ds.GetRasterBand(i)
            bctx = {**ctx, "band": i}
            if band.GetDescription() != name:
                report.add("COG-006", f"band description {band.GetDescription()!r} should be {name!r}", **bctx)
            if not gdalio_same(band.GetNoDataValue(), enc.nodata):
                report.add("COG-006", f"nodata {band.GetNoDataValue()} differs from {enc.nodata}", **bctx)
            # GDAL encodes an identity scale (1) or offset (0) by omitting it (Section 20.3).
            scale = band.GetScale() if band.GetScale() is not None else 1.0
            offset = band.GetOffset() if band.GetOffset() is not None else 0.0
            if scale != enc.scale or offset != enc.offset:
                report.add("COG-006", f"scale/offset ({scale}, {offset}) differ from ({enc.scale}, {enc.offset})", **bctx)
            if band.GetUnitType() != variable.curated.unit:
                report.add("COG-006", f"unit {band.GetUnitType()!r} differs from {variable.curated.unit!r}", **bctx)
            meta = band.GetMetadata()
            # GDAL embeds only the valid percent for a band with no valid pixels,
            # so it decides which statistics COG-004 requires.
            try:
                evalid = float(meta["STATISTICS_VALID_PERCENT"])
            except (KeyError, ValueError):
                report.add("COG-004", "embedded valid percent is missing", **bctx)
                continue
            declared_empty = name in plan.empty_timesteps
            if evalid == 0:
                if not declared_empty:
                    report.add("OBS-007", "band has no valid pixels and is not a declared empty timestep", **bctx)
                elif _recompute(band, enc.nodata) is not None:
                    report.add("COG-004", "embedded valid percent is 0 but the band has valid pixels", **bctx)
                else:
                    bands.append(BandObservation(name, None, None, None, None, 0.0))
                continue
            if declared_empty:
                report.add("OBS-007", f"declared empty timestep has valid pixels ({evalid}%)", **bctx)
                continue
            try:
                embedded = tuple(float(meta[key]) for key in (
                    "STATISTICS_MINIMUM", "STATISTICS_MAXIMUM", "STATISTICS_MEAN", "STATISTICS_STDDEV"
                )) + (evalid,)
            except (KeyError, ValueError):
                report.add("COG-004", "embedded statistics are missing", **bctx)
                continue
            if not all(math.isfinite(v) for v in embedded):
                report.add("OBS-007", f"non-finite statistic {embedded}", **bctx)
                continue
            measured = _recompute(band, enc.nodata)
            if measured is None:
                report.add("COG-004", f"embedded valid percent is {evalid} but the band has no valid pixels", **bctx)
                continue
            emin, emax, emean, estd, evalid = embedded
            mmin, mmax, mmean, mstd, mvalid = measured
            # GDAL writes the extremes as ~14-digit text. Read back as float64, a float32
            # extreme is off in the digits the text dropped; in the band's own type the
            # two are the same number.
            as_band = np.dtype(gdal_array.GDALTypeCodeToNumericTypeCode(band.DataType)).type
            if as_band(emin) != as_band(mmin) or as_band(emax) != as_band(mmax):
                report.add("COG-004", f"embedded min/max ({emin}, {emax}) differ from measured ({mmin}, {mmax})", **bctx)
            if not _close(emean, mmean, STAT_RELATIVE_TOLERANCE) or not _close(estd, mstd, STAT_RELATIVE_TOLERANCE):
                report.add("COG-004", f"embedded mean/stddev ({emean}, {estd}) differ from measured ({mmean}, {mstd})", **bctx)
            if abs(evalid - mvalid) > VALID_PERCENT_TOLERANCE:
                report.add("COG-004", f"embedded valid percent {evalid} differs from measured {mvalid}", **bctx)
            bands.append(BandObservation(name, emin, emax, emean, estd, evalid))

    if len(report.errors) > start:
        return None
    # COG-009: checksum only after byte validation.
    return AssetObservation(
        variable_id=variable.id,
        chunk=chunk,
        path=rel_path,
        size=path.stat().st_size,
        checksum=gdalio.sha256_multihash(source),
        data_type=data_type,
        nodata=enc.nodata,
        scale=enc.scale,
        offset=enc.offset,
        unit=unit,
        crs_code=code,
        shape=plan.grid.shape,
        transform=plan.grid.affine,
        bands=tuple(bands),
        blocksize=blocksize,
        interleave=interleave,
        compression=compression,
        overview_count=overview_count,
        overview_resampling=resampling.upper() if overview_count else enc.overview_resampling,
    )


def _inspect_job(args) -> tuple[AssetObservation | None, list]:
    plan, staging, variable_id, chunk, rel, names = args
    gdalio.configure()
    report = Report()
    variable = next(v for v in plan.variables if v.id == variable_id)
    asset = inspect_asset(Path(staging) / rel, rel, plan=plan, variable=variable, chunk=chunk, expected_names=names, report=report)
    return asset, report.findings


def first_overviews(ds) -> int:
    return ds.GetRasterBand(1).GetOverviewCount()


def gdalio_same(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, float) and math.isnan(a):
        return isinstance(b, float) and math.isnan(b)
    return float(a) == float(b)


def observe(plan: object, staging: Path, report: Report) -> FinalObservation | None:
    """Inspect every COG in `staging` and freeze the result (TXN-001 steps 2 and 3)."""
    plan = require_plan(plan)
    gdalio.configure()
    jobs = []
    for variable in plan.variables:
        if plan.temporal:
            for chunk in range(plan.layout.chunk_count):
                jobs.append((variable.id, chunk, plan.layout.cog_path(variable.id, chunk), list(plan.layout.chunk_keys(chunk))))
        else:
            jobs.append((variable.id, None, f"cogs/{variable.id}.tif", [variable.id]))
    results = [None] * len(jobs)
    with ProcessPoolExecutor(max_workers=workers(), mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(_inspect_job, (plan, str(staging), *job)): i for i, job in enumerate(jobs)}
        # Progress in completion order; results stay in job order.
        for done, future in enumerate(as_completed(futures), 1):
            i = futures[future]
            results[i] = future.result()
            log.info("inspected %d of %d COGs: %s", done, len(jobs), jobs[i][2])
    assets: list[AssetObservation] = []
    for asset, findings in results:  # in job order, so the observation is deterministic
        report.extend(findings)
        if asset:
            assets.append(asset)
    if not report.ok:
        return None
    return FinalObservation(
        release_id=plan.release_id,
        declaration_digest=plan.declaration_digest,
        created=plan.created,
        curated=plan.curated,
        resolved_manifest=plan.resolved_manifest,
        producer=plan.producer,
        gdal_version=gdalio.gdal_version(),
        crs_code=plan.grid.crs_code,
        shape=plan.grid.shape,
        transform=plan.grid.affine,
        bbox_wgs84=gdalio.wgs84_bbox(plan.grid.crs_wkt, plan.grid.geotransform, plan.grid.width, plan.grid.height),
        axis=plan.axis,
        layout=plan.layout,
        assets=tuple(assets),
        prepare_scripts=plan.prepare_scripts,
        _issuer=_OBSERVATION_ISSUER,
    )
