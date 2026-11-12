"""Source preflight: validate a declaration and every source before any output.

Produces a frozen `ValidatedBuildPlan` (OBS-012) or findings. It creates no
release or scratch output (VAL-004). Requirements: META-001 to META-005,
META-008, OBS-001 to OBS-009, OBS-011, COG-007, COG-013,
COG-014, COG-016, REL-006, MAN-010, MAN-011, VAL-003.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from osgeo import gdal

from . import gdalio, prepare
from .documents import load_document
from .findings import Report
from .identity import check_release_id, declaration_digest, declaration_projection, load_ledger
from .models import Curated, SourceManifest
from .plan import _PLAN_ISSUER, Grid, PlannedVariable, Producer, ValidatedBuildPlan
from .timeaxis import Axis, AxisError, ChunkLayout

GRID_TOLERANCE_PX = 1e-3  # Section 20.3
CATEGORY_PRESERVING = {"MODE", "NEAREST"}


@dataclass(frozen=True)
class DatasetFiles:
    directory: Path

    @property
    def dataset_id(self) -> str:
        return self.directory.name

    @property
    def curated(self) -> Path:
        return self.directory / "curated.yml"

    @property
    def source_manifest(self) -> Path:
        return self.directory / "source-manifest.yml"

    @property
    def ledger(self) -> Path:
        return self.directory / "releases.yml"


@dataclass
class _SourceFacts:
    read_path: str
    wkt: str
    geotransform: tuple
    width: int
    height: int
    band_count: int
    data_type: str
    nodata: float | None
    scale: float
    offset: float
    descriptions: list[str]
    area_or_point: str


def _inspect_source(read_path: str) -> _SourceFacts:
    with gdal.Open(read_path) as ds:
        first = ds.GetRasterBand(1)
        types = {ds.GetRasterBand(i).DataType for i in range(1, ds.RasterCount + 1)}
        if len(types) != 1:
            raise ValueError("bands have mixed datatypes")
        nodatas = {ds.GetRasterBand(i).GetNoDataValue() for i in range(1, ds.RasterCount + 1)}
        if len(nodatas) != 1:
            raise ValueError("bands have different nodata values")
        scales = {(ds.GetRasterBand(i).GetScale() or 1.0, ds.GetRasterBand(i).GetOffset() or 0.0) for i in range(1, ds.RasterCount + 1)}
        if len(scales) != 1:
            raise ValueError("bands have different scale/offset")
        scale, offset = scales.pop()
        return _SourceFacts(
            read_path=read_path,
            wkt=ds.GetProjection(),
            geotransform=tuple(ds.GetGeoTransform()),
            width=ds.RasterXSize,
            height=ds.RasterYSize,
            band_count=ds.RasterCount,
            data_type=gdalio.GDAL_TO_STAC_TYPE.get(first.DataType, gdal.GetDataTypeName(first.DataType)),
            nodata=nodatas.pop(),
            scale=scale,
            offset=offset,
            descriptions=[ds.GetRasterBand(i).GetDescription() for i in range(1, ds.RasterCount + 1)],
            area_or_point=ds.GetMetadataItem("AREA_OR_POINT") or "Area",
        )


def _grid_differs(a: _SourceFacts, b: _SourceFacts) -> str | None:
    """OBS-003 with the Section 20.3 tolerance, or None when equivalent."""
    if not gdalio.same_crs(a.wkt, b.wkt):
        return "CRS differs"
    if (a.width, a.height) != (b.width, b.height):
        return f"shape {b.height}x{b.width} differs from {a.height}x{a.width}"
    if a.area_or_point != b.area_or_point:
        return f"AREA_OR_POINT {b.area_or_point} differs from {a.area_or_point}"
    ga, gb = a.geotransform, b.geotransform
    px, py = abs(ga[1]), abs(ga[5])
    if abs(ga[0] - gb[0]) > GRID_TOLERANCE_PX * px or abs(ga[3] - gb[3]) > GRID_TOLERANCE_PX * py:
        return "origin differs by more than 0.001 pixel"
    extent = max(a.width, a.height)
    for i, size in ((1, px), (2, px), (4, py), (5, py)):
        if abs(ga[i] - gb[i]) * extent > GRID_TOLERANCE_PX * size:
            return "pixel size or rotation differs by more than 0.001 pixel across the grid"
    return None


def _same_number(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, float) and math.isnan(a):
        return isinstance(b, float) and math.isnan(b)
    return float(a) == float(b)


def _scan_narrowing(read_path: str, band_count: int, source_nodata, target: str, target_nodata) -> str | None:
    """COG-014: every valid source value must be representable in the target type.

    One pass of full-width swaths over every band, so a pixel-interleaved
    source decodes each block once rather than once per band.
    """
    lo, hi = gdalio.INTEGER_RANGES.get(target, (-math.inf, math.inf))
    bands = list(range(1, band_count + 1))
    with gdal.Open(read_path) as ds:
        first = ds.GetRasterBand(1)
        itemsize = gdal.GetDataTypeSizeBytes(first.DataType)
        rows = gdalio.swath_rows(ds.RasterXSize, band_count, itemsize, first.GetBlockSize()[1], ds.RasterYSize)
        for y in range(0, ds.RasterYSize, rows):
            swath = ds.ReadAsArray(0, y, ds.RasterXSize, min(rows, ds.RasterYSize - y), band_list=bands)
            for b, block in enumerate(swath.reshape(band_count, -1), start=1):
                valid = block if source_nodata is None else block[block != source_nodata]
                if valid.size == 0:
                    continue
                if np.issubdtype(block.dtype, np.floating) and target in gdalio.INTEGER_RANGES:
                    if not np.all(np.isfinite(valid)) or not np.all(np.equal(np.mod(valid, 1), 0)):
                        return f"band {b} holds non-integral values"
                vmin, vmax = valid.min(), valid.max()
                if vmin < lo or vmax > hi:
                    return f"band {b} value range [{vmin}, {vmax}] overflows {target}"
                if target.startswith("float"):
                    # All supported integer source types are exact in float64.
                    # Compare there so numpy cannot round the source down to
                    # the target precision while checking equality.
                    with np.errstate(over="ignore", invalid="ignore"):
                        converted = valid.astype(target)
                    if not np.all(np.isfinite(converted)) or not np.array_equal(
                        converted.astype(np.float64), valid.astype(np.float64)
                    ):
                        return f"band {b} has values not exactly representable as {target}"
                if target_nodata is not None and np.any(valid == target_nodata):
                    return f"band {b} has valid values equal to the target nodata {target_nodata}"
    return None


def _bands_with_data(read_path: str, nodata, band: int | None = None) -> np.ndarray:
    """OBS-007: whether each band, or the one static `band`, has a valid pixel.

    One pass of full-width swaths over every band; it stops early once every
    band has shown a valid pixel.
    """
    with gdal.Open(read_path) as ds:
        bands = [band] if band is not None else list(range(1, ds.RasterCount + 1))
        first = ds.GetRasterBand(bands[0])
        itemsize = gdal.GetDataTypeSizeBytes(first.DataType)
        rows = gdalio.swath_rows(ds.RasterXSize, len(bands), itemsize, first.GetBlockSize()[1], ds.RasterYSize)
        seen = np.zeros(len(bands), dtype=bool)
        for y in range(0, ds.RasterYSize, rows):
            swath = ds.ReadAsArray(0, y, ds.RasterXSize, min(rows, ds.RasterYSize - y), band_list=bands)
            swath = swath.reshape(len(bands), -1)
            if nodata is None or math.isnan(nodata):
                valid = np.ones(swath.shape, dtype=bool)
            else:
                valid = swath != nodata
            if np.issubdtype(swath.dtype, np.floating):
                valid &= np.isfinite(swath)
            seen |= valid.any(axis=1)
            if seen.all():
                break
    return seen


def _declared_empty(curated: Curated, axis: Axis, report: Report, **ctx) -> frozenset[str]:
    """OBS-007: the timesteps `empty_timesteps` declares, as ordered non-overlapping runs on the axis."""
    keys = axis.keys()
    declared: set[str] = set()
    previous_last = -1
    for run in curated.empty_timesteps:
        label = f"empty_timesteps {run.first}..{run.last}"
        try:
            first, last = axis.index(run.first), axis.index(run.last)
        except AxisError as exc:
            report.add("OBS-007", f"{label}: {exc}", **ctx)
            continue
        if first > last:
            report.add("OBS-007", f"{label}: `first` comes after `last`", **ctx)
        elif first <= previous_last:
            report.add("OBS-007", f"{label}: runs must follow axis order without overlapping", **ctx)
        else:
            declared.update(keys[first : last + 1])
            previous_last = last
    return frozenset(declared)


def _runs(indices: list[int], keys: tuple[str, ...]) -> str:
    """Consecutive axis indices as `0417..0589, 0600`."""
    runs: list[list[int]] = []
    for i in indices:
        if runs and i == runs[-1][1] + 1:
            runs[-1][1] = i
        else:
            runs.append([i, i])
    return ", ".join(keys[a] if a == b else f"{keys[a]}..{keys[b]}" for a, b in runs)


def _prepared_source(dataset_id: str, var, *, mirror: Path | None, scripts_root: Path, tools: dict,
                     report: Report, ctx) -> tuple[str, dict, str] | None:
    """Where a prepared source is, its fetch record, and its script's SHA-256 (OBS-016).

    A prepared source exists only in the mirror, so there is no remote copy to
    fall back on; a missing or stale record means fetch has to run first.
    """
    spec = var.source.prepare
    if mirror is None:
        report.add("OBS-016", "a prepared source is read only from the source mirror; pass one", **ctx)
        return None
    output = prepare.output_path(mirror, dataset_id, var.id)
    record = prepare.read_record(output) if output.is_file() else None
    if record is None:
        report.add("OBS-016", f"no prepared output with a record at {output}; run fetch on this host", **ctx)
        return None
    script = prepare.script_path(spec, scripts_root)
    if not script.is_file():
        report.add("META-013", f"prepare script {spec.script} does not exist", **ctx)
        return None
    script_sha256 = prepare.sha256_hex(script)
    reason = prepare.stale_reason(record, spec, script_sha256, tools)
    if reason is not None:
        report.add("OBS-016", f"the prepared output is stale ({reason} since fetch); rerun fetch", **ctx)
        return None
    return str(output), record, script_sha256


def preflight_dataset(
    files: DatasetFiles,
    *,
    release_id: str | None,
    mirror: Path | None,
    producer: Producer,
    report: Report,
    scripts_root: Path = prepare.INGEST_ROOT,
    toolchain: dict | None = None,
) -> ValidatedBuildPlan | None:
    ds = files.dataset_id
    ctx = {"dataset": ds}
    start_errors = len(report.errors)

    curated = load_document(Curated, files.curated, requirement="META-001", report=report, **ctx)
    manifest = load_document(SourceManifest, files.source_manifest, requirement="META-004", report=report, **ctx)
    if curated is None or manifest is None:
        return None
    if curated.id != ds:
        report.add("META-001", f"curated id {curated.id!r} does not match its directory {ds!r}", path=str(files.curated), **ctx)
    if manifest.dataset != ds:
        report.add("META-004", f"source manifest dataset {manifest.dataset!r} does not match {ds!r}", path=str(files.source_manifest), **ctx)

    # OBS-001: identical variable sets.
    curated_ids = {v.id for v in curated.variables}
    manifest_ids = {v.id for v in manifest.variables}
    for missing in sorted(curated_ids - manifest_ids):
        report.add("OBS-001", "curated variable has no source", variable=missing, **ctx)
    for extra in sorted(manifest_ids - curated_ids):
        report.add("OBS-001", "source variable is not curated", variable=extra, **ctx)

    temporal = curated.profile == "TemporalCubeDataset"
    axis: Axis | None = None
    empty: frozenset[str] = frozenset()
    if temporal:
        t = curated.temporal
        try:
            axis = Axis.regular(t.origin, t.step, t.end, t.precision) if t.step else Axis.enumerated(t.values, t.precision)
        except AxisError as exc:
            report.add("OBS-004", str(exc), path=str(files.curated), **ctx)
        if manifest.release.chunk_size is None:
            report.add("META-004", "a temporal dataset's release block needs chunk_size", path=str(files.source_manifest), **ctx)
        if axis is not None:
            empty = _declared_empty(curated, axis, report, path=str(files.curated), **ctx)
    elif manifest.release.chunk_size is not None:
        report.add("META-004", "a static dataset's release block omits chunk_size (Section 20.3)", path=str(files.source_manifest), **ctx)

    curated_by_id = {v.id: v for v in curated.variables}
    facts: dict[str, _SourceFacts] = {}
    checksums: dict[str, str] = {}
    records: dict[str, dict] = {}  # fetch records of prepared sources
    scripts: dict[str, str] = {}
    tools = toolchain if toolchain is not None else prepare.toolchain()
    for var in sorted(manifest.variables, key=lambda v: v.id):
        vctx = {**ctx, "variable": var.id}
        enc, src = var.encoding, var.source
        label = src.uri or src.prepare.script
        if temporal and src.bands != "descriptions":
            report.add("META-008", "a temporal source maps bands to the axis by description (`bands: descriptions`)", **vctx)
        if not temporal and src.band is None:
            report.add("META-004", "a static source names its band (`band: N`)", **vctx)

        # Encoding sanity (OBS-008, OBS-009, COG-007).
        if not (math.isfinite(enc.scale) and math.isfinite(enc.offset)) or enc.scale == 0:
            report.add("OBS-009", "scale and offset must be finite and scale non-zero", **vctx)
        if enc.nodata is not None and enc.data_type in gdalio.INTEGER_RANGES:
            lo, hi = gdalio.INTEGER_RANGES[enc.data_type]
            if not (isinstance(enc.nodata, int) or float(enc.nodata).is_integer()) or not lo <= enc.nodata <= hi:
                report.add("OBS-008", f"nodata {enc.nodata} is not representable as {enc.data_type}", **vctx)
        categorical = var.id in curated_by_id and curated_by_id[var.id].categories is not None
        if categorical and enc.overview_resampling not in CATEGORY_PRESERVING:
            report.add("COG-007", f"categorical variable uses {enc.overview_resampling}; use MODE or NEAREST", **vctx)
        if not categorical and enc.overview_resampling in {"MODE"}:
            report.warn("COG-007", "continuous variable uses MODE overview resampling", **vctx)
        if categorical and enc.nodata is not None and int(enc.nodata) in curated_by_id[var.id].categories:
            report.add("OBS-008", "nodata collides with a declared category value", **vctx)
        expected_predictor = 3 if enc.data_type.startswith("float") else 2
        if enc.predictor not in (1, expected_predictor):
            report.add("COG-013", f"predictor {enc.predictor} does not suit {enc.data_type}", **vctx)

        if src.prepare is not None:
            found = _prepared_source(ds, var, mirror=mirror, scripts_root=scripts_root, tools=tools, report=report, ctx=vctx)
            if found is None:
                continue
            path, records[var.id], scripts[var.id] = found
        try:
            if src.prepare is None:
                path = gdalio.read_path(src.uri, mirror)
            fact = _inspect_source(path)
        except Exception as exc:  # noqa: BLE001 - every source failure becomes a finding (OBS-002)
            report.add("OBS-002", f"cannot inspect source {label}: {exc}", **vctx)
            continue
        facts[var.id] = fact

        # COG-016: preserve the source encoding unless a narrowing is approved.
        if enc.data_type != fact.data_type:
            if enc.narrowing is None:
                report.add("COG-016", f"output data_type {enc.data_type} differs from source {fact.data_type} without an approved narrowing", **vctx)
            else:
                problem = _scan_narrowing(path, fact.band_count, fact.nodata, enc.data_type, enc.nodata)
                if problem:
                    report.add("COG-014", problem, **vctx)
        if not _same_number(enc.nodata, fact.nodata) and enc.narrowing is None:
            report.add("COG-016", f"output nodata {enc.nodata} differs from source nodata {fact.nodata}", **vctx)
        if not (_same_number(enc.scale, fact.scale) and _same_number(enc.offset, fact.offset)):
            report.add("COG-016", f"output scale/offset ({enc.scale}, {enc.offset}) differ from the source's ({fact.scale}, {fact.offset})", **vctx)

        # OBS-005 / OBS-006: band count and descriptions against the reviewed axis.
        if temporal and axis is not None:
            if fact.band_count != axis.count:
                report.add("OBS-005", f"source has {fact.band_count} bands; the axis has {axis.count} timesteps", **vctx)
            expected = axis.keys()
            for i, (got, want) in enumerate(zip(fact.descriptions, expected), start=1):
                if got != want:
                    report.add("OBS-006", f"band description {got!r} should be {want!r}", band=i, **vctx)
                    break
            dupes = len(fact.descriptions) - len(set(fact.descriptions))
            if dupes:
                report.add("OBS-006", f"{dupes} duplicate band descriptions", **vctx)
        elif not temporal and src.band is not None and src.band > fact.band_count:
            report.add("OBS-005", f"band {src.band} does not exist; the source has {fact.band_count}", **vctx)

        # OBS-007: empty bands must be exactly the declared empty timesteps.
        try:
            if temporal and axis is not None and fact.band_count == axis.count:
                has_data = _bands_with_data(path, fact.nodata)
                keys = axis.keys()
                undeclared = [i for i, key in enumerate(keys) if not has_data[i] and key not in empty]
                filled = [i for i, key in enumerate(keys) if has_data[i] and key in empty]
                if undeclared:
                    report.add("OBS-007", f"timesteps {_runs(undeclared, keys)} have no valid pixels; "
                               "if that is expected, declare them in curated `empty_timesteps`", **vctx)
                if filled:
                    report.add("OBS-007", f"declared empty timesteps {_runs(filled, keys)} have valid pixels", **vctx)
            elif not temporal and src.band is not None and src.band <= fact.band_count:
                if not _bands_with_data(path, fact.nodata, src.band)[0]:
                    report.add("OBS-007", f"band {src.band} has no valid pixels", **vctx)
        except RuntimeError as exc:  # GDAL read failure
            report.add("OBS-002", f"cannot read source {label}: {exc}", **vctx)
            continue

        # OBS-011 / META-004: checksum every source.
        try:
            measured = gdalio.sha256_multihash(path)
        except OSError as exc:
            report.add("OBS-011", f"cannot checksum source: {exc}", **vctx)
            continue
        if src.checksum is not None and src.checksum != measured:
            report.add("OBS-011", f"declared checksum {src.checksum} does not match the source bytes ({measured})", **vctx)
        if var.id in records and records[var.id]["output_checksum"] != measured:
            report.add("OBS-016", "the prepared output changed since fetch recorded it; rerun fetch", **vctx)
        checksums[var.id] = measured

    # OBS-003: one canonical grid, the first variable's in identifier order.
    grid: Grid | None = None
    if facts:
        canonical_id = sorted(facts)[0]
        canonical = facts[canonical_id]
        for vid, fact in sorted(facts.items()):
            problem = _grid_differs(canonical, fact)
            if problem:
                report.add("OBS-003", f"{problem} relative to {canonical_id}", dataset=ds, variable=vid)
        from osgeo import osr

        srs = osr.SpatialReference()
        srs.ImportFromWkt(canonical.wkt)
        code = gdalio.crs_code(srs)
        if code is None:
            report.add("STAC-007", "the source CRS has no authority code for proj:code", **ctx)
        grid = Grid(
            crs_wkt=canonical.wkt,
            crs_code=code or "",
            width=canonical.width,
            height=canonical.height,
            geotransform=canonical.geotransform,
            area_or_point=canonical.area_or_point,
        )

    if len(report.errors) > start_errors:
        return None

    # The resolved manifest carries every checksum (META-004); it is frozen from here.
    # A prepared source's input checksums come from its fetch record (OBS-016).
    def resolve(source, variable_id):
        update = {"checksum": checksums[variable_id]}
        if source.prepare is not None:
            inputs = [i.model_copy(update={"checksum": made["checksum"]})
                      for i, made in zip(source.prepare.inputs, records[variable_id]["inputs"])]
            update["prepare"] = source.prepare.model_copy(update={"inputs": inputs})
        return source.model_copy(update=update)

    resolved = manifest.model_copy(
        update={"variables": [v.model_copy(update={"source": resolve(v.source, v.id)}) for v in manifest.variables]}
    )
    digest = declaration_digest(declaration_projection(curated, resolved))

    if release_id is not None:
        ledger = load_ledger(files.ledger, report)
        check_release_id(
            release_id, dataset_id=ds, created=manifest.release.created, digest=digest, ledger=ledger, report=report
        )
    if len(report.errors) > start_errors:
        return None

    planned = tuple(
        PlannedVariable(
            id=v.id,
            curated=curated_by_id[v.id],
            encoding=v.encoding,
            source_uri=v.source.uri,
            source_read_path=facts[v.id].read_path,
            source_checksum=checksums[v.id],
            source_band=v.source.band,
            source_band_count=facts[v.id].band_count,
            source_nodata=facts[v.id].nodata,
        )
        for v in sorted(resolved.variables, key=lambda v: v.id)
    )
    layout = ChunkLayout(ds, axis, manifest.release.chunk_size) if temporal else None
    return ValidatedBuildPlan(
        release_id=release_id or "",
        curated=curated,
        resolved_manifest=resolved,
        declaration_digest=digest,
        grid=grid,
        axis=axis,
        layout=layout,
        variables=planned,
        producer=producer,
        empty_timesteps=empty,
        prepare_scripts=scripts,
        _issuer=_PLAN_ISSUER,
    )


def preflight_many(
    directories: list[Path],
    *,
    release_ids: dict[str, str | None],
    mirror: Path | None,
    producer: Producer,
) -> tuple[dict[str, ValidatedBuildPlan], dict[str, Report]]:
    """MIG-003: preflight every requested dataset before any transformation."""
    plans: dict[str, ValidatedBuildPlan] = {}
    reports: dict[str, Report] = {}
    for directory in directories:
        files = DatasetFiles(directory)
        report = Report()
        plan = preflight_dataset(
            files, release_id=release_ids.get(files.dataset_id), mirror=mirror, producer=producer, report=report
        )
        reports[files.dataset_id] = report
        if plan is not None and report.ok:
            plans[files.dataset_id] = plan
    return plans, reports
