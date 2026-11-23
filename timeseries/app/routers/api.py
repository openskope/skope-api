import logging
import anyio
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    Path,
    Request,
    Header,
)
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ValidationError

from app.config import get_settings
from app.telemetry import log_event
from app.registry.compose import ServedRelease, normalize_key
from app.registry.metadata import MetadataResponse
from app.schemas.jobs import ExtractionJobRecord, ExtractionProvenance
from app.schemas.timeseries import (
    JobStatus,
    JobStatusResponse,
    ExtractionAcceptedResponse,
    TimeseriesResponse,
    JobErrorResponse,
    TimeRange,
    TimeseriesAnalyzeRequest,
    MovingAverageSmoother,
    TimeseriesRequest,
    ZScoreFixedInterval,
    ZScoreMovingInterval,
)
from app.vendor.timeaxis import MalformedTimestep, UnknownTimestep, parse_key
from app.store.jobs import JobStore, get_job_store
from app.core.validation import (
    validate_dataset_and_variable,
    validate_window_size,
)
from app.core.job_control import (
    RecoverableJobs,
    JobContractError,
    accepted,
    request_fingerprint,
    submission_id,
)
from app.core.spatial import resolve_spatial_window
from app.core.tiles import stream_tile
from app.core.timeseries_tasks import run_timeseries_pipeline_task
from app.core.bounded_analysis import analyze_bounded

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter()


def _normalized(time_range: TimeRange | None, precision: str) -> TimeRange | None:
    """Bring a request's range to the dataset's precision (PROTO-009: 422 if it can't be)."""
    if time_range is None:
        return None
    try:
        return TimeRange(
            gte=normalize_key(time_range.gte, precision),
            lte=normalize_key(time_range.lte, precision),
        )
    except MalformedTimestep as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _covers(
    stored: list[str],
    time_range: TimeRange | None,
    release: ServedRelease,
    precision: str,
) -> bool:
    """Whether a stored series holds every axis timestep in the range.

    A stored series is a contiguous run of the axis, so its ends decide. A
    stored series from a release at another precision covers nothing.
    """
    if release.axis.precision != precision:
        return False
    keys = release.axis.keys()
    lo, hi = parse_key(keys[0], precision), parse_key(keys[-1], precision)
    if time_range is not None:
        lo = max(lo, parse_key(time_range.gte, precision))
        hi = min(hi, parse_key(time_range.lte, precision))
    if lo > hi:
        # No axis timestep in the range: nothing to cover, and the analysis
        # reports the empty range.
        return True
    return (
        parse_key(stored[0], precision) <= lo and parse_key(stored[-1], precision) >= hi
    )


def _with_normalized_ranges(payload, precision: str):
    """The payload with its time range and z-score reference range normalized."""
    update = {"time_range": _normalized(payload.time_range, precision)}
    if isinstance(payload.transform, ZScoreFixedInterval):
        update["transform"] = payload.transform.model_copy(
            update={"time_range": _normalized(payload.transform.time_range, precision)}
        )
    return payload.model_copy(update=update)


def _check_windows(payload, release: ServedRelease) -> None:
    """422 when a moving window can't fit in the dataset's axis."""
    count = len(release.axis.keys())
    if isinstance(payload.transform, ZScoreMovingInterval):
        # Only the axis's last timestep has count - 1 before it.
        if payload.transform.width >= count:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"The z-score window ({payload.transform.width} time steps) "
                    f"must be shorter than the dataset ({count} time steps)."
                ),
            )
    for options in payload.requested_series_options:
        smoother = options.smoother
        if isinstance(smoother, MovingAverageSmoother) and smoother.width > count:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"The smoothing window ({smoother.width} time steps) is "
                    f"longer than the dataset ({count} time steps)."
                ),
            )


class BuildInfo(BaseModel):
    build_id: str
    environment: str


@router.get("/version", response_model=BuildInfo)
async def get_build_info():
    """Public deployment identity for support reports; no host or storage details."""
    return BuildInfo(build_id=settings.skope_build_id, environment=settings.environment)


# Metadata
@router.get("/metadata", response_model=MetadataResponse)
async def get_global_index(request: Request):
    """Every served dataset with its display settings, schema 1.0.0 (PROTO-001).

    Built once at startup from the pinned releases and the display files.
    """
    return request.app.state.metadata


# Tile streaming
@router.get("/tiles/{dataset_id}/{variable_id}/{timestep}/{z}/{x}/{y}")
async def get_map_tile(
    request: Request,
    dataset_id: str = Path(...),
    variable_id: str = Path(...),
    timestep: str = Path(
        ..., description="Canonical ISO timestep at the dataset's precision"
    ),
    z: int = Path(...),
    x: int = Path(...),
    y: int = Path(...),
) -> StreamingResponse:
    """
    Proxies one XYZ tile of one timestep to the internal tile server, drawn
    with the variable's display palette and range.
    """
    app_state = request.app.state
    try:
        release = validate_dataset_and_variable(
            app_state.registry, dataset_id, variable_id
        )
    except ValueError as e:
        logger.warning(f"Invalid request attempt: {e}")
        raise HTTPException(status_code=404, detail=str(e))

    # An exact lookup: a malformed key is 422, a key not on the axis is 404
    # (PROTO-006, PROTO-009).
    try:
        cog_path, band = release.resolve(variable_id, timestep)
    except MalformedTimestep as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except UnknownTimestep as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    # One display range serves the tile and the legend, unchanged (DISP-004),
    # so the client doesn't choose the palette or the rescale.
    display = app_state.display.variable(dataset_id, variable_id)
    rescale = None
    if display.range is not None:
        rescale = ",".join(f"{endpoint:g}" for endpoint in display.range)

    return await stream_tile(
        app_state=app_state,
        cog_path=cog_path,
        band=band,
        z=z,
        x=x,
        y=y,
        colormap=display.palette,
        rescale=rescale,
    )


# 3. Timeseries extraction job — async background task, polls via /timeseries/status/{job_id}
@router.post(
    "/timeseries/extract",
    status_code=202,
    response_model=ExtractionAcceptedResponse,
    responses={code: {"model": JobErrorResponse} for code in (400, 409, 410, 503)},
)
async def create_timeseries_job(
    request: Request,
    payload: TimeseriesRequest,
    background_tasks: BackgroundTasks,
    store: JobStore = Depends(get_job_store),
    idempotency_key: str = Header(
        ...,
        alias="Idempotency-Key",
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9_-]+$",
    ),
    release_id: str | None = Header(None, alias="X-Skope-Release-Id", max_length=128),
    declaration_digest: str | None = Header(
        None, alias="X-Skope-Declaration-Digest", max_length=128
    ),
):
    if (release_id is None) != (declaration_digest is None):
        raise JobContractError(
            400,
            "SOURCE_IDENTITY_REQUIRED",
            "Supply both source release headers, or neither.",
        )
    jobs = RecoverableJobs(store)
    job_id = submission_id(idempotency_key)
    fingerprint = request_fingerprint(payload, release_id, declaration_digest)
    # A retry recovers its accepted release even after a deployment changes the
    # registry, and even when no capacity is available for new work.
    previous = await jobs.lookup(job_id, fingerprint)
    if previous:
        log_event("extraction_submission", job_id=job_id, recovered=True)
        return accepted(previous.record)

    # Validate dataset and variable IDs against the registry before accepting the job
    try:
        release = validate_dataset_and_variable(
            request.app.state.registry, payload.dataset_id, payload.variable_id
        )
    except ValueError as e:
        logger.warning(f"Invalid request attempt: {e}")
        raise HTTPException(status_code=404, detail=str(e))

    if release_id is not None and (release_id, declaration_digest) != (
        release.overview.release_id,
        release.overview.declaration_digest,
    ):
        raise JobContractError(
            409,
            "SOURCE_CHANGED",
            "The dataset release has changed. Reload the dataset before extracting.",
        )

    _check_windows(payload, release)

    # A 422 unless every polygon is valid, the area overlaps the dataset's
    # grid, and its read window fits.
    payload.selected_area.validate_polygons()
    grid = release.overview.dataset.grid
    selection = resolve_spatial_window(
        payload.selected_area.shapes, grid.transform, grid.code, grid.shape
    )
    try:
        validate_window_size(selection, settings.default_max_cells)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # Check the range now, so a range that selects nothing is a 422 (PROTO-009)
    # rather than a failed job. No range means the whole axis.
    payload = _with_normalized_ranges(payload, release.axis.precision)
    keys = release.axis.keys()
    time_range = payload.time_range or TimeRange(gte=keys[0], lte=keys[-1])
    _, requested_timesteps = release.select(
        payload.variable_id, time_range.gte, time_range.lte
    )
    if not requested_timesteps:
        raise HTTPException(
            status_code=422,
            detail=f"The time range [{time_range.gte}, {time_range.lte}] selects no timesteps.",
        )
    # Read the whole axis whatever the range: the stored base series then serves
    # any later range through /analyze, and windows and reference periods reach
    # past the requested range.
    file_mapping, timestep_list = release.select(payload.variable_id, keys[0], keys[-1])

    reservation = await jobs.reserve(
        job_id,
        fingerprint,
        ExtractionProvenance.from_release(release, payload.variable_id),
        payload.max_processing_time,
    )
    if reservation.created:
        try:
            background_tasks.add_task(
                run_timeseries_pipeline_task,
                job_id=job_id,
                owner=reservation.record["owner"],
                payload=payload,
                store=store,
                release=release,
                file_mapping=file_mapping,
                timestep_list=timestep_list,
            )
        except Exception as exc:
            # Keep the key bound to this terminal attempt; a retry must never
            # schedule a second execution after an uncertain acceptance.
            with anyio.CancelScope(shield=True):
                await jobs.finish(
                    job_id,
                    reservation.record["owner"],
                    "FAILED",
                    error="The extraction could not be scheduled. Start a new attempt.",
                    error_code="SCHEDULING_FAILED",
                )
            raise JobContractError(
                503,
                "SCHEDULING_FAILED",
                "The extraction could not be scheduled. Start a new attempt.",
            ) from exc
    log_event("extraction_submission", job_id=job_id, recovered=not reservation.created)
    return accepted(reservation.record)


# 4. Synchronous analysis — applies transform/smoother to a stored base series, no S3 reads
@router.post(
    "/timeseries/analyze",
    response_model=TimeseriesResponse,
    responses={code: {"model": JobErrorResponse} for code in (404, 409, 410, 503, 504)},
)
async def analyze_timeseries(
    request: Request,
    payload: TimeseriesAnalyzeRequest,
    store: JobStore = Depends(get_job_store),
):
    extraction = await RecoverableJobs(store).get(payload.extraction_id)
    if extraction.get("status") != JobStatus.success:
        raise JobContractError(
            409,
            extraction.get("error_code") or "JOB_INCOMPLETE",
            extraction.get("error")
            or f"Extraction not complete: {extraction.get('status')}",
        )

    # 410 when the stored extraction can't serve the request, so the client
    # extracts again; 422 always means the request itself needs changing.
    try:
        record = ExtractionJobRecord.model_validate(extraction)
    except ValidationError as exc:
        raise JobContractError(
            410,
            "EXTRACTION_INCOMPATIBLE",
            "The stored extraction is incompatible. Start a new attempt.",
        ) from exc

    provenance = record.provenance
    release = request.app.state.registry.get(provenance.dataset_id)
    if (
        release is None
        or not release.has_variable(provenance.variable_id)
        or provenance
        != ExtractionProvenance.from_release(release, provenance.variable_id)
    ):
        raise JobContractError(
            410,
            "SOURCE_CHANGED",
            "The source release has changed. Start a new attempt.",
        )

    precision = provenance.precision
    payload = _with_normalized_ranges(payload, precision)
    base_data = record.base_series.model_dump()

    # Slicing a stored series that doesn't cover the range would silently drop
    # timesteps.
    if not _covers(base_data["timesteps"], payload.time_range, release, precision):
        raise JobContractError(
            410,
            "EXTRACTION_INCOMPATIBLE",
            "The extraction doesn't cover the time range. Start a new attempt.",
        )
    _check_windows(payload, release)

    try:
        return await analyze_bounded(
            payload,
            base_data,
            record.result.model_dump(),
            request.app.state.analysis_limiter,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


# 5. Timeseries job status report and results retrieval
@router.get(
    "/timeseries/status/{job_id}",
    response_model=JobStatusResponse,
    response_model_exclude_unset=True,
    responses={code: {"model": JobErrorResponse} for code in (404, 410, 503)},
)
async def get_job_status(
    job_id: str = Path(...), store: JobStore = Depends(get_job_store)
):
    return await RecoverableJobs(store).get(job_id)
