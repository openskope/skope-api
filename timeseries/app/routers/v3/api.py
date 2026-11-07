import uuid
import logging
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    Path,
    Request,
)
from fastapi.responses import StreamingResponse

from app.config import get_settings
from app.registry.compose import ServedRelease, normalize_key
from app.registry.metadata import MetadataResponse
from app.schemas.timeseries import (
    JobStatus,
    JobStatusResponse,
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
    validate_geom_size,
)
from app.core.job_control import ExtractionJobController, get_job_controller
from app.core.tiles import stream_tile
from app.core.timeseries_tasks import run_timeseries_pipeline_task
from app.core.timeseries_processing import execute_analyze_request

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter()


_PRECISION_BY_LENGTH = {4: "year", 7: "month", 10: "day", 20: "datetime"}


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
@router.post("/timeseries/extract", status_code=202)
async def create_timeseries_job(
    request: Request,
    payload: TimeseriesRequest,
    background_tasks: BackgroundTasks,
    store: JobStore = Depends(get_job_store),
    job_controller: ExtractionJobController = Depends(get_job_controller),
):
    # Validate dataset and variable IDs against the registry before accepting the job
    try:
        release = validate_dataset_and_variable(
            request.app.state.registry, payload.dataset_id, payload.variable_id
        )
    except ValueError as e:
        logger.warning(f"Invalid request attempt: {e}")
        raise HTTPException(status_code=404, detail=str(e))

    _check_windows(payload, release)

    # Pre-flight geometry size check on the dataset's grid
    grid = release.overview.dataset.grid
    try:
        validate_geom_size(
            shapes=payload.selected_area.shapes,
            transform=grid.transform,
            dataset_crs=grid.code,
            max_cells=settings.default_max_cells,
        )
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

    if not job_controller.try_acquire():
        raise HTTPException(
            status_code=503,
            detail="The extraction service is at capacity. Retry later.",
            headers={"Retry-After": "5"},
        )

    # Generate a unique job ID, store initial job status, initiate background processing, and return the job ID to the client
    job_id = str(uuid.uuid4())
    try:
        await store.update_job(job_id, {"status": JobStatus.pending})
        background_tasks.add_task(
            run_timeseries_pipeline_task,
            job_id=job_id,
            payload=payload,
            store=store,
            release=release,
            file_mapping=file_mapping,
            timestep_list=timestep_list,
            job_controller=job_controller,
        )
    except Exception:
        job_controller.release()
        raise

    return {"job_id": job_id, "status": "accepted"}


# 4. Synchronous analysis — applies transform/smoother to a stored base series, no S3 reads
@router.post("/timeseries/analyze")
async def analyze_timeseries(
    request: Request,
    payload: TimeseriesAnalyzeRequest,
    store: JobStore = Depends(get_job_store),
):
    extraction = await store.get_job_status(payload.extraction_id)
    if not extraction:
        raise HTTPException(
            status_code=404, detail="Extraction not found. It may have expired."
        )
    if extraction.get("status") != JobStatus.success:
        raise HTTPException(
            status_code=409,
            detail=f"Extraction not complete: {extraction.get('status')}",
        )

    # 410 when the stored extraction can't serve the request, so the client
    # extracts again; 422 always means the request itself needs changing.
    base_data = extraction.get("base_series")
    if not base_data:
        raise HTTPException(
            status_code=410, detail="No base series found. Re-submit /extract."
        )
    # Compare keys at the extraction's own precision, never as raw strings.
    precision = _PRECISION_BY_LENGTH[len(base_data["timesteps"][0])]
    payload = _with_normalized_ranges(payload, precision)

    # Slicing a stored series that doesn't cover the range would silently drop
    # timesteps.
    metadata = extraction.get("result", {})
    release = request.app.state.registry.get(metadata.get("dataset_id"))
    if release is None or not _covers(
        base_data["timesteps"], payload.time_range, release, precision
    ):
        raise HTTPException(
            status_code=410,
            detail="The extraction doesn't cover the time range. Re-submit /extract.",
        )
    _check_windows(payload, release)

    try:
        return execute_analyze_request(
            payload=payload,
            base_series_payload=base_data,
            extraction_metadata=metadata,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


# 5. Timeseries job status report and results retrieval
@router.get(
    "/timeseries/status/{job_id}",
    response_model=JobStatusResponse,
    response_model_exclude_unset=True,
)
async def get_job_status(
    job_id: str = Path(...), store: JobStore = Depends(get_job_store)
):
    job = await store.get_job_status(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job
