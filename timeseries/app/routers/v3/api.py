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
from app.registry.compose import normalize_key
from app.registry.metadata import MetadataResponse
from app.schemas.timeseries import (
    JobStatus,
    JobStatusResponse,
    TimeRange,
    TimeseriesAnalyzeRequest,
    TimeseriesRequest,
    ZScoreFixedInterval,
)
from app.vendor.timeaxis import MalformedTimestep, UnknownTimestep
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


def _with_normalized_ranges(payload, precision: str):
    """The payload with its time range and z-score reference range normalized."""
    update = {"time_range": _normalized(payload.time_range, precision)}
    if isinstance(payload.transform, ZScoreFixedInterval):
        update["transform"] = payload.transform.model_copy(
            update={"time_range": _normalized(payload.transform.time_range, precision)}
        )
    return payload.model_copy(update=update)


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

    # Select the timesteps now, so a range that selects nothing is a 422
    # (PROTO-009) rather than a failed job. No range means the whole axis.
    payload = _with_normalized_ranges(payload, release.axis.precision)
    keys = release.axis.keys()
    time_range = payload.time_range or TimeRange(gte=keys[0], lte=keys[-1])
    file_mapping, timestep_list = release.select(
        payload.variable_id, time_range.gte, time_range.lte
    )
    if not timestep_list:
        raise HTTPException(
            status_code=422,
            detail=f"The time range [{time_range.gte}, {time_range.lte}] selects no timesteps.",
        )

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

    base_data = extraction.get("base_series")
    if not base_data:
        raise HTTPException(
            status_code=422, detail="No base series found. Re-submit /extract."
        )
    # Compare keys at the extraction's own precision, never as raw strings.
    precision = _PRECISION_BY_LENGTH[len(base_data["timesteps"][0])]
    payload = _with_normalized_ranges(payload, precision)

    try:
        return execute_analyze_request(
            payload=payload,
            base_series_payload=base_data,
            extraction_metadata=extraction.get("result", {}),
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
