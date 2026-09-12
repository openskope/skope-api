import uuid
import logging
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    Path,
    Query,
    Request,
)
from fastapi.responses import StreamingResponse

from app.config import get_settings
from app.schemas.timeseries import TimeseriesAnalyzeRequest, TimeseriesRequest
from app.store.jobs import JobStore, get_job_store
from app.core.validation import (
    validate_dataset_and_variable,
    validate_geom_size,
    validate_tile_style,
)
from app.core.job_control import ExtractionJobController, get_job_controller
from app.core.tiles import stream_tile
from app.core.timeseries_tasks import run_timeseries_pipeline_task
from app.core.timeseries_processing import execute_analyze_request

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter()


# Metadata
@router.get("/metadata")
async def get_global_index(request: Request):
    """Returns the Global Registry."""
    registry_dict = request.app.state.global_registry
    return list(registry_dict.values())


# Tile streaming
@router.get("/tiles/{dataset_id}/{variable_id}/{year}/{z}/{x}/{y}")
async def get_map_tile(
    request: Request,
    dataset_id: str = Path(...),
    variable_id: str = Path(...),
    year: str = Path(...),
    z: int = Path(...),
    x: int = Path(...),
    y: int = Path(...),
    colormap: str | None = Query(
        None, max_length=64, description="Optional color palette override"
    ),
    rescale: str = Query(
        "0,100",
        max_length=64,
        description="min,max data values to map to the colormap",
    ),
) -> StreamingResponse:
    """
    Lightweight endpoint to proxy XYZ tile requests to the internal streaming service.
    """
    app_state = request.app.state
    registry = app_state.global_registry
    try:
        validate_dataset_and_variable(registry, dataset_id, variable_id)
    except ValueError as e:
        logger.warning(f"Invalid request attempt: {e}")
        raise HTTPException(status_code=404, detail=str(e))

    variable = next(
        var
        for var in registry[dataset_id].get("variables", [])
        if var.get("id") == variable_id
    )
    requested_colormap = colormap.strip() if colormap else ""
    effective_colormap = (
        variable.get("colormap", "viridis")
        if requested_colormap.lower() in {"", "undefined", "null"}
        else requested_colormap
    )
    try:
        effective_colormap, rescale = validate_tile_style(effective_colormap, rescale)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return await stream_tile(
        app_state=app_state,
        dataset_id=dataset_id,
        variable_id=variable_id,
        year=year,
        z=z,
        x=x,
        y=y,
        colormap=effective_colormap,
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
    registry = request.app.state.global_registry
    try:
        validate_dataset_and_variable(registry, payload.dataset_id, payload.variable_id)
    except ValueError as e:
        logger.warning(f"Invalid request attempt: {e}")
        raise HTTPException(status_code=404, detail=str(e))

    # Pre-flight geometry size check using registry CRS/transform
    dataset_entry = registry[payload.dataset_id]
    try:
        validate_geom_size(
            shapes=payload.selected_area.shapes,
            dataset_entry=dataset_entry,
            max_cells=settings.default_max_cells,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    if not job_controller.try_acquire():
        raise HTTPException(
            status_code=503,
            detail="The extraction service is at capacity. Retry later.",
            headers={"Retry-After": "5"},
        )

    # Generate a unique job ID, store initial job status, initiate background processing, and return the job ID to the client
    job_id = str(uuid.uuid4())
    try:
        await store.update_job(job_id, {"status": "PENDING"})
        background_tasks.add_task(
            run_timeseries_pipeline_task,
            job_id=job_id,
            payload=payload,
            store=store,
            registry=request.app.state.global_registry,
            data_reader=request.app.state.data_reader,
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
    if extraction.get("status") != "SUCCESS":
        raise HTTPException(
            status_code=409,
            detail=f"Extraction not complete: {extraction.get('status')}",
        )

    base_data = extraction.get("base_series")
    if not base_data:
        raise HTTPException(
            status_code=422, detail="No base series found. Re-submit /extract."
        )

    try:
        return execute_analyze_request(
            payload=payload,
            base_series_payload=base_data,
            extraction_metadata=extraction.get("result", {}),
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


# 5. Timeseries job status report and results retrieval
@router.get("/timeseries/status/{job_id}")
async def get_job_status(
    job_id: str = Path(...), store: JobStore = Depends(get_job_store)
):
    job = await store.get_job_status(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    job.pop("base_series", None)
    return job
