import logging
import anyio
from fastapi import HTTPException

# Local imports
from app.registry.compose import ServedRelease
from app.schemas.jobs import ExtractionJobRecord, ExtractionProvenance
from app.schemas.timeseries import JobStatus, TimeseriesRequest
from app.store.jobs import JobStore
from app.core.timeseries_processing import execute_timeseries_job
from app.core.job_control import ExtractionJobController

logger = logging.getLogger(__name__)


async def run_timeseries_pipeline_task(
    job_id: str,
    payload: TimeseriesRequest,
    store: JobStore,
    release: ServedRelease,
    file_mapping: dict[str, list[int]],
    timestep_list: list[str],
    job_controller: ExtractionJobController,
):
    """Extract the timesteps the request handler already selected: the whole
    axis, whatever the requested range. The response holds only the requested
    range (PROTO-007), computed from the whole series."""
    provenance = ExtractionProvenance.from_release(release, payload.variable_id)

    async def save(status: JobStatus, **fields):
        record = ExtractionJobRecord(
            schema_version="1.0.0", provenance=provenance, status=status, **fields
        )
        await store.update_job(job_id, record.stored())

    try:
        with anyio.fail_after(payload.max_processing_time / 1000):
            await save(JobStatus.processing)

            grid = release.overview.dataset.grid
            timeseries_response, base_series_payload = await execute_timeseries_job(
                request=payload,
                file_mapping=file_mapping,
                timestep_list=timestep_list,
                dataset_crs=grid.code,
                dataset_transform_array=list(grid.transform),
            )

            # Save result + base series to JobStore.
            # base_series stores both mean and median zonal stats with timestep index,
            # enabling the synchronous /analyze endpoint to apply any transform/smoother without additional S3 reads.
            await save(
                JobStatus.success,
                result=timeseries_response,
                base_series=base_series_payload,
            )

    except TimeoutError:
        logger.warning(
            "Job %s exceeded its %d ms processing deadline.",
            job_id,
            payload.max_processing_time,
        )
        await save(
            JobStatus.timed_out,
            error=f"Processing exceeded {payload.max_processing_time} ms.",
        )

    except ValueError as ve:
        logger.error(f"Job {job_id} failed validation: {ve}")
        await save(JobStatus.failed, error=str(ve))

    except HTTPException as he:
        logger.error(f"Job {job_id} failed upstream fetch: {he.detail}")
        await save(JobStatus.failed, error=he.detail)

    except Exception:
        logger.exception("Job %s encountered a fatal execution error.", job_id)
        await save(JobStatus.failed, error="An internal processing error occurred.")
    finally:
        job_controller.release()
