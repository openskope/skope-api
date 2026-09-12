import logging
import anyio
from fastapi import HTTPException

# Local imports
from app.config import get_settings
from app.schemas.timeseries import TimeseriesRequest
from app.core.slice_resolver import resolve_temporal_slice
from app.store.index_loaders import fetch_lookup_dict
from app.store.jobs import JobStore
from app.store.data_reader import DataReader
from app.core.timeseries_processing import execute_timeseries_job
from app.core.job_control import ExtractionJobController

logger = logging.getLogger(__name__)
settings = get_settings()


async def run_timeseries_pipeline_task(
    job_id: str,
    payload: TimeseriesRequest,
    store: JobStore,
    registry: dict,
    data_reader: DataReader,
    job_controller: ExtractionJobController,
):
    try:
        with anyio.fail_after(payload.max_processing_time / 1000):
            await store.update_job(job_id, {"status": "PROCESSING"})

            dataset_metadata = registry[payload.dataset_id]
            dataset_crs = dataset_metadata["crs"]
            dataset_transform = dataset_metadata["transform"]

            lookup_data = await fetch_lookup_dict(
                dataset_id=payload.dataset_id,
                storage_base_url=settings.storage_base_url,
                data_reader=data_reader,
            )

            time_range = payload.time_range
            if time_range is None:
                gte = dataset_metadata["timespan"]["period"]["gte"]
                lte = dataset_metadata["timespan"]["period"]["lte"]
            else:
                gte = time_range.gte
                lte = time_range.lte

            file_mapping, timestep_list = resolve_temporal_slice(
                lookup_data=lookup_data,
                variable_id=payload.variable_id,
                start_step=gte,
                end_step=lte,
                base_url=settings.storage_base_url,
            )

            timeseries_response, base_series_payload = await execute_timeseries_job(
                request=payload,
                file_mapping=file_mapping,
                timestep_list=timestep_list,
                dataset_crs=dataset_crs,
                dataset_transform_array=dataset_transform,
                resolved_time_range=(gte, lte),
            )

            # Save result + base series to JobStore.
            # base_series stores both mean and median zonal stats with timestep index,
            # enabling the synchronous /analyze endpoint to apply any transform/smoother without additional S3 reads.
            await store.update_job(
                job_id,
                {
                    "status": "SUCCESS",
                    "result": timeseries_response.model_dump(),
                    "base_series": base_series_payload,
                },
            )

    except TimeoutError:
        logger.warning(
            "Job %s exceeded its %d ms processing deadline.",
            job_id,
            payload.max_processing_time,
        )
        await store.update_job(
            job_id,
            {
                "status": "FAILED",
                "error": f"Processing exceeded {payload.max_processing_time} ms.",
            },
        )

    except ValueError as ve:
        logger.error(f"Job {job_id} failed validation: {ve}")
        await store.update_job(job_id, {"status": "FAILED", "error": str(ve)})

    except HTTPException as he:
        logger.error(f"Job {job_id} failed upstream fetch: {he.detail}")
        await store.update_job(job_id, {"status": "FAILED", "error": he.detail})

    except Exception:
        logger.exception("Job %s encountered a fatal execution error.", job_id)
        await store.update_job(
            job_id,
            {"status": "FAILED", "error": "An internal processing error occurred."},
        )
    finally:
        job_controller.release()
