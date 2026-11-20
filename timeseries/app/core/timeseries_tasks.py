import logging
import time
import anyio
from fastapi import HTTPException

from app.telemetry import data_quality_fields, log_event
from app.registry.compose import ServedRelease
from app.schemas.timeseries import TimeseriesRequest
from app.store.jobs import JobStore
from app.core.zonal_stats import execute_timeseries_job
from app.core.job_control import RecoverableJobs

logger = logging.getLogger(__name__)


async def run_timeseries_pipeline_task(
    job_id: str,
    owner: str,
    payload: TimeseriesRequest,
    store: JobStore,
    release: ServedRelease,
    file_mapping: dict[str, list[int]],
    timestep_list: list[str],
):
    started = time.monotonic()
    jobs = RecoverableJobs(store)
    processing_deadline = await jobs.claim(job_id, owner)
    if processing_deadline is None:
        return
    # Claim/store round trips consume the published budget rather than starting
    # another full budget when the claim response arrives.
    remaining = max(
        0, min(payload.max_processing_time / 1000, processing_deadline - time.time())
    )
    deadline = anyio.current_time() + remaining
    status, fields = "FAILED", {
        "error": "An internal processing error occurred.",
        "error_code": "EXTRACTION_FAILED",
    }
    try:
        if remaining <= 0:
            raise TimeoutError
        with anyio.fail_after(remaining):
            grid = release.overview.dataset.grid
            response, base_series = await execute_timeseries_job(
                request=payload,
                file_mapping=file_mapping,
                timestep_list=timestep_list,
                dataset_crs=grid.code,
                dataset_transform_array=list(grid.transform),
            )
            fields = {
                "result": response.model_dump(mode="json"),
                "base_series": base_series,
            }
            if anyio.current_time() >= deadline:
                # Native/blocking work can finish before cancellation gets a
                # checkpoint. It still must not publish a late success.
                raise TimeoutError
            status = "SUCCESS"
    except TimeoutError:
        logger.warning(
            "Job %s exceeded its %d ms processing deadline.",
            job_id,
            payload.max_processing_time,
        )
        fields = {
            "error": f"Processing exceeded {payload.max_processing_time} ms.",
            "error_code": "PROCESSING_TIMEOUT",
        }
        status = "TIMED_OUT"
    except ValueError as exc:
        fields = {"error": str(exc), "error_code": "INVALID_EXTRACTION"}
    except HTTPException as exc:
        fields = {"error": str(exc.detail), "error_code": "UPSTREAM_FAILURE"}
    except Exception:
        logger.exception("Job %s encountered a fatal execution error.", job_id)
    except BaseException:
        fields = {
            "error": "The extraction worker stopped before completion.",
            "error_code": "WORKER_LOST",
        }
        raise
    finally:
        # Shutdown cancellation must not strand admission. If storage itself is
        # unavailable, the finite ownership lease handles recovery instead.
        with anyio.CancelScope(shield=True):
            recorded = False
            try:
                recorded = await jobs.finish(job_id, owner, status, **fields)
            except Exception:
                logger.exception(
                    "Unable to record the terminal state of job %s", job_id
                )
            finally:
                log_event(
                    "extraction_finished",
                    job_id=job_id,
                    dataset_id=payload.dataset_id,
                    variable_id=payload.variable_id,
                    outcome=status,
                    recorded=recorded,
                    error_code=fields.get("error_code"),
                    duration_ms=round((time.monotonic() - started) * 1000, 3),
                    files=len(file_mapping),
                    timesteps=len(timestep_list),
                    **data_quality_fields(
                        fields.get("base_series"),
                        (
                            series["values"]
                            for series in fields.get("result", {}).get("series", [])
                        ),
                    ),
                )
