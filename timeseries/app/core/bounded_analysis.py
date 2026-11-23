"""Run CPU analysis in a cancellable process rather than blocking the API loop."""

import time

import anyio
from anyio import to_process

from app.telemetry import data_quality_fields, log_event
from app.config import get_settings
from app.core.job_control import JobContractError
from app.core.analysis import execute_analyze_request


def _analyze(payload, base_series, metadata):
    return execute_analyze_request(
        payload=payload, base_series_payload=base_series, extraction_metadata=metadata
    )


async def analyze_bounded(payload, base_series, metadata, limiter):
    started = time.monotonic()
    outcome = "error"
    capacity_acquired = None
    finished = None
    result = None
    try:
        # Includes waiting for a process slot. Cancellation kills an executing
        # process; a timed-out HTTP request does not leave CPU work running.
        with anyio.fail_after(get_settings().analysis_processing_time / 1000):
            async with limiter:
                capacity_acquired = time.monotonic()
                try:
                    # Admission is held above so its wait can be measured.
                    # A per-call limiter avoids acquiring that same token twice
                    # or introducing a second, process-pool capacity queue.
                    result = await to_process.run_sync(
                        _analyze,
                        payload,
                        base_series,
                        metadata,
                        cancellable=True,
                        limiter=anyio.CapacityLimiter(1),
                    )
                finally:
                    finished = time.monotonic()
        outcome = "success"
        return result
    except TimeoutError as exc:
        outcome = "timeout"
        raise JobContractError(
            504,
            "ANALYSIS_TIMEOUT",
            "Analysis did not finish within its time budget. Please try again.",
        ) from exc
    finally:
        ended = time.monotonic()
        log_event(
            "analysis_finished",
            outcome=outcome,
            duration_ms=round((ended - started) * 1000, 3),
            capacity_wait_ms=round(
                (
                    (capacity_acquired if capacity_acquired is not None else ended)
                    - started
                )
                * 1000,
                3,
            ),
            compute_ms=(
                round((finished - capacity_acquired) * 1000, 3)
                if capacity_acquired is not None
                else 0
            ),
            job_id=getattr(payload, "extraction_id", None),
            timesteps=len((base_series or {}).get("timesteps", [])),
            series=len(getattr(payload, "requested_series_options", [])),
            **data_quality_fields(
                base_series,
                (
                    (series.values for series in result.series)
                    if result is not None
                    else ()
                ),
            ),
        )
