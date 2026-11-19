"""Run CPU analysis in a cancellable process rather than blocking the API loop."""

import anyio
from anyio import to_process

from app.config import get_settings
from app.core.job_control import JobContractError
from app.core.timeseries_processing import execute_analyze_request


def _analyze(payload, base_series, metadata):
    return execute_analyze_request(
        payload=payload, base_series_payload=base_series, extraction_metadata=metadata
    )


async def analyze_bounded(payload, base_series, metadata, limiter):
    try:
        # Includes waiting for a process slot. Cancellation kills an executing
        # process; a timed-out HTTP request does not leave CPU work running.
        with anyio.fail_after(get_settings().analysis_processing_time / 1000):
            return await to_process.run_sync(
                _analyze,
                payload,
                base_series,
                metadata,
                cancellable=True,
                limiter=limiter,
            )
    except TimeoutError as exc:
        raise JobContractError(
            504,
            "ANALYSIS_TIMEOUT",
            "Analysis did not finish within its time budget. Please try again.",
        ) from exc
