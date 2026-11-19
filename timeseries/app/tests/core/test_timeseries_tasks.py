import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import anyio
import pytest

from app.core.job_control import RecoverableJobs
from app.core.timeseries_tasks import run_timeseries_pipeline_task
from app.schemas.jobs import ExtractionProvenance
from app.schemas.timeseries import TimeseriesRequest


def payload(milliseconds):
    return TimeseriesRequest(
        dataset_id="annual",
        variable_id="ppt",
        selected_area={"type": "Point", "coordinates": [0, 0]},
        zonal_statistic="mean",
        transform={"type": "NoTransform"},
        time_range=None,
        requested_series_options=[
            {"name": "Original", "smoother": {"type": "NoSmoother"}}
        ],
        max_processing_time=milliseconds,
    )


async def test_expired_claim_does_not_start_extraction(monkeypatch):
    monkeypatch.setattr(
        RecoverableJobs, "claim", AsyncMock(return_value=time.time() - 1)
    )
    finish = AsyncMock(return_value=True)
    monkeypatch.setattr(RecoverableJobs, "finish", finish)
    execute = AsyncMock()
    monkeypatch.setattr("app.core.timeseries_tasks.execute_timeseries_job", execute)
    await run_timeseries_pipeline_task("job", "owner", payload(10), None, None, {}, [])
    execute.assert_not_called()
    assert finish.call_args.args[2] == "TIMED_OUT"


async def test_claim_reply_latency_consumes_processing_budget(
    monkeypatch, fs_job_store, served_registry
):
    release = served_registry.get("annual")
    jobs = RecoverableJobs(fs_job_store)
    job = (
        await jobs.reserve(
            "latency-job",
            "fingerprint",
            ExtractionProvenance.from_release(release, "ppt"),
            100,
        )
    ).record
    original = RecoverableJobs.claim

    async def delayed_claim(self, job_id, owner):
        deadline = await original(self, job_id, owner)
        await anyio.sleep(0.08)
        return deadline

    completed = []

    async def execute(**kwargs):
        await anyio.sleep(0.06)
        completed.append(True)
        return SimpleNamespace(model_dump=lambda **kwargs: {}), {}

    monkeypatch.setattr(RecoverableJobs, "claim", delayed_claim)
    monkeypatch.setattr("app.core.timeseries_tasks.execute_timeseries_job", execute)
    await run_timeseries_pipeline_task(
        "latency-job", job["owner"], payload(100), fs_job_store, release, {}, []
    )
    assert completed == []
    assert (await jobs.get("latency-job"))["error_code"] == "PROCESSING_TIMEOUT"


@pytest.mark.parametrize("stage", ["extraction", "serialization"])
async def test_blocking_work_cannot_publish_success_after_deadline(
    monkeypatch, served_registry, stage
):
    monkeypatch.setattr(
        RecoverableJobs,
        "claim",
        AsyncMock(side_effect=lambda *args: time.time() + 0.01),
    )
    finish = AsyncMock(return_value=True)
    monkeypatch.setattr(RecoverableJobs, "finish", finish)

    def serialize(**kwargs):
        if stage == "serialization":
            time.sleep(0.03)
        return {}

    async def execute(**kwargs):
        if stage == "extraction":
            time.sleep(0.03)  # Prevents the cancellation callback from running.
        return SimpleNamespace(model_dump=serialize), {}

    monkeypatch.setattr("app.core.timeseries_tasks.execute_timeseries_job", execute)
    await run_timeseries_pipeline_task(
        "job", "owner", payload(10), None, served_registry.get("annual"), {}, []
    )
    assert finish.call_args.args[2] == "TIMED_OUT"
