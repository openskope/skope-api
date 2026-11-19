import os
import time
from pathlib import Path

import anyio
import pytest
from anyio import to_process

from app.config import get_settings
from app.core.bounded_analysis import analyze_bounded
from app.core.job_control import JobContractError


def worker_pid():
    return os.getpid()


def slow_analysis(pid_file, _base, _metadata):
    Path(pid_file).write_text(str(os.getpid()))
    time.sleep(30)


async def test_analysis_deadline_kills_executing_process(monkeypatch, tmp_path):
    limiter = anyio.CapacityLimiter(1)
    await to_process.run_sync(worker_pid, limiter=limiter)  # warm the process
    monkeypatch.setattr("app.core.bounded_analysis._analyze", slow_analysis)
    monkeypatch.setattr(get_settings(), "analysis_processing_time", 300)
    pid_file = tmp_path / "analysis.pid"
    started = time.monotonic()
    with pytest.raises(JobContractError) as exc:
        await analyze_bounded(str(pid_file), {}, {}, limiter)
    assert exc.value.code == "ANALYSIS_TIMEOUT"
    assert time.monotonic() - started < 3
    pid = int(pid_file.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    assert limiter.borrowed_tokens == 0


async def test_analysis_budget_includes_waiting_for_capacity(monkeypatch):
    limiter = anyio.CapacityLimiter(1)
    await limiter.acquire()
    events = []
    monkeypatch.setattr(
        "app.core.bounded_analysis.log_event",
        lambda event, **fields: events.append(fields),
    )
    monkeypatch.setattr(get_settings(), "analysis_processing_time", 10)

    async def blocked():
        with pytest.raises(JobContractError) as exc:
            await analyze_bounded(None, {}, {}, limiter)
        assert exc.value.code == "ANALYSIS_TIMEOUT"

    async with anyio.create_task_group() as group:
        group.start_soon(blocked)
    limiter.release()
    assert events[0]["outcome"] == "timeout"
    assert events[0]["compute_ms"] == 0
    assert events[0]["capacity_wait_ms"] > 0
    assert events[0]["output_series_counts"] == []


async def test_analysis_logs_capacity_wait_and_completeness(monkeypatch):
    from types import SimpleNamespace

    limiter = anyio.CapacityLimiter(1)
    await limiter.acquire()
    events = []
    monkeypatch.setattr(
        "app.core.bounded_analysis.log_event",
        lambda event, **fields: events.append((event, fields)),
    )
    result = SimpleNamespace(series=[SimpleNamespace(values=[None, 3])])

    async def compute(*args, **kwargs):
        assert limiter.borrowed_tokens == 1
        # No double acquisition of the deployment's per-worker analysis slot.
        assert kwargs["limiter"] is not limiter
        await anyio.sleep(0.02)
        return result

    monkeypatch.setattr(to_process, "run_sync", compute)
    payload = SimpleNamespace(extraction_id="job", requested_series_options=[None])
    base = {"timesteps": ["a", "b", "c"], "mean": [1, None, 3], "median": [1, None, 3]}

    async def analyze():
        assert await analyze_bounded(payload, base, {}, limiter) is result

    async with anyio.create_task_group() as group:
        group.start_soon(analyze)
        await anyio.sleep(0.02)
        assert not events
        limiter.release()
    assert limiter.borrowed_tokens == 0
    assert len(events) == 1
    event, fields = events[0]
    assert event == "analysis_finished"
    assert fields["outcome"] == "success"
    assert fields["capacity_wait_ms"] >= 10
    assert fields["compute_ms"] >= 10
    assert (
        fields["duration_ms"]
        >= fields["capacity_wait_ms"] + fields["compute_ms"] - 0.01
    )
    assert fields["base_series_counts"]["mean"]["missing_timesteps"] == 1
    assert fields["output_series_counts"][0]["timesteps"] == 2
    assert fields["output_series_counts"][0]["missing_timesteps"] == 1


async def test_analysis_timing_preserves_shared_capacity(monkeypatch):
    from types import SimpleNamespace

    limiter = anyio.CapacityLimiter(1)
    active = 0
    peak = 0

    async def compute(*args, **kwargs):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        try:
            await anyio.sleep(0.01)
            return SimpleNamespace(series=[])
        finally:
            active -= 1

    monkeypatch.setattr(to_process, "run_sync", compute)
    async with anyio.create_task_group() as group:
        for _ in range(3):
            group.start_soon(analyze_bounded, None, {}, {}, limiter)
    assert peak == 1
    assert limiter.borrowed_tokens == 0
