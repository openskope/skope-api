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
    monkeypatch.setattr(get_settings(), "analysis_processing_time", 10)

    async def blocked():
        with pytest.raises(JobContractError) as exc:
            await analyze_bounded(None, {}, {}, limiter)
        assert exc.value.code == "ANALYSIS_TIMEOUT"

    async with anyio.create_task_group() as group:
        group.start_soon(blocked)
    limiter.release()
