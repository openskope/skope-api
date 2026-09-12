import json
import os
import time

import pytest

from app.store.jobs import cleanup_stale_jobs, _JOB_TTL_SECONDS

# ---------------------------------------------------------------------------
# FileSystemJobStore


async def test_update_job_creates_file(fs_job_store, tmp_jobs_dir):
    await fs_job_store.update_job("job-001", {"status": "pending"})
    file_path = os.path.join(tmp_jobs_dir, "job-001.json")
    assert os.path.exists(file_path)
    with open(file_path) as f:
        assert json.load(f) == {"status": "pending"}


async def test_update_job_atomic_overwrite(fs_job_store, tmp_jobs_dir):
    await fs_job_store.update_job("job-002", {"status": "pending"})
    await fs_job_store.update_job("job-002", {"status": "complete"})
    file_path = os.path.join(tmp_jobs_dir, "job-002.json")
    with open(file_path) as f:
        assert json.load(f) == {"status": "complete"}
    # No leftover .tmp file
    assert not os.path.exists(file_path + ".tmp")


async def test_get_job_status_existing(fs_job_store):
    await fs_job_store.update_job("job-003", {"status": "processing", "progress": 50})
    result = await fs_job_store.get_job_status("job-003")
    assert result == {"status": "processing", "progress": 50}


async def test_get_job_status_missing_returns_none(fs_job_store):
    result = await fs_job_store.get_job_status("nonexistent-id")
    assert result is None


# ---------------------------------------------------------------------------
# cleanup_stale_jobs


def test_cleanup_removes_old_files(tmp_path, monkeypatch):
    monkeypatch.setattr("app.store.jobs._JOBS_DIR", str(tmp_path))
    # Create a file and backdate it to 25 hours ago
    old_file = tmp_path / "old-job.json"
    old_file.write_text('{"status": "done"}')
    old_time = time.time() - (25 * 3600)
    os.utime(str(old_file), (old_time, old_time))
    # Create a recent file
    recent_file = tmp_path / "recent-job.json"
    recent_file.write_text('{"status": "pending"}')

    cleanup_stale_jobs(max_age_hours=24)

    assert not old_file.exists()
    assert recent_file.exists()


def test_cleanup_missing_dir_no_error(tmp_path, monkeypatch):
    monkeypatch.setattr("app.store.jobs._JOBS_DIR", str(tmp_path / "nonexistent"))
    cleanup_stale_jobs(max_age_hours=24)  # should return silently


def test_cleanup_keeps_recent_files(tmp_path, monkeypatch):
    monkeypatch.setattr("app.store.jobs._JOBS_DIR", str(tmp_path))
    recent_file = tmp_path / "fresh-job.json"
    recent_file.write_text('{"status": "pending"}')

    cleanup_stale_jobs(max_age_hours=24)

    assert recent_file.exists()


# ---------------------------------------------------------------------------
# RedisJobStore


async def test_redis_update_job_stores_value(redis_job_store):
    await redis_job_store.update_job("job-001", {"status": "PENDING"})
    raw = await redis_job_store._client.get("job:job-001")
    assert json.loads(raw) == {"status": "PENDING"}


async def test_redis_update_job_overwrites(redis_job_store):
    await redis_job_store.update_job("job-002", {"status": "PENDING"})
    await redis_job_store.update_job("job-002", {"status": "SUCCESS", "result": {}})
    result = await redis_job_store.get_job_status("job-002")
    assert result == {"status": "SUCCESS", "result": {}}


async def test_redis_get_job_status_existing(redis_job_store):
    payload = {"status": "PROCESSING"}
    await redis_job_store.update_job("job-003", payload)
    assert await redis_job_store.get_job_status("job-003") == payload


async def test_redis_get_job_status_missing_returns_none(redis_job_store):
    assert await redis_job_store.get_job_status("nonexistent-id") is None


async def test_redis_update_job_sets_ttl(redis_job_store):
    await redis_job_store.update_job("job-004", {"status": "PENDING"})
    ttl = await redis_job_store._client.ttl("job:job-004")
    # TTL should be set and within expected range (allow 1s of drift)
    assert 0 < ttl <= _JOB_TTL_SECONDS


async def test_redis_update_job_resets_ttl_on_overwrite(redis_job_store):
    await redis_job_store.update_job("job-005", {"status": "PENDING"})
    await redis_job_store.update_job("job-005", {"status": "SUCCESS"})
    ttl = await redis_job_store._client.ttl("job:job-005")
    assert 0 < ttl <= _JOB_TTL_SECONDS


async def test_redis_key_namespacing(redis_job_store):
    await redis_job_store.update_job("job-006", {"status": "PENDING"})
    # Key must use the 'job:' prefix — bare id should not exist
    assert await redis_job_store._client.get("job-006") is None
    assert await redis_job_store._client.get("job:job-006") is not None


async def test_redis_full_success_payload(redis_job_store):
    payload = {
        "status": "SUCCESS",
        "result": {"values": [1.0, 2.0, 3.0], "nodata_count": 0},
        "base_series": {"timesteps": ["0100", "0101", "0102"]},
    }
    await redis_job_store.update_job("job-007", payload)
    assert await redis_job_store.get_job_status("job-007") == payload
