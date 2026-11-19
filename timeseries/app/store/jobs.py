from abc import ABC, abstractmethod
import json
import os
import time
import math
import fcntl
import re
from typing import Callable, TypeVar

import anyio
import redis.asyncio as redis_lib
from fastapi import Request
from redis.exceptions import WatchError

T = TypeVar("T")

_JOBS_DIR = "/tmp/skope_jobs"
_JOB_ID = re.compile(r"[A-Za-z0-9_-]{1,128}")


def valid_job_id(job_id: str) -> bool:
    return _JOB_ID.fullmatch(job_id) is not None


def _require_job_id(job_id: str) -> None:
    if not valid_job_id(job_id):
        raise ValueError("Invalid job identifier.")


def _snapshot(state: dict) -> dict[str, str]:
    return {name: json.dumps(value, sort_keys=True) for name, value in state.items()}


class JobStore(ABC):
    @abstractmethod
    async def update_job(self, job_id: str, status_data: dict) -> None: ...

    @abstractmethod
    async def get_job_status(self, job_id: str) -> dict | None: ...

    @abstractmethod
    async def atomic_extraction(self, job_id: str, change: Callable[[dict], T]) -> T:
        """Serialize ownership, admission and job data changes; callback must not do I/O."""
        ...

    async def healthcheck(self) -> None:
        return None

    async def close(self) -> None:
        return None


# Utility exposed to the app layer to clear old cache files
def cleanup_stale_jobs(max_age_hours: int = 24):
    if not os.path.exists(_JOBS_DIR):
        return
    now = time.time()
    for filename in os.listdir(_JOBS_DIR):
        filepath = os.path.join(_JOBS_DIR, filename)
        if os.path.isfile(filepath):
            retention = max_age_hours * (2 if filename.startswith(".meta-") else 1)
            if filename.endswith(".lock") or filename == ".admission.json":
                continue
            if os.stat(filepath).st_mtime < now - (retention * 3600):
                os.remove(filepath)


# File system implementation for simplicity
# Could be replaced with Redis or SQLite
class FileSystemJobStore(JobStore):
    def __init__(self, directory: str = _JOBS_DIR):
        self.directory = directory
        os.makedirs(self.directory, exist_ok=True)

    async def update_job(self, job_id: str, status_data: dict) -> None:
        _require_job_id(job_id)
        file_path = os.path.join(self.directory, f"{job_id}.json")
        temp_path = f"{file_path}.tmp"

        def _write() -> None:
            with open(temp_path, "w") as f:
                json.dump(status_data, f)
            os.replace(temp_path, file_path)

        await anyio.to_thread.run_sync(_write)

    async def get_job_status(self, job_id: str) -> dict | None:
        _require_job_id(job_id)
        file_path = os.path.join(self.directory, f"{job_id}.json")

        def _read() -> dict | None:
            if not os.path.exists(file_path):
                return None
            with open(file_path, "r") as f:
                return json.load(f)

        return await anyio.to_thread.run_sync(_read)

    async def atomic_extraction(self, job_id: str, change: Callable[[dict], T]) -> T:
        _require_job_id(job_id)
        paths = {
            "data": f"{job_id}.json",
            "meta": f".meta-{job_id}.json",
            "active": ".admission.json",
        }

        def transaction():
            def read(name, default):
                path = os.path.join(self.directory, name)
                if not os.path.exists(path):
                    return default
                with open(path) as handle:
                    return json.load(handle)

            state = {
                key: read(path, {} if key == "active" else None)
                for key, path in paths.items()
            }
            before = _snapshot(state)
            result = change(state)
            after = _snapshot(state)
            # Write payload before publishing SUCCESS in metadata. These file
            # replacements are serialized, but are not a crash-atomic database.
            for key, name in paths.items():
                if before[key] == after[key]:
                    continue
                path = os.path.join(self.directory, name)
                if state[key] is None:
                    if os.path.exists(path):
                        os.remove(path)
                else:
                    with open(path + ".tmp", "w") as handle:
                        handle.write(after[key])
                    os.replace(path + ".tmp", path)
            return result

        # A blocking flock in a worker thread ignores cancellation. Poll the
        # nonblocking lock asynchronously so the caller's deadline bounds waits.
        with open(os.path.join(self.directory, ".admission.lock"), "a") as lock:
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    await anyio.sleep(0.01)
            try:
                # Once writing starts, wait for it to finish before releasing
                # the lock, including on cancellation.
                return await anyio.to_thread.run_sync(transaction)
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)


_JOB_TTL_SECONDS = 86400  # 24 hours — matches cleanup_stale_jobs default


class RedisJobStore(JobStore):
    def __init__(self, redis_url: str):
        self._client = redis_lib.Redis.from_url(
            redis_url, decode_responses=True, socket_connect_timeout=3, socket_timeout=3
        )

    async def update_job(self, job_id: str, status_data: dict) -> None:
        _require_job_id(job_id)
        await self._client.set(
            f"job:{job_id}", json.dumps(status_data), ex=_JOB_TTL_SECONDS
        )

    async def get_job_status(self, job_id: str) -> dict | None:
        _require_job_id(job_id)
        raw = await self._client.get(f"job:{job_id}")
        if raw is None:
            return None
        return json.loads(raw)

    async def atomic_extraction(self, job_id: str, change: Callable[[dict], T]) -> T:
        _require_job_id(job_id)
        keys = [
            f"submission:{job_id}",
            f"job:{job_id}",
            "extraction:admission",
            f"extraction:lease:{job_id}",
        ]
        # WATCH/MULTI makes the idempotency check and capacity reservation one
        # transaction across API workers. No process-local lock is authoritative.
        for _ in range(32):
            async with self._client.pipeline() as pipe:
                try:
                    await pipe.watch(*keys)
                    raw = await pipe.mget(keys)
                    state = {
                        name: (
                            json.loads(value)
                            if value
                            else ({} if name == "active" else None)
                        )
                        for name, value in zip(("meta", "data", "active"), raw[:3])
                    }
                    original_meta = dict(state["meta"]) if state["meta"] else {}
                    # WATCH observes automatic expiry too. A lease that expires
                    # between the callback and EXEC must fence that commit.
                    state["lease_valid"] = bool(
                        state["meta"] and raw[3] == state["meta"]["owner"]
                    )
                    before = _snapshot(state)
                    result = change(state)
                    after = _snapshot(state)
                    if before == after:
                        return result
                    now = time.time()
                    meta = state["meta"] or {}
                    pipe.multi()
                    for name, key, expires in [
                        (
                            "meta",
                            keys[0],
                            meta.get("key_expires_at", now + 2 * _JOB_TTL_SECONDS),
                        ),
                        (
                            "data",
                            keys[1],
                            meta.get("expires_at", now + _JOB_TTL_SECONDS),
                        ),
                    ]:
                        if before[name] == after[name]:
                            continue
                        if state[name] is None or expires <= now:
                            pipe.delete(key)
                        else:
                            pipe.set(
                                key,
                                after[name],
                                pxat=math.ceil(expires * 1000),
                            )
                    if before["active"] != after["active"]:
                        pipe.set(keys[2], after["active"])
                    if meta.get("status") in {"PENDING", "PROCESSING"}:
                        if (meta.get("owner"), meta.get("lease_expires_at")) != (
                            original_meta.get("owner"),
                            original_meta.get("lease_expires_at"),
                        ):
                            pipe.set(
                                keys[3],
                                meta["owner"],
                                pxat=math.ceil(meta["lease_expires_at"] * 1000),
                            )
                    elif raw[3] is not None:
                        pipe.delete(keys[3])
                    await pipe.execute()
                    return result
                except WatchError:
                    continue
        raise RuntimeError(
            "Job coordination was contended; retry the same idempotency key."
        )

    async def healthcheck(self) -> None:
        await self._client.ping()

    async def close(self) -> None:
        await self._client.aclose()


def create_job_store(redis_url: str | None) -> JobStore:
    if redis_url:
        return RedisJobStore(redis_url)
    return FileSystemJobStore()


def get_job_store(request: Request) -> JobStore:
    return request.app.state.job_store
