from abc import ABC, abstractmethod
import json
import os
import time

import anyio
import redis.asyncio as redis_lib
from fastapi import Request

_JOBS_DIR = "/tmp/skope_jobs"


class JobStore(ABC):
    @abstractmethod
    async def update_job(self, job_id: str, status_data: dict) -> None: ...

    @abstractmethod
    async def get_job_status(self, job_id: str) -> dict | None: ...

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
            if os.stat(filepath).st_mtime < now - (max_age_hours * 3600):
                os.remove(filepath)


# File system implementation for simplicity
# Could be replaced with Redis or SQLite
class FileSystemJobStore(JobStore):
    def __init__(self, directory: str = _JOBS_DIR):
        self.directory = directory
        os.makedirs(self.directory, exist_ok=True)

    async def update_job(self, job_id: str, status_data: dict) -> None:
        file_path = os.path.join(self.directory, f"{job_id}.json")
        temp_path = f"{file_path}.tmp"

        def _write() -> None:
            with open(temp_path, "w") as f:
                json.dump(status_data, f)
            os.replace(temp_path, file_path)

        await anyio.to_thread.run_sync(_write)

    async def get_job_status(self, job_id: str) -> dict | None:
        file_path = os.path.join(self.directory, f"{job_id}.json")

        def _read() -> dict | None:
            if not os.path.exists(file_path):
                return None
            with open(file_path, "r") as f:
                return json.load(f)

        return await anyio.to_thread.run_sync(_read)


_JOB_TTL_SECONDS = 86400  # 24 hours — matches cleanup_stale_jobs default


class RedisJobStore(JobStore):
    def __init__(self, redis_url: str):
        self._client = redis_lib.Redis.from_url(redis_url, decode_responses=True)

    async def update_job(self, job_id: str, status_data: dict) -> None:
        await self._client.set(
            f"job:{job_id}", json.dumps(status_data), ex=_JOB_TTL_SECONDS
        )

    async def get_job_status(self, job_id: str) -> dict | None:
        raw = await self._client.get(f"job:{job_id}")
        if raw is None:
            return None
        return json.loads(raw)

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
