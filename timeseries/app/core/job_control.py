from functools import lru_cache
from threading import Lock

from app.config import get_settings


class ExtractionJobController:
    """Process-local admission control for expensive extraction jobs."""

    def __init__(self, limit: int):
        if limit < 1:
            raise ValueError("Extraction job limit must be at least 1.")
        self.limit = limit
        self._active = 0
        self._lock = Lock()

    def try_acquire(self) -> bool:
        with self._lock:
            if self._active >= self.limit:
                return False
            self._active += 1
            return True

    def release(self) -> None:
        with self._lock:
            if self._active == 0:
                raise RuntimeError(
                    "Extraction job slot released without being acquired."
                )
            self._active -= 1

    @property
    def active(self) -> int:
        with self._lock:
            return self._active


@lru_cache
def get_job_controller() -> ExtractionJobController:
    return ExtractionJobController(get_settings().max_concurrent_jobs)
