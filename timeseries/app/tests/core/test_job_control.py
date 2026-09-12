import pytest

from app.core.job_control import ExtractionJobController


def test_job_controller_rejects_work_at_capacity():
    controller = ExtractionJobController(limit=1)

    assert controller.try_acquire() is True
    assert controller.try_acquire() is False
    assert controller.active == 1

    controller.release()
    assert controller.active == 0
    assert controller.try_acquire() is True


def test_job_controller_rejects_invalid_limit():
    with pytest.raises(ValueError, match="at least 1"):
        ExtractionJobController(limit=0)


def test_job_controller_rejects_unbalanced_release():
    controller = ExtractionJobController(limit=1)

    with pytest.raises(RuntimeError, match="without being acquired"):
        controller.release()
