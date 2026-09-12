"""
Pipeline integration tests.

These tests exercise the full HTTP request → background task → job store
flow using real raster files from tests/pipeline/data/ and a LocalDataReader.
Background tasks run synchronously inside Starlette's TestClient, so each
client.post() returns only after the task has written its final status.
"""

import pytest
from fastapi.responses import Response

from app.tests.pipeline.conftest import SINGLE_CELL_POLYGON
from app.core.job_control import ExtractionJobController, get_job_controller

EXTRACT_URL = "/timeseries/extract"
ANALYZE_URL = "/timeseries/analyze"
STATUS_URL = "/timeseries/status"


# ---------------------------------------------------------------------------
# Public surface


@pytest.mark.integration
def test_settings_endpoint_is_not_exposed(pipeline_client):
    response = pipeline_client.get("/settings")

    assert response.status_code == 404


# ---------------------------------------------------------------------------
# Helpers


def _extract_payload(dataset_id: str, gte: str, lte: str, **overrides) -> dict:
    payload = {
        "dataset_id": dataset_id,
        "variable_id": "ppt",
        "selected_area": SINGLE_CELL_POLYGON,
        "zonal_statistic": "mean",
        "transform": {"type": "NoTransform"},
        "requested_series_options": [
            {"name": "raw", "smoother": {"type": "NoSmoother"}}
        ],
        "time_range": {"gte": gte, "lte": lte},
    }
    payload.update(overrides)
    return payload


def _analyze_payload(extraction_id: str, **overrides) -> dict:
    payload = {
        "extraction_id": extraction_id,
        "transform": {"type": "NoTransform"},
        "requested_series_options": [
            {"name": "raw", "smoother": {"type": "NoSmoother"}}
        ],
    }
    payload.update(overrides)
    return payload


def _do_extract(client, dataset_id: str, gte: str, lte: str, **overrides) -> str:
    """POST /extract and return the job_id. Asserts 202."""
    resp = client.post(
        EXTRACT_URL, json=_extract_payload(dataset_id, gte, lte, **overrides)
    )
    assert resp.status_code == 202, resp.text
    job_id = resp.json()["job_id"]
    assert job_id
    return job_id


def _get_status(client, job_id: str) -> dict:
    resp = client.get(f"{STATUS_URL}/{job_id}")
    assert resp.status_code == 200, resp.text
    return resp.json()


@pytest.mark.integration
@pytest.mark.parametrize("query", ["", "?colormap=undefined"])
def test_tile_uses_registry_colormap_when_override_is_absent_or_invalid(
    pipeline_client, monkeypatch, query
):
    captured = {}

    async def fake_stream_tile(**kwargs):
        captured.update(kwargs)
        return Response(content=b"tile", media_type="image/png")

    monkeypatch.setattr("app.routers.v3.api.stream_tile", fake_stream_tile)

    response = pipeline_client.get(f"/tiles/test-annual/ppt/0001/0/0/0{query}")

    assert response.status_code == 200
    assert captured["colormap"] == "viridis"


@pytest.mark.integration
@pytest.mark.parametrize(
    "query",
    [
        "?colormap=../viridis",
        "?rescale=0",
        "?rescale=nan,100",
        "?rescale=100,0",
    ],
)
def test_tile_rejects_invalid_style_parameters(pipeline_client, query):
    response = pipeline_client.get(f"/tiles/test-annual/ppt/0001/0/0/0{query}")

    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Extract pipeline — happy path


@pytest.mark.integration
def test_extract_annual_full_range_succeeds(pipeline_client):
    job_id = _do_extract(pipeline_client, "test-annual", "0001", "0005")
    job = _get_status(pipeline_client, job_id)

    assert job["status"] == "SUCCESS"
    assert "base_series" not in job
    assert len(job["result"]["series"][0]["values"]) == 5


@pytest.mark.integration
def test_extract_monthly_full_range_succeeds(pipeline_client):
    job_id = _do_extract(pipeline_client, "test-monthly", "0001-01", "0005-12")
    job = _get_status(pipeline_client, job_id)

    assert job["status"] == "SUCCESS"
    assert "base_series" not in job
    assert len(job["result"]["series"][0]["values"]) == 60


@pytest.mark.integration
def test_extract_partial_range_returns_correct_slice(pipeline_client):
    # Request years 2–4 out of 5: expect 3 timesteps
    job_id = _do_extract(pipeline_client, "test-annual", "0002", "0004")
    job = _get_status(pipeline_client, job_id)

    assert job["status"] == "SUCCESS"
    assert "base_series" not in job
    series = job["result"]["series"][0]
    assert series["time_range"] == {"gte": "0002", "lte": "0004"}
    assert len(series["values"]) == 3


@pytest.mark.integration
def test_extract_null_range_uses_dataset_period(pipeline_client):
    job_id = _do_extract(
        pipeline_client, "test-annual", "0001", "0005", time_range=None
    )
    job = _get_status(pipeline_client, job_id)

    assert job["status"] == "SUCCESS"
    assert job["result"]["series"][0]["time_range"] == {
        "gte": "0001",
        "lte": "0005",
    }


@pytest.mark.integration
def test_extract_rejected_when_worker_is_at_capacity(pipeline_client):
    controller = ExtractionJobController(limit=1)
    assert controller.try_acquire()
    pipeline_client.app.dependency_overrides[get_job_controller] = lambda: controller

    response = pipeline_client.post(
        EXTRACT_URL, json=_extract_payload("test-annual", "0001", "0005")
    )

    assert response.status_code == 503
    assert response.headers["retry-after"] == "5"
    controller.release()


@pytest.mark.integration
def test_extract_processing_deadline_is_enforced(pipeline_client):
    job_id = _do_extract(
        pipeline_client,
        "test-annual",
        "0001",
        "0005",
        max_processing_time=0,
    )

    job = _get_status(pipeline_client, job_id)
    assert job["status"] == "FAILED"
    assert job["error"] == "Processing exceeded 0 ms."


# ---------------------------------------------------------------------------
# Extract pipeline — error cases


@pytest.mark.integration
def test_extract_unknown_dataset_returns_404(pipeline_client):
    resp = pipeline_client.post(
        EXTRACT_URL, json=_extract_payload("nonexistent-ds", "0001", "0005")
    )
    assert resp.status_code == 404


@pytest.mark.integration
def test_extract_unknown_variable_returns_404(pipeline_client):
    payload = _extract_payload("test-annual", "0001", "0005")
    payload["variable_id"] = "no-such-var"
    resp = pipeline_client.post(EXTRACT_URL, json=payload)
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Analyze pipeline — extract first, then analyze


@pytest.mark.integration
def test_analyze_on_extract_result_returns_correct_response(pipeline_client):
    job_id = _do_extract(pipeline_client, "test-annual", "0001", "0005")
    assert _get_status(pipeline_client, job_id)["status"] == "SUCCESS"

    resp = pipeline_client.post(ANALYZE_URL, json=_analyze_payload(job_id))

    assert resp.status_code == 200
    body = resp.json()
    assert body["dataset_id"] == "test-annual"
    assert body["variable_id"] == "ppt"
    assert len(body["series"][0]["values"]) == 5


@pytest.mark.integration
def test_analyze_with_time_range_slices_series(pipeline_client):
    job_id = _do_extract(pipeline_client, "test-annual", "0001", "0005")
    assert _get_status(pipeline_client, job_id)["status"] == "SUCCESS"

    resp = pipeline_client.post(
        ANALYZE_URL,
        json=_analyze_payload(job_id, time_range={"gte": "0002", "lte": "0004"}),
    )

    assert resp.status_code == 200
    assert len(resp.json()["series"][0]["values"]) == 3


@pytest.mark.integration
def test_analyze_with_zscore_transform_returns_valid_response(pipeline_client):
    job_id = _do_extract(pipeline_client, "test-annual", "0001", "0005")
    assert _get_status(pipeline_client, job_id)["status"] == "SUCCESS"

    resp = pipeline_client.post(
        ANALYZE_URL,
        json=_analyze_payload(
            job_id,
            transform={"type": "ZScoreFixedInterval"},
            zonal_statistic="mean",
        ),
    )

    assert resp.status_code == 200
    values = resp.json()["series"][0]["values"]
    assert len(values) == 5
    # All-constant rasters produce std=0, which returns zeros rather than NaN
    assert all(v is not None for v in values)


# ---------------------------------------------------------------------------
# Analyze pipeline — error cases (no raster I/O needed)


@pytest.mark.integration
def test_analyze_nonexistent_job_returns_404(pipeline_client):
    resp = pipeline_client.post(ANALYZE_URL, json=_analyze_payload("does-not-exist"))
    assert resp.status_code == 404


@pytest.mark.integration
async def test_analyze_pending_job_returns_409(pipeline_client, job_store):
    await job_store.update_job("pending-job", {"status": "PENDING"})

    resp = pipeline_client.post(ANALYZE_URL, json=_analyze_payload("pending-job"))
    assert resp.status_code == 409


@pytest.mark.integration
async def test_analyze_failed_job_returns_409(pipeline_client, job_store):
    await job_store.update_job(
        "failed-job", {"status": "FAILED", "error": "upstream error"}
    )

    resp = pipeline_client.post(ANALYZE_URL, json=_analyze_payload("failed-job"))
    assert resp.status_code == 409
