"""
Pipeline integration tests.

These tests exercise the full HTTP request → background task → job store
flow against small synthetic releases composed at startup (see conftest.py).
Background tasks run synchronously inside Starlette's TestClient, so each
client.post() returns only after the task has written its final status.
"""

import time
from dataclasses import replace

import pytest
from fastapi.responses import Response
from fastapi.testclient import TestClient

from app.tests.pipeline.conftest import SINGLE_CELL_POLYGON
from app.core.job_control import ExtractionJobController, get_job_controller
from app.schemas.jobs import ExtractionJobRecord

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


def _capture_tile(monkeypatch):
    captured = {}

    async def fake_stream_tile(**kwargs):
        captured.update(kwargs)
        return Response(content=b"tile", media_type="image/png")

    monkeypatch.setattr("app.routers.v3.api.stream_tile", fake_stream_tile)
    return captured


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
@pytest.mark.parametrize("query", ["", "?colormap=magma&rescale=0,1"])
def test_tile_uses_the_display_palette_and_range(pipeline_client, monkeypatch, query):
    captured = _capture_tile(monkeypatch)

    response = pipeline_client.get(f"/tiles/test_annual/ppt/0001/0/0/0{query}")

    assert response.status_code == 200
    assert captured["colormap"] == "skope-precip"
    assert captured["rescale"] == "0,4.5"


# ---------------------------------------------------------------------------
# Extract pipeline — happy path


@pytest.mark.integration
def test_extract_annual_full_range_succeeds(pipeline_client):
    job_id = _do_extract(pipeline_client, "test_annual", "0001", "0005")
    job = _get_status(pipeline_client, job_id)

    assert job["status"] == "SUCCESS"
    assert "base_series" not in job
    assert len(job["result"]["series"][0]["values"]) == 5


@pytest.mark.integration
def test_status_keeps_null_statistics_and_lists_only_set_fields(pipeline_client):
    # 0001 has no predecessors, so it has no moving z-score and no statistics.
    job_id = _do_extract(
        pipeline_client,
        "test_annual",
        "0001",
        "0001",
        transform={"type": "ZScoreMovingInterval", "width": 2},
    )
    job = _get_status(pipeline_client, job_id)

    assert set(job) == {"status", "result"}
    assert job["result"]["summary_stats"][0]["stdev"] is None


@pytest.mark.integration
def test_extract_monthly_full_range_succeeds(pipeline_client):
    job_id = _do_extract(pipeline_client, "test_monthly", "0001-01", "0005-12")
    job = _get_status(pipeline_client, job_id)

    assert job["status"] == "SUCCESS"
    assert "base_series" not in job
    assert len(job["result"]["series"][0]["values"]) == 60


@pytest.mark.integration
def test_extract_partial_range_returns_correct_slice(pipeline_client):
    # Request years 2–4 out of 5: expect 3 timesteps
    job_id = _do_extract(pipeline_client, "test_annual", "0002", "0004")
    job = _get_status(pipeline_client, job_id)

    assert job["status"] == "SUCCESS"
    assert "base_series" not in job
    series = job["result"]["series"][0]
    assert series["time_range"] == {"gte": "0002", "lte": "0004"}
    assert len(series["values"]) == 3


@pytest.mark.integration
def test_extract_null_range_uses_dataset_period(pipeline_client):
    job_id = _do_extract(
        pipeline_client, "test_annual", "0001", "0005", time_range=None
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
        EXTRACT_URL, json=_extract_payload("test_annual", "0001", "0005")
    )

    assert response.status_code == 503
    assert response.headers["retry-after"] == "5"
    controller.release()


@pytest.mark.integration
def test_extract_processing_deadline_is_enforced(pipeline_client):
    job_id = _do_extract(
        pipeline_client,
        "test_annual",
        "0001",
        "0005",
        max_processing_time=0,
    )

    job = _get_status(pipeline_client, job_id)
    assert job["status"] == "TIMED_OUT"
    assert job["error"] == "Processing exceeded 0 ms."


@pytest.mark.integration
def test_extract_stops_reading_at_its_deadline(pipeline_client, monkeypatch):
    # One band per read, each taking longer than the deadline. The release's
    # 5 bands sit in 3 files read in parallel, so a stopped job reads at most
    # one band per file; a job that runs on reads all 5.
    reads = []

    def slow_chunks(band_indices, chunk_size):
        for band in band_indices:
            yield [band]
            # Resumed once the band has been read.
            time.sleep(0.3)
            reads.append(band)

    monkeypatch.setattr(
        "app.core.timeseries_processing.generate_band_chunks", slow_chunks
    )
    controller = ExtractionJobController(limit=1)
    pipeline_client.app.dependency_overrides[get_job_controller] = lambda: controller

    job_id = _do_extract(
        pipeline_client,
        "test_annual",
        "0001",
        "0005",
        max_processing_time=100,
    )

    assert _get_status(pipeline_client, job_id)["status"] == "TIMED_OUT"
    assert len(reads) <= 3
    assert controller.active == 0


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
    payload = _extract_payload("test_annual", "0001", "0005")
    payload["variable_id"] = "no-such-var"
    resp = pipeline_client.post(EXTRACT_URL, json=payload)
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Analyze pipeline — extract first, then analyze


@pytest.mark.integration
def test_analyze_on_extract_result_returns_correct_response(pipeline_client):
    job_id = _do_extract(pipeline_client, "test_annual", "0001", "0005")
    assert _get_status(pipeline_client, job_id)["status"] == "SUCCESS"

    resp = pipeline_client.post(ANALYZE_URL, json=_analyze_payload(job_id))

    assert resp.status_code == 200
    body = resp.json()
    assert body["dataset_id"] == "test_annual"
    assert body["variable_id"] == "ppt"
    assert len(body["series"][0]["values"]) == 5


@pytest.mark.integration
def test_analyze_with_time_range_slices_series(pipeline_client):
    job_id = _do_extract(pipeline_client, "test_annual", "0001", "0005")
    assert _get_status(pipeline_client, job_id)["status"] == "SUCCESS"

    resp = pipeline_client.post(
        ANALYZE_URL,
        json=_analyze_payload(job_id, time_range={"gte": "0002", "lte": "0004"}),
    )

    assert resp.status_code == 200
    assert len(resp.json()["series"][0]["values"]) == 3


@pytest.mark.integration
def test_analyze_with_zscore_transform_returns_valid_response(pipeline_client):
    job_id = _do_extract(pipeline_client, "test_annual", "0001", "0005")
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


@pytest.mark.integration
def test_extract_stores_the_whole_axis_for_any_range(pipeline_client):
    job_id = _do_extract(pipeline_client, "test_annual", "0002", "0004")
    assert _get_status(pipeline_client, job_id)["status"] == "SUCCESS"

    resp = pipeline_client.post(ANALYZE_URL, json=_analyze_payload(job_id))

    assert resp.status_code == 200
    assert resp.json()["series"][0]["values"] == [1.0, 2.0, 3.0, 4.0, 5.0]


@pytest.mark.integration
def test_extract_smooths_with_the_years_before_its_range(pipeline_client):
    job_id = _do_extract(
        pipeline_client,
        "test_annual",
        "0003",
        "0005",
        requested_series_options=[
            {
                "name": "smoothed",
                "smoother": {
                    "type": "MovingAverageSmoother",
                    "method": "trailing",
                    "width": 3,
                },
            }
        ],
    )

    series = _get_status(pipeline_client, job_id)["result"]["series"][0]
    assert series["values"] == [2.0, 3.0, 4.0]
    assert series["timesteps"] == ["0003", "0004", "0005"]


@pytest.mark.integration
def test_extract_reference_interval_can_lie_outside_its_range(pipeline_client):
    job_id = _do_extract(
        pipeline_client,
        "test_annual",
        "0004",
        "0005",
        transform={
            "type": "ZScoreFixedInterval",
            "time_range": {"gte": "0001", "lte": "0002"},
        },
    )

    job = _get_status(pipeline_client, job_id)
    assert job["status"] == "SUCCESS", job
    # Reference mean 1.5, population stdev 0.5.
    assert job["result"]["series"][0]["values"] == [5.0, 7.0]


# ---------------------------------------------------------------------------
# Analyze pipeline — error cases (no raster I/O needed)


@pytest.mark.integration
async def test_analyze_rejects_a_range_the_extraction_does_not_cover(
    pipeline_client, job_store
):
    job_id = _do_extract(pipeline_client, "test_annual", "0001", "0005")
    stored = await job_store.get_job_status(job_id)
    stored["base_series"] = {
        "timesteps": ["0002", "0003", "0004"],
        "mean": [2.0, 3.0, 4.0],
        "median": [2.0, 3.0, 4.0],
    }
    await job_store.update_job("short-job", stored)

    covered = pipeline_client.post(
        ANALYZE_URL,
        json=_analyze_payload("short-job", time_range={"gte": "0002", "lte": "0003"}),
    )
    uncovered = pipeline_client.post(
        ANALYZE_URL,
        json=_analyze_payload("short-job", time_range={"gte": "0001", "lte": "0005"}),
    )

    assert covered.status_code == 200
    # 410, not 422: the client extracts again rather than showing an error.
    assert uncovered.status_code == 410


@pytest.mark.integration
async def test_job_lifecycle_keeps_private_release_provenance(
    pipeline_client, job_store, monkeypatch
):
    written = []
    update = job_store.update_job

    async def capture(job_id, data):
        written.append(ExtractionJobRecord.model_validate(data))
        await update(job_id, data)

    monkeypatch.setattr(job_store, "update_job", capture)
    job_id = _do_extract(pipeline_client, "test_annual", "0001", "0005")
    assert [record.status for record in written] == ["PENDING", "PROCESSING", "SUCCESS"]
    release = pipeline_client.app.state.registry.get("test_annual")
    for record in written:
        assert record.schema_version == "1.0.0"
        assert record.provenance.release_id == release.overview.release_id
        assert (
            record.provenance.declaration_digest == release.overview.declaration_digest
        )
        assert record.provenance.precision == "year"
    assert set(_get_status(pipeline_client, job_id)) == {"status", "result"}


@pytest.mark.integration
@pytest.mark.parametrize(
    "field,value",
    [("release_id", "test_annual-r-2026.02.02"), ("declaration_digest", "e" * 64)],
)
def test_analyze_rejects_a_previous_source_release(
    pipeline_client, monkeypatch, field, value
):
    job_id = _do_extract(pipeline_client, "test_annual", "0001", "0005")
    releases = pipeline_client.app.state.registry.releases
    release = releases["test_annual"]
    monkeypatch.setitem(
        releases,
        "test_annual",
        replace(release, overview=release.overview.model_copy(update={field: value})),
    )

    response = pipeline_client.post(ANALYZE_URL, json=_analyze_payload(job_id))

    assert response.status_code == 410
    assert "source release has changed" in response.json()["detail"]


@pytest.mark.integration
@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", "2.0.0"),
        ("schema_version", None),
        ("provenance", None),
        ("base_series", {"timesteps": [], "mean": [], "median": []}),
        ("base_series", {"timesteps": ["bad"], "mean": [1], "median": [1]}),
        ("base_series", {"timesteps": ["0001"], "mean": [1, 2], "median": [1]}),
    ],
)
async def test_analyze_requires_a_supported_job_record(
    pipeline_client, job_store, field, value
):
    job_id = _do_extract(pipeline_client, "test_annual", "0001", "0005")
    stored = await job_store.get_job_status(job_id)
    if value is None:
        stored.pop(field)
    else:
        stored[field] = value
    await job_store.update_job(job_id, stored)

    response = pipeline_client.post(ANALYZE_URL, json=_analyze_payload(job_id))

    assert response.status_code == 410
    assert "incompatible" in response.json()["detail"]


@pytest.mark.integration
@pytest.mark.parametrize(
    "overrides, message",
    [
        # test_annual has 5 timesteps: 4 can precede the last one.
        ({"transform": {"type": "ZScoreMovingInterval", "width": 5}}, "z-score window"),
        (
            {
                "requested_series_options": [
                    {
                        "name": "raw",
                        "smoother": {
                            "type": "MovingAverageSmoother",
                            "method": "trailing",
                            "width": 6,
                        },
                    }
                ]
            },
            "smoothing window",
        ),
        (
            {
                "transform": {
                    "type": "ZScoreFixedInterval",
                    "time_range": {"gte": "0003", "lte": "0003"},
                }
            },
            "needs at least 2",
        ),
    ],
)
def test_analyze_explains_a_request_it_cannot_compute(
    pipeline_client, overrides, message
):
    job_id = _do_extract(pipeline_client, "test_annual", "0001", "0005")
    assert _get_status(pipeline_client, job_id)["status"] == "SUCCESS"

    resp = pipeline_client.post(ANALYZE_URL, json=_analyze_payload(job_id, **overrides))

    assert resp.status_code == 422
    assert message in resp.json()["detail"]


@pytest.mark.integration
def test_extract_refuses_a_window_longer_than_the_dataset(pipeline_client):
    resp = pipeline_client.post(
        EXTRACT_URL,
        json=_extract_payload(
            "test_annual",
            "0001",
            "0005",
            transform={"type": "ZScoreMovingInterval", "width": 5},
        ),
    )

    assert resp.status_code == 422


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


# ---------------------------------------------------------------------------
# Releases: tiles by canonical key, ranges, startup refusal


@pytest.mark.integration
@pytest.mark.parametrize(
    "path, cog, band",
    [
        ("test_annual/ppt/0001", "test_annual--0001--0002.tif", 1),
        ("test_annual/ppt/0005", "test_annual--0005--0005.tif", 1),
        ("test_monthly/ppt/0003-02", "test_monthly--0003-01--0003-12.tif", 2),
    ],
)
def test_tile_resolves_the_timestep_to_its_cog_and_band(
    pipeline_client, monkeypatch, path, cog, band
):
    captured = _capture_tile(monkeypatch)

    response = pipeline_client.get(f"/tiles/{path}/0/0/0")

    assert response.status_code == 200
    assert captured["cog_path"].endswith(f"/cogs/ppt/{cog}")
    assert captured["band"] == band


@pytest.mark.integration
@pytest.mark.parametrize(
    "path, status",
    [
        ("test_annual/ppt/0006", 404),  # after the last timestep
        ("test_monthly/ppt/0003", 422),  # coarser than the dataset: malformed
        ("test_annual/ppt/0001-01-01", 422),  # tiles take the exact key only
        ("test_annual/ppt/year1", 422),
        ("test_annual/nope/0001", 404),
        ("nope/ppt/0001", 404),
    ],
)
def test_tile_rejects_unknown_and_malformed_timesteps(
    pipeline_client, monkeypatch, path, status
):
    _capture_tile(monkeypatch)

    assert pipeline_client.get(f"/tiles/{path}/0/0/0").status_code == status


@pytest.mark.integration
def test_extract_range_that_selects_nothing_is_rejected(pipeline_client):
    resp = pipeline_client.post(
        EXTRACT_URL, json=_extract_payload("test_annual", "0010", "0020")
    )

    assert resp.status_code == 422
    assert "selects no timesteps" in resp.json()["detail"]


@pytest.mark.integration
def test_extract_malformed_range_is_rejected(pipeline_client):
    resp = pipeline_client.post(
        EXTRACT_URL, json=_extract_payload("test_annual", "0002-06-01", "0004")
    )

    assert resp.status_code == 422


@pytest.mark.integration
@pytest.mark.parametrize("endpoint", ["extract", "analyze"])
@pytest.mark.parametrize("reference", [False, True])
@pytest.mark.parametrize(
    "dataset_id,lower,upper",
    [
        ("test_annual", "0000", "0004"),
        ("test_monthly", "0002-13", "0004-12"),
        ("test_monthly", "0002-00-01", "0004-01-01"),
    ],
)
def test_invalid_calendar_ranges_return_422(
    pipeline_client, endpoint, reference, dataset_id, lower, upper
):
    invalid_range = {"gte": lower, "lte": upper}
    overrides = (
        {"transform": {"type": "ZScoreFixedInterval", "time_range": invalid_range}}
        if reference
        else {"time_range": invalid_range}
    )
    if endpoint == "extract":
        payload = _extract_payload(dataset_id, "0001-01-01", "0004-01-01", **overrides)
        url = EXTRACT_URL
    else:
        job_id = _do_extract(pipeline_client, dataset_id, "0001-01-01", "0004-01-01")
        payload = _analyze_payload(job_id, **overrides)
        url = ANALYZE_URL

    response = pipeline_client.post(url, json=payload)

    assert response.status_code == 422, response.text
    assert "not a valid date" in response.json()["detail"]


@pytest.mark.integration
def test_extract_values_follow_the_axis_across_files(pipeline_client):
    job_id = _do_extract(pipeline_client, "test_monthly", "0001-11", "0002-02")

    series = _get_status(pipeline_client, job_id)["result"]["series"][0]
    # Months 11-14 of the axis span two COGs; each pixel holds its index + 1.
    assert series["values"] == [11.0, 12.0, 13.0, 14.0]
    assert series["timesteps"] == ["0001-11", "0001-12", "0002-01", "0002-02"]
    assert series["time_range"] == {"gte": "0001-11", "lte": "0002-02"}


@pytest.mark.integration
def test_analyze_range_finer_than_the_dataset_keeps_its_first_timestep(
    pipeline_client,
):
    job_id = _do_extract(pipeline_client, "test_annual", "0001", "0005")

    resp = pipeline_client.post(
        ANALYZE_URL,
        json=_analyze_payload(
            job_id, time_range={"gte": "0002-01-01", "lte": "0004-01-01"}
        ),
    )

    assert resp.status_code == 200
    assert resp.json()["series"][0]["values"] == [2.0, 3.0, 4.0]
    assert resp.json()["series"][0]["timesteps"] == ["0002", "0003", "0004"]


@pytest.mark.integration
def test_api_refuses_to_start_on_a_wrong_pin(tmp_path, monkeypatch):
    import app.main
    from app.main import app as fastapi_app
    from app.registry.compose import ReleaseRefused
    from app.tests.pipeline.conftest import GRID, _cube
    from app.tests.release_builder import build_release, write_pin

    root = tmp_path / "releases"
    pinned = build_release(root, "test_annual", {"ppt": _cube(5)}, transform=GRID)
    pin = write_pin(
        tmp_path / "releases.yml",
        root,
        [pinned.model_copy(update={"manifest_sha256": "0" * 64})],
    )
    monkeypatch.setattr(app.main.settings, "release_pin_path", str(pin))
    monkeypatch.setattr(app.main.settings, "release_root", str(root))

    with pytest.raises(ReleaseRefused):
        with TestClient(fastapi_app):
            pass


@pytest.mark.integration
def test_api_refuses_to_start_without_a_display_entry(tmp_path, monkeypatch):
    import app.main
    from app.main import app as fastapi_app
    from app.registry.compose import ReleaseRefused
    from app.tests.pipeline.conftest import DISPLAY, GRID, _cube, write_display
    from app.tests.release_builder import build_release, write_pin

    root = tmp_path / "releases"
    pinned = build_release(root, "test_extra", {"ppt": _cube(2)}, transform=GRID)
    pin = write_pin(tmp_path / "releases.yml", root, [pinned])
    monkeypatch.setattr(app.main.settings, "release_pin_path", str(pin))
    monkeypatch.setattr(app.main.settings, "release_root", str(root))
    write_display(tmp_path, monkeypatch, DISPLAY)

    with pytest.raises(ReleaseRefused) as exc_info:
        with TestClient(fastapi_app):
            pass
    assert [r.requirement for r in exc_info.value.refusals] == ["DISP-001"]


@pytest.mark.integration
def test_metadata_serves_schema_1_0_0_in_display_order(pipeline_client):
    body = pipeline_client.get("/metadata").json()

    assert body["schema_version"] == "1.0.0"
    assert [d["id"] for d in body["datasets"]] == ["test_annual", "test_monthly"]
    monthly = body["datasets"][1]
    assert monthly["time"]["origin"] == "0001-01"
    assert monthly["time"]["end"] == "0005-12"
    assert monthly["variables"][0]["display"]["range"] == [1, 60]
