import json

from fastapi import BackgroundTasks, FastAPI
import httpx
import pytest

from app import telemetry


@pytest.fixture
def events(monkeypatch):
    events = []
    monkeypatch.setattr(telemetry, "ENABLED", True)
    monkeypatch.setattr(telemetry, "GET_SAMPLE_RATE", 1)
    monkeypatch.setattr(
        telemetry, "log_event", lambda event, **fields: events.append((event, fields))
    )
    return events


async def test_request_timing_excludes_background_work_and_private_inputs(events):
    app = FastAPI()
    app.add_middleware(telemetry.RequestTimingMiddleware)

    @app.post("/jobs/{job_id}")
    async def submit(job_id: str, background_tasks: BackgroundTasks):
        def background():
            assert len(events) == 1
            assert telemetry.request_id.get() == "a" * 32

        background_tasks.add_task(background)
        return {"ok": True}

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/jobs/private-id?secret=hidden", headers={"X-Request-ID": "a" * 32}
        )
    assert response.headers["x-request-id"] == "a" * 32
    assert telemetry.request_id.get() is None
    event, fields = events[0]
    assert event == "http_request"
    assert fields["route"] == "/jobs/{job_id}"
    assert fields["status"] == 200
    assert fields["duration_ms"] >= 0
    assert "private-id" not in json.dumps(events)
    assert "hidden" not in json.dumps(events)


async def test_sampling_keeps_errors_and_writes_but_skips_health(events, monkeypatch):
    monkeypatch.setattr(telemetry, "GET_SAMPLE_RATE", 0)
    app = FastAPI()
    app.add_middleware(telemetry.RequestTimingMiddleware)

    @app.get("/ok")
    @app.post("/ok")
    @app.get("/healthz")
    async def ok():
        return {}

    @app.get("/error", status_code=503)
    async def error():
        return {}

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        await client.get("/ok")
        await client.get("/healthz")
        await client.post("/ok")
        await client.get("/error")
    assert [(fields["method"], fields["status"]) for _, fields in events] == [
        ("POST", 200),
        ("GET", 503),
    ]


async def test_disabled_telemetry_is_passthrough(events, monkeypatch):
    monkeypatch.setattr(telemetry, "ENABLED", False)
    app = FastAPI()
    app.add_middleware(telemetry.RequestTimingMiddleware)

    @app.get("/")
    async def index():
        return {}

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/")
    assert "x-request-id" not in response.headers
    assert not events


async def test_stream_failure_is_logged_once_and_resets_context(events):
    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send(
            {"type": "http.response.body", "body": b"partial", "more_body": True}
        )
        raise RuntimeError("stream failed")

    async def send(message):
        pass

    scope = {"type": "http", "method": "GET", "path": "/tile", "headers": []}
    with pytest.raises(RuntimeError):
        await telemetry.RequestTimingMiddleware(app)(scope, None, send)
    assert len(events) == 1
    assert events[0][1]["outcome"] == "error"
    assert telemetry.request_id.get() is None


def test_json_event_contains_correlation_and_obeys_switch(monkeypatch):
    messages = []
    monkeypatch.setattr(telemetry.logger, "info", messages.append)
    monkeypatch.setattr(telemetry, "ENABLED", True)
    token = telemetry.request_id.set("a" * 32)
    try:
        telemetry.log_event("analysis_finished", duration_ms=10)
        monkeypatch.setattr(telemetry, "ENABLED", False)
        telemetry.log_event("analysis_finished", duration_ms=20)
    finally:
        telemetry.request_id.reset(token)
    assert len(messages) == 1
    event = json.loads(messages[0])
    assert event["request_id"] == "a" * 32
    assert event["service"] == "api"
    assert event["duration_ms"] == 10


def test_completeness_counts_distinguish_input_and_output(monkeypatch):
    monkeypatch.setattr(telemetry, "ENABLED", True)
    fields = telemetry.data_quality_fields(
        {"timesteps": ["a", "b", "c"], "mean": [1, None, 3], "median": [None] * 3},
        [[None, None], [1, float("nan"), float("inf")], []],
    )
    assert fields["base_series_counts"]["mean"] == {
        "timesteps": 3,
        "valid_timesteps": 2,
        "missing_timesteps": 1,
        "all_missing": False,
    }
    assert fields["base_series_counts"]["median"]["all_missing"] is True
    assert fields["output_series_counts"] == [
        {
            "series_index": 0,
            "timesteps": 2,
            "valid_timesteps": 0,
            "missing_timesteps": 2,
            "all_missing": True,
        },
        {
            "series_index": 1,
            "timesteps": 3,
            "valid_timesteps": 1,
            "missing_timesteps": 2,
            "all_missing": False,
        },
        {
            "series_index": 2,
            "timesteps": 0,
            "valid_timesteps": 0,
            "missing_timesteps": 0,
            "all_missing": False,
        },
    ]
    json.dumps(fields, allow_nan=False)


def test_disabled_completeness_does_not_read_values(monkeypatch):
    monkeypatch.setattr(telemetry, "ENABLED", False)

    def inaccessible_values():
        raise AssertionError("disabled telemetry must not iterate values")
        yield

    assert telemetry.data_quality_fields(output_values=inaccessible_values()) == {}


@pytest.mark.parametrize("failure", [TypeError, ValueError, OSError, RuntimeError])
def test_logging_failure_is_best_effort(monkeypatch, failure):
    monkeypatch.setattr(telemetry, "ENABLED", True)

    def broken_sink(*args, **kwargs):
        raise failure("sink failed")

    monkeypatch.setattr(telemetry.logger, "info", broken_sink)
    telemetry.log_event("test_event", duration_ms=1)


def test_non_json_event_is_dropped_without_raising(monkeypatch):
    monkeypatch.setattr(telemetry, "ENABLED", True)
    telemetry.log_event("test_event", duration_ms=float("nan"))
