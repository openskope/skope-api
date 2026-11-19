"""Small, dependency-free JSON telemetry shared by the API and TiTiler images."""

from contextvars import ContextVar
from datetime import datetime, timezone
import json
import logging
import math
import os
import re
import time
import uuid

request_id = ContextVar("request_id", default=None)
ENABLED = os.getenv("SKOPE_TELEMETRY_ENABLED", "true").lower() in {"1", "true", "yes"}
GET_SAMPLE_RATE = float(os.getenv("SKOPE_TELEMETRY_GET_SAMPLE_RATE", "0.1"))
if not 0 <= GET_SAMPLE_RATE <= 1:
    raise ValueError("SKOPE_TELEMETRY_GET_SAMPLE_RATE must be between 0 and 1")

logger = logging.getLogger("skope.telemetry")
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)


def log_event(event, *, service="api", **fields):
    if not ENABLED:
        return
    try:
        logger.info(
            json.dumps(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "service": service,
                    "event": event,
                    "pid": os.getpid(),
                    "request_id": request_id.get(),
                    **fields,
                },
                allow_nan=False,
                separators=(",", ":"),
            )
        )
    except Exception:
        # Observability must not fail a request, prevent a terminal job update,
        # or mask the original exception. Do not recursively log sink failures.
        pass


def data_quality_fields(base_series=None, output_values=()):
    """Count missing data without logging values or doing work while disabled."""
    if not ENABLED:
        return {}

    def counts(values):
        total = len(values)
        valid = sum(value is not None and math.isfinite(value) for value in values)
        return {
            "timesteps": total,
            "valid_timesteps": valid,
            "missing_timesteps": total - valid,
            "all_missing": total > 0 and valid == 0,
        }

    return {
        "base_series_counts": {
            statistic: counts(base_series[statistic])
            for statistic in ("mean", "median")
            if base_series is not None and statistic in base_series
        },
        # Positional indexes match response order without logging user labels.
        "output_series_counts": [
            {"series_index": index, **counts(values)}
            for index, values in enumerate(output_values)
        ],
    }


class RequestTimingMiddleware:
    def __init__(self, app, service="api"):
        self.app = app
        self.service = service

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not ENABLED:
            return await self.app(scope, receive, send)
        incoming = dict(scope.get("headers", [])).get(b"x-request-id", b"")
        identifier = (
            incoming.decode("ascii")
            if re.fullmatch(rb"[0-9a-f]{32}", incoming)
            else uuid.uuid4().hex
        )
        token = request_id.set(identifier)
        started = time.monotonic()
        status = 500
        recorded = False

        def record(outcome):
            nonlocal recorded
            if recorded:
                return
            recorded = True
            route = getattr(scope.get("route"), "path", "unmatched")
            if scope["path"] in {"/health", "/healthz"}:
                return
            # Stable sampling keeps the gateway and TiTiler's entries together.
            if (
                outcome == "complete"
                and status < 400
                and scope["method"] == "GET"
                and int(identifier[:8], 16) / 2**32 >= GET_SAMPLE_RATE
            ):
                return
            log_event(
                "http_request",
                service=self.service,
                method=scope["method"],
                route=route,
                status=status,
                outcome=outcome,
                duration_ms=round((time.monotonic() - started) * 1000, 3),
            )

        async def timed_send(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                message = dict(message)
                message["headers"] = [
                    (key, value)
                    for key, value in message.get("headers", [])
                    if key.lower() != b"x-request-id"
                ] + [(b"x-request-id", identifier.encode("ascii"))]
            await send(message)
            if message["type"] == "http.response.body" and not message.get("more_body"):
                # Do not include FastAPI background-job runtime in HTTP latency.
                record("complete")

        try:
            await self.app(scope, receive, timed_send)
        except BaseException:
            record("error")
            raise
        finally:
            try:
                record("incomplete")
            finally:
                request_id.reset(token)
