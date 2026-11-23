import logging
import anyio
from pathlib import Path

import httpx
import sentry_sdk
from sentry_sdk.integrations.asgi import SentryAsgiMiddleware
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.status import HTTP_422_UNPROCESSABLE_CONTENT
from contextlib import asynccontextmanager

from app.core.job_control import JobContractError
from app.telemetry import RequestTimingMiddleware, log_event
from app.config import get_settings
from app.exceptions import TimeseriesValidationError
from app.registry.compose import (
    Refusal,
    ReleaseRefused,
    load_pin,
    log_refusals,
    verify_releases,
)
from app.registry.display import DisplayFileError, check_display, load_display
from app.registry.metadata import build_metadata
from app.store.jobs import cleanup_stale_jobs, create_job_store
from app.routers import api

settings = get_settings()
logger = logging.getLogger(__name__)


def compose_registry():
    """Check the pinned releases and the display files, and compose the registry.

    On any failed check (PIN-004, DISP-012, DISP-013), log each failure and
    raise, so the process exits instead of serving some datasets.
    """
    pin = load_pin(Path(settings.release_pin_path))
    try:
        registry = verify_releases(pin, Path(settings.release_root))
        try:
            preferences, palettes = load_display(
                Path(settings.display_preferences_path),
                Path(settings.display_palettes_path),
            )
        except DisplayFileError as exc:
            raise ReleaseRefused(
                [Refusal("DISP-012", "-", "-", "display files are valid", "valid", exc)]
            ) from exc
        refusals = check_display(registry, preferences, palettes)
        if refusals:
            raise ReleaseRefused(refusals)
    except ReleaseRefused as exc:
        log_refusals(exc.refusals)
        raise
    logger.info(
        "Serving %s",
        ", ".join(r.overview.release_id for r in registry.releases.values()),
    )
    return registry, preferences, palettes


@asynccontextmanager
async def lifespan(app: FastAPI):
    cleanup_stale_jobs()
    logger.info("Stale jobs cleaned up.")

    # The limits ensure we don't overwhelm Titiler while handling concurrency
    limits = httpx.Limits(max_keepalive_connections=20, max_connections=100)
    async_client = httpx.AsyncClient(
        timeout=httpx.Timeout(10.0, read=60.0), limits=limits
    )
    job_store = create_job_store(settings.redis_url)
    try:
        (
            app.state.registry,
            app.state.display,
            app.state.palettes,
        ) = compose_registry()
        app.state.metadata = build_metadata(
            app.state.registry, app.state.display, app.state.palettes
        )
        await job_store.healthcheck()
        app.state.client = async_client
        app.state.job_store = job_store
        app.state.analysis_limiter = anyio.CapacityLimiter(1)

        yield
    finally:
        await async_client.aclose()
        await job_store.close()


app = FastAPI(title="SKOPE API Services", lifespan=lifespan)
app.add_middleware(RequestTimingMiddleware)

if settings.is_production:
    sentry_sdk.init(dsn=settings.sentry_dsn)
    try:
        app.add_middleware(SentryAsgiMiddleware)
    except Exception:
        logger.error("Unable to initialize Sentry middleware")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Retry-After", "X-Request-ID"],
)


@app.exception_handler(JobContractError)
async def job_contract_error_handler(request: Request, exc: JobContractError):
    log_event("job_contract_error", status=exc.status, error_code=exc.code)
    return JSONResponse(
        status_code=exc.status,
        content={"detail": {"code": exc.code, "msg": exc.message}},
        headers=(
            {"Retry-After": "5"}
            if exc.code in {"CAPACITY_EXHAUSTED", "COORDINATION_UNAVAILABLE"}
            else None
        ),
    )


@app.exception_handler(TimeseriesValidationError)
async def timeseries_error_handler(request: Request, exc: TimeseriesValidationError):
    return JSONResponse(
        status_code=HTTP_422_UNPROCESSABLE_CONTENT,
        content={"detail": exc.to_request_validation_error().errors()},
    )


app.include_router(api.router)
