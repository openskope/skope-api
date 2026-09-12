import logging
import httpx
import sentry_sdk
from sentry_sdk.integrations.asgi import SentryAsgiMiddleware
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.status import HTTP_422_UNPROCESSABLE_CONTENT
from contextlib import asynccontextmanager

from app.config import get_settings
from app.exceptions import TimeseriesValidationError
from app.store.jobs import cleanup_stale_jobs, create_job_store
from app.store.data_reader import get_data_reader
from app.store.index_loaders import load_registry, resolve_colormaps
from app.routers.v3 import api as v3_api

settings = get_settings()
logger = logging.getLogger(__name__)


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
        await job_store.healthcheck()
        app.state.client = async_client
        app.state.job_store = job_store
        app.state.global_registry = load_registry(settings.registry_path)
        app.state.data_reader = get_data_reader(settings.storage_base_url)

        await resolve_colormaps(
            app.state.global_registry,
            settings.colormaps_path,
            async_client,
            settings.tile_server_url,
        )

        yield
    finally:
        await async_client.aclose()
        await job_store.close()


app = FastAPI(title="SKOPE API Services", lifespan=lifespan)

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
)


@app.exception_handler(TimeseriesValidationError)
async def timeseries_error_handler(request: Request, exc: TimeseriesValidationError):
    return JSONResponse(
        status_code=HTTP_422_UNPROCESSABLE_CONTENT,
        content={"detail": exc.to_request_validation_error().errors()},
    )


app.include_router(v3_api.router)
