from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.status import HTTP_504_GATEWAY_TIMEOUT, HTTP_422_UNPROCESSABLE_ENTITY
import logging

from app.exceptions import TimeseriesValidationError, TimeseriesTimeoutError
from app.settings import settings
from app.routers.v1 import api as v1_api
from app.routers.v2 import api as v2_api

logger = logging.getLogger(__name__)

app = FastAPI(title="OpenSKOPE Time Series API")

# Since the whole API is public there is no danger in allowing all cross origin requests for now
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(TimeseriesTimeoutError)
async def timeseries_timeout_error_handler(
    request: Request, exc: TimeseriesTimeoutError
):
    return JSONResponse(
        status_code=HTTP_504_GATEWAY_TIMEOUT,
        content={"detail": exc.message, "processing_time": exc.processing_time},
    )


@app.exception_handler(TimeseriesValidationError)
async def timeseries_error_handler(request: Request, exc: TimeseriesValidationError):
    return JSONResponse(
        status_code=HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.to_request_validation_error().errors()},
    )


app.include_router(v1_api.router)
app.include_router(v2_api.router)
