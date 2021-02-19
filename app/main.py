from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.status import HTTP_504_GATEWAY_TIMEOUT, HTTP_422_UNPROCESSABLE_ENTITY
import logging

from app.exceptions import TimeseriesValidationError, TimeseriesTimeoutError
from app.settings import settings
from app.routers import datasets, metadata

settings.init()
logger = logging.getLogger(__name__)


app = FastAPI()

@app.exception_handler(TimeseriesTimeoutError)
async def timeseries_timeout_error_handler(request: Request, exc: TimeseriesTimeoutError):
    return JSONResponse(
        status_code=HTTP_504_GATEWAY_TIMEOUT,
        content={
        'detail': exc.message,
        'processing_time': exc.processing_time
    })


@app.exception_handler(TimeseriesValidationError)
async def timeseries_error_handler(request: Request, exc: TimeseriesValidationError):
    return JSONResponse(
        status_code=HTTP_422_UNPROCESSABLE_ENTITY,
        content={'detail': exc.to_request_validation_error().errors()})


app.include_router(datasets.router)
app.include_router(metadata.router)
