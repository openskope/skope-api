from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import logging

from app.exceptions import TimeseriesError
from app.settings import settings
from app.routers import datasets, metadata

settings.init()
logger = logging.getLogger(__name__)


app = FastAPI()

@app.exception_handler(TimeseriesError)
async def timeseries_error_handler(request: Request, exc: TimeseriesError):
    return JSONResponse(
        status_code=422,
        content={'detail': exc.to_request_validation_error().errors()})


app.include_router(datasets.router)
app.include_router(metadata.router)
