from fastapi import FastAPI, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.status import HTTP_504_GATEWAY_TIMEOUT, HTTP_422_UNPROCESSABLE_ENTITY

from app.config import get_settings, Settings
from app.exceptions import TimeseriesValidationError, TimeseriesTimeoutError
from app.routers.v1 import api as v1_api
from app.routers.v2 import api as v2_api


app = FastAPI(title="SKOPE API Services")


# Since the whole API is public there is no danger in allowing all cross origin requests for now
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/settings")
async def info(settings: Settings = Depends(get_settings)):
    info_dict = dict(settings.__dict__)
    info_dict.update(logfile=settings.logging_config_file)
    return info_dict


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
