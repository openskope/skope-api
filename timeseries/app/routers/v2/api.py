from fastapi import APIRouter, Depends

from app.schemas.timeseries import TimeseriesRequest
from app.schemas.dataset import load_api_metadata


metadata_router = APIRouter(tags=['metadata'])
timeseries_router = APIRouter(tags=['timeseries'])

@metadata_router.get("/metadata")
def metadata(api_metadata=Depends(load_api_metadata)):
    return list(api_metadata.values())

@timeseries_router.post('/timeseries')
async def timeseries_v2(data: TimeseriesRequest):
    return await data.extract()

router = APIRouter()
router.include_router(metadata_router)
router.include_router(timeseries_router)
