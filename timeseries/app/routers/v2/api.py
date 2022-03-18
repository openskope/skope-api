from fastapi import APIRouter, Depends

from app.schemas.timeseries import TimeseriesRequest
from app.schemas.dataset import load_api_metadata, DatasetManager, get_dataset_manager


metadata_router = APIRouter(tags=['metadata'])
timeseries_router = APIRouter(tags=['timeseries'])

@metadata_router.get("/metadata")
def metadata(api_metadata=Depends(load_api_metadata)):
    return list(api_metadata.values())

@timeseries_router.post('/timeseries')
async def timeseries_v2(data: TimeseriesRequest, dataset_manager: DatasetManager = Depends(get_dataset_manager) ):
    variable_metadata = dataset_manager.get_variable_metadata(data.dataset_id, data.variable_id)
    return await data.extract(variable_metadata)

router = APIRouter()
router.include_router(metadata_router)
router.include_router(timeseries_router)
