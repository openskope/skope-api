from fastapi import APIRouter, Depends

from app.schemas.dataset import DatasetManager, get_dataset_manager
from app.schemas.timeseries import TimeseriesV1Request


router = APIRouter(tags=['timeseries'], prefix='/v1')

@router.post('/timeseries')
async def timeseries_v1(data: TimeseriesV1Request, dataset_manager: DatasetManager = Depends(get_dataset_manager)):
    variable_metadata = dataset_manager.get_variable_metadata(data.datasetId, data.variableName)
    # FIXME: convert to a service layer operation that operates on the 
    # TimeseriesV1Request instead 
    return await data.extract(variable_metadata)
