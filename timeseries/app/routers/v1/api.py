from fastapi import APIRouter, Depends

from app.core.services import extract_timeseries
from app.schemas.dataset import DatasetManager, get_dataset_manager
from app.schemas.timeseries import TimeseriesV1Request


router = APIRouter(tags=["timeseries"], prefix="/v1")


@router.post("/timeseries")
async def timeseries_v1(
    v1_request: TimeseriesV1Request,
    dataset_manager: DatasetManager = Depends(get_dataset_manager),
):
    variable_metadata = dataset_manager.get_variable_metadata(
        v1_request.datasetId, v1_request.variableName
    )
    timeseries_request = v1_request.to_timeseries_request(variable_metadata)
    response = await extract_timeseries(timeseries_request, dataset_manager)
    start = timeseries_request.time_range.gte.isoformat()
    end = timeseries_request.time_range.lte.isoformat()
    return {
        "datasetId": v1_request.datasetId,
        "variableName": v1_request.variableName,
        "boundaryGeometry": v1_request.boundaryGeometry,
        "start": start,
        "end": end,
        "values": response.series[0].values,
    }
