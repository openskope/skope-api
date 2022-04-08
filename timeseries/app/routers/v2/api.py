from fastapi import APIRouter, Depends
import logging

from app.core.services import extract_timeseries
from app.schemas.timeseries import TimeseriesRequest
from app.schemas.dataset import get_dataset_manager, load_api_metadata

logger = logging.getLogger(__name__)


metadata_router = APIRouter(tags=["metadata"])
timeseries_router = APIRouter(tags=["timeseries"])


@metadata_router.get("/metadata")
def metadata(api_metadata=Depends(load_api_metadata)):
    return list(api_metadata.values())


@timeseries_router.post("/timeseries")
async def retrieve_timeseries(
    request: TimeseriesRequest, dataset_manager=Depends(get_dataset_manager)
):
    return await extract_timeseries(request, dataset_manager)


router = APIRouter()
router.include_router(metadata_router)
router.include_router(timeseries_router)
