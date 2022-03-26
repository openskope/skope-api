from anyio import create_task_group, fail_after
from fastapi import APIRouter, Depends
import logging

from app.core.services import extract_timeseries
from app.exceptions import TimeseriesTimeoutError
from app.schemas.timeseries import TimeseriesRequest
from app.schemas.dataset import get_dataset_manager, load_api_metadata
from app.settings import settings

logger = logging.getLogger(__name__)


metadata_router = APIRouter(tags=['metadata'])
timeseries_router = APIRouter(tags=['timeseries'])

@metadata_router.get("/metadata")
def metadata(api_metadata=Depends(load_api_metadata)):
    return list(api_metadata.values())

@timeseries_router.post('/timeseries')
async def retrieve_timeseries(data: TimeseriesRequest, dataset_manager=Depends(get_dataset_manager)):
    timeout = settings.max_processing_time
    logger.debug("time out after %s", timeout)
    async with create_task_group() as tg:
        try:
            with fail_after(timeout) as scope:
                output = { 'response': {} }
                await tg.start(extract_timeseries, data, dataset_manager, output)
                return output['response']
        except TimeoutError as e:
            raise TimeseriesTimeoutError(e.message, timeout)

        
        

router = APIRouter()
router.include_router(metadata_router)
router.include_router(timeseries_router)
