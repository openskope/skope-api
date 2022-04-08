from anyio import create_task_group, fail_after, TASK_STATUS_IGNORED
from anyio.abc import TaskStatus
from typing import Dict

import logging
import rasterio

from app.exceptions import TimeseriesTimeoutError
from app.schemas.dataset import DatasetManager
from app.schemas.timeseries import TimeseriesRequest, TimeseriesResponse


logger = logging.getLogger(__name__)


async def extract_timeseries(
    request: TimeseriesRequest, dataset_manager: DatasetManager
):
    timeout = request.max_processing_time
    logger.debug("time out after %s", timeout)
    async with create_task_group() as tg:
        try:
            with fail_after(timeout) as scope:
                output = {"response": {}}
                await tg.start(
                    extract_timeseries_task, request, dataset_manager, output
                )
                return output["response"]
        except TimeoutError as e:
            raise TimeseriesTimeoutError(e, timeout)


async def extract_timeseries_task(
    timeseries_request: TimeseriesRequest,
    dataset_manager: DatasetManager,
    output: Dict,
    *,
    task_status: TaskStatus = TASK_STATUS_IGNORED
):
    task_status.started()
    metadata = timeseries_request.get_variable_metadata(dataset_manager)
    band_range = timeseries_request.get_band_range_to_extract(metadata)
    band_range_transform = timeseries_request.get_band_ranges_for_transform(metadata)
    logger.debug(
        "extract band range %s, transform band range: %s",
        band_range,
        band_range_transform,
    )
    with rasterio.Env():
        with rasterio.open(metadata.path) as dataset:
            data_slice = timeseries_request.extract_slice(dataset, band_range)
            xs = data_slice["data"]
            n_cells = data_slice["n_cells"]
            area = data_slice["area"]
            # only needed for fixed interval z score
            transform_xs = (
                timeseries_request.extract_slice(
                    dataset, band_range=band_range_transform
                )["data"]
                if band_range_transform
                else None
            )

    txs = timeseries_request.apply_transform(xs, transform_xs)

    series, pd_series = timeseries_request.apply_series(
        txs, metadata=metadata, band_range=band_range
    )
    output["response"] = TimeseriesResponse(
        dataset_id=timeseries_request.dataset_id,
        variable_id=timeseries_request.variable_id,
        area=area,
        n_cells=n_cells,
        series=series,
        transform=timeseries_request.transform,
        zonal_statistic=timeseries_request.zonal_statistic,
        summary_stats=timeseries_request.get_summary_stats(pd_series, xs),
    )
