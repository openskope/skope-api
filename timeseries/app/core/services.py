from anyio import create_task_group, fail_after, TASK_STATUS_IGNORED
from anyio.abc import TaskStatus
from typing import Dict

import logging
import rasterio

from app.exceptions import TimeseriesTimeoutError
from app.schemas.dataset import DatasetManager
from app.schemas.timeseries import TimeseriesRequest, TimeseriesResponse, NoTransform

logger = logging.getLogger(__name__)


class RequestedSeries:

    timeseries_request = None
    original_timeseries_raw_data = None
    output_timeseries = None
    pandas_series = None
    band_range_adjustment = None

    def __init__(
        self,
        timeseries_request,
        original_timeseries_raw_data,
        output_timeseries,
        pandas_series,
        band_range_adjustment,
    ):
        self.timeseries_request = timeseries_request
        self.original_timeseries_raw_data = original_timeseries_raw_data
        self.output_timeseries = output_timeseries
        self.pandas_series = pandas_series
        self.band_range_adjustment = band_range_adjustment

    @property
    def original_timeseries_data(self):
        # FIXME: if smoothing was applied, need to take a little bit off to get back
        # the correct stats over the original time interval
        # pull last band range adjustment instead
        original_timeseries = self.original_timeseries_raw_data["data"]
        band_range_adjustment = self.band_range_adjustment
        if band_range_adjustment is not None:
            start = -band_range_adjustment[0]
            end = len(original_timeseries) - band_range_adjustment[1]
            logger.debug(
                "adjusting by %s pulling timeseries from %s to %s",
                band_range_adjustment,
                start,
                end,
            )
            return original_timeseries[start:end]
        return original_timeseries

    def get_summary_stats(self):
        return self.timeseries_request.get_summary_stats(
            self.pandas_series, self.original_timeseries_data
        )

    def to_timeseries_response_dict(self):
        timeseries_request = self.timeseries_request
        return dict(
            dataset_id=timeseries_request.dataset_id,
            variable_id=timeseries_request.variable_id,
            area=self.original_timeseries_raw_data["area"],
            n_cells=self.original_timeseries_raw_data["n_cells"],
            transform=timeseries_request.transform,
            zonal_statistic=timeseries_request.zonal_statistic,
            series=self.output_timeseries,
            summary_stats=self.get_summary_stats(),
        )


"""
Service layer class intermediary that proxies a TimeseriesRequest and service layer calls
"""


class RequestedSeriesMetadata:

    timeseries_request = None
    variable_metadata = None

    def __init__(
        self, timeseries_request: TimeseriesRequest, dataset_manager: DatasetManager
    ):
        self.timeseries_request = timeseries_request
        self.variable_metadata = timeseries_request.get_variable_metadata(
            dataset_manager
        )

    @property
    def dataset_path(self):
        return self.variable_metadata.path

    @property
    def selected_area(self):
        return self.timeseries_request.selected_area

    @property
    def zonal_statistic(self):
        return self.timeseries_request.zonal_statistic

    @property
    def transform(self):
        return self.timeseries_request.transform

    @property
    def requested_band_range(self):
        return self.timeseries_request.get_requested_band_range(self.variable_metadata)

    @property
    def band_range_to_extract(self):
        return self.timeseries_request.get_band_range_to_extract(self.variable_metadata)

    @property
    def transform_band_range(self):
        return self.timeseries_request.get_transform_band_range(self.variable_metadata)

    @property
    def has_transform(self):
        transform = self.transform
        return transform and not isinstance(transform, NoTransform)

    def apply_transform(self, original_timeseries_raw_data, dataset):
        logger.debug("applying transform %s", self.transform)
        original_series_data = original_timeseries_raw_data.get("data")
        if not self.has_transform:
            return original_series_data

        transform_band_range = self.transform_band_range
        transformed_series_data = original_series_data
        if transform_band_range is not None:
            # only extract an additional raster slice if the band ranges differ
            transformed_series_raw_data = self.selected_area.extract_raster_slice(
                dataset,
                zonal_statistic=self.zonal_statistic,
                band_range=transform_band_range,
            )
            transformed_series_data = transformed_series_raw_data.get("data")
        return self.transform.apply(original_series_data, transformed_series_data)

    async def process(self):
        with rasterio.Env():
            with rasterio.open(self.dataset_path) as dataset:
                selected_area = self.selected_area
                # { n_cells: number, area: number, data: List }
                requested_band_range = self.requested_band_range
                band_range_to_extract = self.band_range_to_extract
                logger.debug(
                    "originally requested band range to actually extracted band range: %s -> %s",
                    requested_band_range,
                    band_range_to_extract,
                )

                original_timeseries_raw_data = selected_area.extract_raster_slice(
                    dataset,
                    zonal_statistic=self.zonal_statistic,
                    band_range=band_range_to_extract,
                )
                transformed_series = self.apply_transform(
                    original_timeseries_raw_data, dataset
                )
                logger.debug("transformed series: %s", transformed_series)
                # apply smoothing if any
                (
                    timeseries,
                    pd_series,
                    band_range_adjustment,
                ) = self.timeseries_request.apply_smoothing(
                    transformed_series, self.variable_metadata, band_range_to_extract
                )
                return RequestedSeries(
                    timeseries_request=self.timeseries_request,
                    original_timeseries_raw_data=original_timeseries_raw_data,
                    output_timeseries=timeseries,
                    pandas_series=pd_series,
                    band_range_adjustment=band_range_adjustment,
                )


async def extract_timeseries(
    request: TimeseriesRequest, dataset_manager: DatasetManager
):
    timeout = request.max_processing_time
    logger.debug("setting request timeout to %s", timeout)
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
    """
    FIXME: make a single call to generate a list of series metadata sufficient
    for fulfill the rasterio calls with appropriate band ranges, transform options,
    smoothing options
    each object:
    1. band range to apply
    2. transform
    3. smoother
    """
    requested_series_metadata = RequestedSeriesMetadata(
        timeseries_request, dataset_manager
    )
    requested_series = await requested_series_metadata.process()
    output["response"] = TimeseriesResponse(
        **requested_series.to_timeseries_response_dict()
    )

    """
    response = TimeseriesResponse(
        dataset_id=timeseries_request.dataset_id,
        variable_id=timeseries_request.variable_id,
        area=area,
        n_cells=n_cells,
        series=series,
        transform=timeseries_request.transform,
        zonal_statistic=timeseries_request.zonal_statistic,
        summary_stats=timeseries_request.get_summary_stats(
            pd_series, original_timeseries
        ),
    )
    # band_range = timeseries_request.get_band_range_to_extract(metadata)
    # band_range_transform = timeseries_request.get_band_ranges_for_transform(metadata)
    logger.debug("requested series metadata %s", requested_series_metadata)
    with rasterio.Env():
        with rasterio.open(requested_series_metadata.dataset_path) as dataset:
            timeseries_data = extract(dataset, requested_series_metadata)
            # retrieve the raw, original data from the data cube
            data_slice = timeseries_request.extract_slice(dataset, band_range)
            original_timeseries = data_slice["data"]
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

    txs = timeseries_request.apply_transform(original_timeseries, transform_xs)

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
        summary_stats=timeseries_request.get_summary_stats(
            pd_series, original_timeseries
        ),
    )
    """
