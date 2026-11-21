"""Zonal statistics: read an area's cells from a dataset's COGs, per timestep."""

import logging
import time
import warnings
import anyio
import numpy as np
import rasterio
from rasterio.windows import Window
from typing import Callable, Iterator, Sequence

# Local imports
from app.telemetry import log_event
from app.config import get_settings
from app.core.analysis import execute_analyze_request
from app.core.spatial import calculate_spatial_coverage, resolve_spatial_window
from app.exceptions import SelectedAreaPolygonIsTooLarge
from app.schemas.timeseries import TimeseriesRequest, TimeseriesResponse

logger = logging.getLogger(__name__)
settings = get_settings()

RASTERIO_ENV_KWARGS = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": "tif,tiff,ovr",
    "VSI_CACHE": "TRUE",
    # Bound each HTTP transfer, including its body. Extraction cancellation is
    # cooperative between chunks; these limits stop a stalled remote read.
    "GDAL_HTTP_CONNECTTIMEOUT": str(settings.raster_connect_timeout_seconds),
    "GDAL_HTTP_TIMEOUT": str(settings.raster_http_timeout_seconds),
    # Hidden retries (with GDAL's default 30s delay) outlive the job budget.
    "GDAL_HTTP_MAX_RETRY": "0",
    "AWS_NO_SIGN_REQUEST": "YES",
}


def calculate_safe_chunk_size(
    width: int, height: int, max_cells: int = settings.default_max_cells
) -> int:
    n_cells_per_band = width * height
    if n_cells_per_band == 0:
        raise ValueError("The requested geometry resulted in a 0-pixel window.")

    n_bands_per_chunk = max_cells // n_cells_per_band
    if n_bands_per_chunk == 0:
        raise SelectedAreaPolygonIsTooLarge(
            n_cells=n_cells_per_band, max_cells=max_cells
        )

    return n_bands_per_chunk


def generate_band_chunks(
    band_indices: Sequence[int], chunk_size: int
) -> Iterator[Sequence[int]]:
    for i in range(0, len(band_indices), chunk_size):
        yield band_indices[i : i + chunk_size]


def extract_summarystat_timeseries(
    file_uri: str,
    band_indices: list[int],
    window: Window,
    precomputed_mask: np.ndarray,
    chunk_size: int,
    check_cancelled: Callable[[], None] | None = None,
) -> tuple[str, np.ndarray, np.ndarray]:
    """Returns (uri, mean_array, median_array) — both zonal statistics are always computed
    so the analysis layer can select either without a second raster read.

    `check_cancelled` runs before and after each chunk read. Cancellation is
    cooperative: an in-progress native read must finish or fail first."""
    mean_results: list[float] = []
    median_results: list[float] = []
    n_inside = int(precomputed_mask.sum())
    try:
        with rasterio.Env(**RASTERIO_ENV_KWARGS):
            with rasterio.open(file_uri) as src:
                for band_chunk in generate_band_chunks(band_indices, chunk_size):
                    if check_cancelled is not None:
                        check_cancelled()
                    raw = src.read(band_chunk, window=window)

                    if check_cancelled is not None:
                        check_cancelled()

                    # Only the cells inside the area, one band at a time: the
                    # statistics never sift through the cells outside it.
                    # Nodata is compared in the file's own type, where it's
                    # exact.
                    # Float64 preserves float64 inputs and all supported integer
                    # values; float32 would round them before either reduction.
                    cells = np.empty((len(band_chunk), n_inside), dtype=np.float64)
                    for i, band in enumerate(raw):
                        inside = band[precomputed_mask]
                        cells[i] = inside
                        if src.nodata is not None:
                            cells[i, inside == src.nodata] = np.nan
                    del raw

                    # A timestep with no valid cell in the area (for example,
                    # a year the dataset has no data for) yields NaN, which the
                    # response carries as null. numpy warns about each one.
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        # Keep both input values and mean accumulation in
                        # float64; z-scores can amplify reduction error.
                        mean_results.extend(np.nanmean(cells, axis=1, dtype=np.float64))
                        median_results.extend(np.nanmedian(cells, axis=1))

        return (file_uri, np.array(mean_results), np.array(median_results))
    except Exception as e:
        logger.error(f"Failed to process file {file_uri}: {e}")
        raise


async def execute_timeseries_job(
    request: TimeseriesRequest,
    file_mapping: dict[str, list[int]],
    timestep_list: list[str],
    dataset_crs: str,
    dataset_transform_array: list[float],
    max_concurrency: int = 10,
) -> tuple[TimeseriesResponse, dict]:
    """Reads the area over `timestep_list` (the whole axis) into a base series,
    and answers the request from it the way /analyze does."""
    uris = list(file_mapping.keys())
    if not uris:
        raise ValueError("No matching files found in the requested time range.")

    reprojected_shapes, dataset_transform, window = resolve_spatial_window(
        request.selected_area.shapes,
        dataset_transform_array,
        dataset_crs,
    )

    mask, n_cells, total_area = calculate_spatial_coverage(
        reprojected_shapes, dataset_transform, window, dataset_crs
    )
    chunk_size = calculate_safe_chunk_size(
        width=int(window.width), height=int(window.height)
    )

    limiter = anyio.CapacityLimiter(max_concurrency)
    results: list[tuple[str, np.ndarray]] = []

    async def _worker_wrapper(uri: str, bands: list[int]):
        async with limiter:
            result = await anyio.to_thread.run_sync(
                extract_summarystat_timeseries,
                uri,
                bands,
                window,
                mask,
                chunk_size,
                anyio.from_thread.check_cancelled,
            )
            results.append(result)

    # A passed deadline cancels this task; the cancellation is not an Exception,
    # so it leaves as "stopped".
    started = time.monotonic()
    outcome = "stopped"
    try:
        async with anyio.create_task_group() as tg:
            for uri, bands in file_mapping.items():
                tg.start_soon(_worker_wrapper, uri, bands)
        outcome = "done"
    except Exception as e:
        outcome = "failed"
        logger.error(f"Task group failed during Map phase: {e}")
        raise RuntimeError("Timeseries extraction failed.") from e
    finally:
        log_event(
            "raster_reads_finished",
            outcome=outcome,
            duration_ms=round((time.monotonic() - started) * 1000, 3),
            cells=n_cells,
            window_cells=int(window.width * window.height),
            timesteps=len(timestep_list),
            files=len(file_mapping),
            chunk_bands=chunk_size,
        )

    # Sort by insertion order of file_mapping to preserve chronological ordering
    uri_order = {uri: i for i, uri in enumerate(uris)}
    results.sort(key=lambda x: uri_order[x[0]])

    full_mean = np.concatenate([res[1] for res in results])
    full_median = np.concatenate([res[2] for res in results])
    # Values are labelled by position, so a count mismatch would silently shift
    # every label after it. Fail instead.
    if len(full_mean) != len(timestep_list) or len(full_median) != len(timestep_list):
        raise RuntimeError(
            f"Extracted {len(full_mean)} values for {len(timestep_list)} timesteps."
        )

    # TODO: "mean"/"median" keys are hardwired to match ZonalStatistic enum values.
    # If new statistics are added to the enum, updates to extract_summarystat_timeseries and this block are needed
    base_series_payload = {
        "timesteps": timestep_list,
        "mean": [None if np.isnan(v) else float(v) for v in full_mean],
        "median": [None if np.isnan(v) else float(v) for v in full_median],
    }
    timeseries_response = execute_analyze_request(
        payload=request,
        base_series_payload=base_series_payload,
        extraction_metadata={
            "dataset_id": request.dataset_id,
            "variable_id": request.variable_id,
            "area": total_area,
            "n_cells": n_cells,
        },
    )
    return timeseries_response, base_series_payload
