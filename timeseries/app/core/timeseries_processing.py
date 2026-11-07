import logging
import time
import warnings
import anyio
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
import rasterio.windows
from rasterio.features import geometry_mask
from pyproj import CRS
from shapely.ops import unary_union
from typing import Callable, Dict, List, Iterator, Tuple, Sequence

# Local imports
from app.config import get_settings
from app.core.validation import resolve_spatial_window
from app.exceptions import SelectedAreaPolygonIsTooLarge
from app.schemas.timeseries import (
    TimeseriesAnalyzeRequest,
    TimeseriesRequest,
    TimeseriesResponse,
    Series,
    SummaryStat,
    TimeRange,
    MovingAverageSmoother,
    NoTransform,
    ZScoreMovingInterval,
    ZScoreFixedInterval,
)

logger = logging.getLogger(__name__)
settings = get_settings()

RASTERIO_ENV_KWARGS = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": "tif,tiff,ovr",
    "VSI_CACHE": "TRUE",
    "GDAL_HTTP_RETRY__COUNT": "3",
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


def calculate_spatial_coverage(
    shapes: list,
    dataset_transform: rasterio.Affine,
    window: Window,
    dataset_crs: str,
) -> Tuple[np.ndarray, int, float]:
    window_transform = rasterio.windows.transform(window, dataset_transform)
    mask = geometry_mask(
        shapes,
        transform=window_transform,
        invert=True,
        out_shape=(window.height, window.width),
    )
    if np.sum(mask) == 0:
        # Fallback for point geometries or small, sub-pixel polygons that don't cover any pixel center
        mask = geometry_mask(
            shapes,
            transform=window_transform,
            invert=True,
            out_shape=(window.height, window.width),
            all_touched=True,
        )
    n_cells = int(np.sum(mask))

    crs = CRS.from_string(dataset_crs)
    union = unary_union(shapes)
    if crs.is_geographic:
        area, _ = crs.get_geod().geometry_area_perimeter(union)
        total_area = abs(area)
    else:
        total_area = float(union.area)

    return mask, n_cells, total_area


def extract_summarystat_timeseries(
    file_uri: str,
    band_indices: List[int],
    window: Window,
    precomputed_mask: np.ndarray,
    chunk_size: int,
    check_cancelled: Callable[[], None] | None = None,
) -> Tuple[str, np.ndarray, np.ndarray]:
    """Returns (uri, mean_array, median_array) — both zonal statistics are always computed
    so the analysis layer can select either without a second raster read.

    `check_cancelled` runs before each chunk read and raises to stop the
    extraction, so a passed deadline stops it within one chunk read."""
    mean_results: List[float] = []
    median_results: List[float] = []
    try:
        with rasterio.Env(**RASTERIO_ENV_KWARGS):
            with rasterio.open(file_uri) as src:
                for band_chunk in generate_band_chunks(band_indices, chunk_size):
                    if check_cancelled is not None:
                        check_cancelled()
                    data = src.read(band_chunk, window=window).astype(np.float32)

                    if src.nodata is not None:
                        data[data == src.nodata] = np.nan

                    data[:, ~precomputed_mask] = np.nan

                    # A timestep with no valid cell in the area (for example,
                    # a year the dataset has no data for) yields NaN, which the
                    # response carries as null. numpy warns about each one.
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        # Sum in float64: float32 sums over many cells lose
                        # about 7 digits, which z-scores then amplify.
                        mean_results.extend(
                            np.nanmean(data, axis=(1, 2), dtype=np.float64)
                        )
                        median_results.extend(np.nanmedian(data, axis=(1, 2)))

        return (file_uri, np.array(mean_results), np.array(median_results))
    except Exception as e:
        logger.error(f"Failed to process file {file_uri}: {e}")
        raise


def apply_zscore_transform(
    base_series: pd.Series, transform, default_reference: TimeRange | None = None
) -> pd.Series:
    """Applies a Z-score transform to a time-indexed pd.Series.

    For ZScoreFixedInterval, the reference slice is obtained by label-based
    indexing on the series. Without a time_range of its own, the reference is
    `default_reference` (the requested range), or the whole series without one.
    """
    if transform is None or isinstance(transform, NoTransform):
        return base_series

    if isinstance(transform, ZScoreMovingInterval):
        # Compare each observation with the preceding window. The current
        # observation must not influence its own reference distribution.
        roll = base_series.shift(1).rolling(transform.width)
        rolling_std = roll.std(ddof=0).replace(0, np.nan)
        return (base_series - roll.mean()) / rolling_std

    if isinstance(transform, ZScoreFixedInterval):
        reference = transform.time_range or default_reference
        if reference is None:
            ref = base_series
        else:
            ref = base_series.loc[reference.gte : reference.lte]
            if len(ref) == 0:
                raise ValueError(
                    f"Reference range [{reference.gte}, {reference.lte}] "
                    f"has no overlap with the extracted series "
                    f"[{base_series.index[0]}, {base_series.index[-1]}]. "
                    "Ensure the reference interval falls within the extraction time range."
                )
        std = ref.std(ddof=0)
        if np.isnan(std) or std == 0:
            # A reference period without values (the empty years, say) or
            # without spread gives nothing to divide by, so no timestep has a
            # z-score, as in a moving window.
            return pd.Series(np.nan, index=base_series.index)
        return (base_series - ref.mean()) / std

    return base_series


def apply_temporal_transform(timeseries_data: pd.Series, smoother_config) -> pd.Series:
    if isinstance(smoother_config, MovingAverageSmoother):
        center = smoother_config.method == "centered"
        # A timestep has a smoothed value only when every timestep in its
        # window has a value: the axis ends and the empty years widen by the
        # window instead of averaging fewer values.
        return timeseries_data.rolling(
            window=smoother_config.width,
            center=center,
            min_periods=smoother_config.width,
        ).mean()
    return timeseries_data


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def summarize_series(name: str, series: pd.Series) -> SummaryStat:
    """Summary statistics over the timesteps that have a value.

    Timesteps without a value are skipped. The standard deviation is the
    population one, as in the z-scores. A statistic that can't be computed (no
    values at all) is None, never NaN, which is not valid JSON.
    """
    clean = series.dropna()
    return SummaryStat(
        name=name,
        mean=_finite_or_none(clean.mean()),
        median=_finite_or_none(clean.median()),
        stdev=_finite_or_none(clean.std(ddof=0)),
    )


async def execute_timeseries_job(
    request: TimeseriesRequest,
    file_mapping: Dict[str, List[int]],
    timestep_list: List[str],
    dataset_crs: str,
    dataset_transform_array: List[float],
    max_concurrency: int = 10,
) -> Tuple[TimeseriesResponse, dict]:
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
    logger.info(
        f"RAM Governor set chunk size to {chunk_size}. Area: {total_area} sqm. Cells: {n_cells}"
    )

    limiter = anyio.CapacityLimiter(max_concurrency)
    results: List[Tuple[str, np.ndarray]] = []

    async def _worker_wrapper(uri: str, bands: List[int]):
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
        logger.info(
            "Extraction %s after %.1f s: %d cells x %d timesteps, chunk %d.",
            outcome,
            time.monotonic() - started,
            n_cells,
            len(timestep_list),
            chunk_size,
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


def execute_analyze_request(
    payload: TimeseriesAnalyzeRequest | TimeseriesRequest,
    base_series_payload: dict,
    extraction_metadata: dict,
) -> TimeseriesResponse:
    """Applies transform and smoothing to a stored base series without any raster I/O.

    Called synchronously by POST /v3/timeseries/analyze, and by the extraction
    job for its own response, so both answer a request the same way.
    """
    stat_key = payload.zonal_statistic.value  # "mean" or "median"
    # dtype=float turns the stored nulls (timesteps without a value) into NaN,
    # even when every stored value is null.
    full_base_series = pd.Series(
        base_series_payload[stat_key],
        index=base_series_payload["timesteps"],
        dtype=float,
    )

    if payload.time_range:
        response_index = full_base_series.loc[
            payload.time_range.gte : payload.time_range.lte
        ].index
        if response_index.empty:
            raise ValueError(
                f"Requested time_range [{payload.time_range.gte}, {payload.time_range.lte}] "
                f"has no overlap with the extraction range "
                f"[{base_series_payload['timesteps'][0]}, {base_series_payload['timesteps'][-1]}]."
            )
    else:
        response_index = full_base_series.index

    # Transform and smooth the FULL extraction series, and only then slice to the
    # requested range, so a timestep's value never depends on the range asked for:
    # - ZScoreFixedInterval can reference any stored timestep as a reference period
    # - ZScoreMovingInterval and smoothing windows reach the timesteps before
    #   (and, for centered smoothing, after) the response range
    transformed_full = apply_zscore_transform(
        full_base_series, payload.transform, default_reference=payload.time_range
    )

    output_series_list = []
    summary_stats = []
    for option in payload.requested_series_options:
        smoothed_full = apply_temporal_transform(transformed_full, option.smoother)
        smoothed = smoothed_full.loc[response_index]
        output_series_list.append(
            Series(
                options=option,
                time_range=TimeRange(gte=response_index[0], lte=response_index[-1]),
                timesteps=list(smoothed.index),
                values=smoothed.replace({np.nan: None}).to_list(),
            )
        )
        summary_stats.append(summarize_series(option.name, smoothed))

    return TimeseriesResponse(
        dataset_id=extraction_metadata["dataset_id"],
        variable_id=extraction_metadata["variable_id"],
        area=extraction_metadata["area"],
        n_cells=extraction_metadata["n_cells"],
        summary_stats=summary_stats,
        series=output_series_list,
        transform=payload.transform,
        zonal_statistic=payload.zonal_statistic,
    )
