"""Analysis of a stored base series: z-scores, smoothing and summary statistics.

No raster reads: /timeseries/analyze and the extraction job both answer a request
from a base series here.
"""

import numpy as np
import pandas as pd

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
        n_values = int(ref.notna().sum())
        if n_values < 2:
            # The empty years, say. The client shows this to the user.
            raise ValueError(
                f"The z-score's reference period [{ref.index[0]}, {ref.index[-1]}] "
                f"has {n_values} time step{'' if n_values == 1 else 's'} with "
                "values; it needs at least 2. Choose other years."
            )
        std = ref.std(ddof=0)
        if std == 0:
            # No spread to divide by, so no timestep has a z-score, as in a
            # moving window.
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


def execute_analyze_request(
    payload: TimeseriesAnalyzeRequest | TimeseriesRequest,
    base_series_payload: dict,
    extraction_metadata: dict,
) -> TimeseriesResponse:
    """Applies transform and smoothing to a stored base series without any raster I/O.

    Called synchronously by POST /timeseries/analyze, and by the extraction
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
