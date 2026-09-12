import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Timestep normalization

_CANONICAL_EXT = "-01-01T00:00:00Z"
_PRECISION_LEN = {"year": 4, "month": 7, "day": 10, "datetime": 20}
_LEN_TO_PRECISION = {v: k for k, v in _PRECISION_LEN.items()}


def _detect_precision(key: str) -> str:
    prec = _LEN_TO_PRECISION.get(len(key))
    if prec is None:
        raise ValueError(
            f"Timestep '{key}' has an unrecognized ISO-8601 length ({len(key)}). "
            "Expected 'YYYY', 'YYYY-MM', 'YYYY-MM-DD', or 'YYYY-MM-DDTHH:MM:SSZ'."
        )
    return prec


def _normalize_timestep(timestep: str, var_lookup: dict) -> str:
    """Normalizes an incoming timestep to the precision level used by the lookup dict.

    - Same precision: returned unchanged
    - Finer with canonical suffix: truncated ("0103-01-01" → "0103" for yearly dict)
    - Finer with non-canonical suffix: raises ValueError ("0103-08-12" for yearly dict)
    - Coarser than dict: raises ValueError ("0103" for monthly dict)
    """
    dict_precision = _detect_precision(next(iter(var_lookup)))
    ts_prec = _detect_precision(timestep)
    if ts_prec == dict_precision:
        return timestep

    ts_len = _PRECISION_LEN[ts_prec]
    dict_len = _PRECISION_LEN[dict_precision]

    if ts_len < dict_len:
        raise ValueError(
            f"Timestep '{timestep}' ({ts_prec} precision) is coarser than "
            f"this dataset's {dict_precision} resolution."
        )

    # Finer: truncate only if the discarded suffix matches canonical defaults.
    # _CANONICAL_EXT[dict_len-4 : ts_len-4] isolates exactly the suffix chars
    # that separate the two precision levels (e.g. dict=year→ts=day gives "-01-01").
    truncated = timestep[:dict_len]
    discarded = timestep[dict_len:]
    expected_discard = _CANONICAL_EXT[dict_len - 4 : ts_len - 4]
    if discarded != expected_discard:
        raise ValueError(
            f"Timestep '{timestep}' specifies sub-{dict_precision} precision, but "
            f"this dataset uses {dict_precision} resolution. Use '{truncated}' instead."
        )
    return truncated


# ---------------------------------------------------------------------------
# Resolvers


def resolve_temporal_slice(
    lookup_data: dict,
    variable_id: str,
    start_step: str,
    end_step: str,
    base_url: str,
) -> tuple[Dict[str, List[int]], List[str]]:
    """
    Resolves a temporal window into everything needed to execute the extraction:
    a mapping of storage URIs to their band indices, and the ordered list of
    timestep strings (ISO dates) for building a time-indexed pd.Series.
    """
    var_lookup = lookup_data.get(variable_id)
    if not var_lookup:
        raise ValueError(f"Variable '{variable_id}' not found in lookup dictionary.")

    norm_start = _normalize_timestep(start_step, var_lookup)
    norm_end = _normalize_timestep(end_step, var_lookup)

    file_mapping: Dict[str, List[int]] = {}
    timestep_list: List[str] = []

    for step_str, entry in var_lookup.items():
        if step_str > norm_end:
            break

        if step_str >= norm_start:
            suff_uri = entry["file"]
            uri = f"{base_url.rstrip('/')}/{suff_uri}"
            band = entry["bidx"]

            if uri not in file_mapping:
                file_mapping[uri] = []
            file_mapping[uri].append(band)
            timestep_list.append(step_str)

    # Protect against internal data gaps where the slice yields no results
    if not file_mapping:
        raise ValueError("No data available within the specific requested time slice.")

    for uri in file_mapping:
        file_mapping[uri].sort()

    return file_mapping, timestep_list


def resolve_uri_single_band(
    lookup_data: dict, variable_id: str, timestep: str, base_url: str
) -> tuple[str, int]:
    """
    Convenience function for single-timestep requests (tile endpoint) that need
    to resolve to a single file and band index.
    """
    var_lookup = lookup_data.get(variable_id, {})
    if not var_lookup:
        raise ValueError(f"Variable '{variable_id}' not found in lookup dictionary.")

    norm_timestep = _normalize_timestep(timestep, var_lookup)

    if norm_timestep not in var_lookup:
        logger.error(
            f"Timestep {norm_timestep} is missing in lookup for '{variable_id}'."
        )
        raise ValueError(
            f"Timestep '{norm_timestep}' is not available for '{variable_id}' in this dataset."
        )

    entry = var_lookup[norm_timestep]
    return f"{base_url.rstrip('/')}/{entry['file']}", entry["bidx"]
