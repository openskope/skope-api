import asyncio
import yaml
import os
import json
import logging
import re
from pathlib import Path

import httpx

from app.store.data_reader import DataReader

logger = logging.getLogger(__name__)

_CACHE_DIR = "/tmp/skope_dicts"
os.makedirs(_CACHE_DIR, exist_ok=True)

# Strict ISO-8601 zero-padded pattern (YYYY-MM-DDTHH:MM:SSZ)
ISO_TIME_PATTERN = re.compile(
    r"^\d{4}(?:-\d{2}(?:-\d{2}(?:T\d{2}:\d{2}:\d{2}(?:Z|[+-]\d{2}:\d{2})?)?)?)?$"
)

# ------------------------------------------------------------------
# Registry (metadata.yml)


def load_registry(filepath: Path) -> dict:
    """Loads the metadata registry from the local filesystem.
    Raises on any missing file, parse error, or incomplete dataset entry
    so that misconfiguration aborts startup immediately.
    """
    if not filepath.is_file():
        raise FileNotFoundError(f"Registry file not found at {filepath}")

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            registry_list = yaml.safe_load(f)
    except Exception as e:
        raise ValueError(f"Failed to parse registry YAML: {e}") from e

    registry_dict = {}
    for ds in registry_list:
        ds_id = ds.get("id")
        if not ds_id:
            raise ValueError("Registry contains a dataset entry without an 'id'.")
        if "crs" not in ds:
            raise ValueError(f"Dataset '{ds_id}' is missing required 'crs'.")
        if "transform" not in ds or len(ds.get("transform", [])) not in [6, 9]:
            raise ValueError(
                f"Dataset '{ds_id}' is missing a valid 6- or 9-element 'transform' array."
            )
        registry_dict[ds_id] = ds

    logger.info(f"Successfully loaded Global Registry from {filepath}.")
    return registry_dict


# ------------------------------------------------------------------
# Colormaps


async def resolve_colormaps(
    registry_dict: dict,
    colormaps_path: Path,
    client: httpx.AsyncClient,
    tile_server_url: str,
) -> None:
    """Resolves colormap stops for all colormaps referenced in registry variables.

    Custom colormaps are read from colormaps_path. Unknown names are fetched from
    TiTiler's /colorMaps/{name} endpoint. Stops are injected into each variable dict
    in-place. Raises at startup if any colormap name cannot be resolved.
    """
    if colormaps_path.exists():
        colormaps: dict[str, list[str]] = json.loads(colormaps_path.read_text())
    else:
        logger.warning(
            f"Custom colormaps file not found at {colormaps_path.resolve()}. "
            "All colormap names will be fetched from TiTiler."
        )
        colormaps = {}

    # Every variable advertises an effective colormap. This keeps metadata,
    # frontend colorbars, and tile rendering on the same contract even when a
    # registry entry does not choose a dataset-specific palette.
    for ds in registry_dict.values():
        for var in ds.get("variables", []):
            if not var.get("colormap"):
                var["colormap"] = "viridis"

    names_needed = {
        var.get("colormap")
        for ds in registry_dict.values()
        for var in ds.get("variables", [])
        if var.get("colormap")
    }

    names_from_titiler = names_needed - set(colormaps.keys())
    for name in names_from_titiler:
        colormaps[name] = await _fetch_colormap_from_titiler(
            client, tile_server_url, name
        )

    for ds in registry_dict.values():
        for var in ds.get("variables", []):
            cm = var.get("colormap")
            if cm and cm not in colormaps:
                raise ValueError(
                    f"Colormap '{cm}' (used by variable '{var.get('id')}') "
                    f"is not in {colormaps_path.resolve()} and was not found in TiTiler."
                )
            if cm:
                var["colormap_stops"] = colormaps[cm]

    logger.info(
        f"Resolved {len(names_needed)} colormap(s) {sorted(names_needed)}: "
        f"{len(names_needed) - len(names_from_titiler)} from {colormaps_path.name}, "
        f"{len(names_from_titiler)} via TiTiler"
    )


async def _fetch_colormap_from_titiler(
    client: httpx.AsyncClient, tile_server_url: str, name: str
) -> list[str]:
    """Fetches a named colormap from TiTiler with retry on connection errors."""
    url = f"{tile_server_url}/colorMaps/{name}"
    for attempt in range(1, 4):
        try:
            resp = await client.get(url)
            resp.raise_for_status()
            rgba_dict: dict[str, list[int]] = resp.json()
            return [
                "#{:02x}{:02x}{:02x}".format(*rgba_dict[str(i)][:3]) for i in range(256)
            ]
        except httpx.ConnectError:
            if attempt == 3:
                raise
            wait = 2**attempt  # 2 s, 4 s
            logger.warning(
                f"TiTiler unreachable (attempt {attempt}/3), retrying in {wait}s…"
            )
            await asyncio.sleep(wait)
    raise RuntimeError("unreachable")


# ------------------------------------------------------------------
# Lookup Dictionary


def _get_cached_lookup(dataset_id: str) -> dict | None:
    """Reads the dataset lookup from the local worker's shared /tmp disk."""
    file_path = os.path.join(_CACHE_DIR, f"{dataset_id}_lookup.json")
    if os.path.exists(file_path):
        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.warning(
                f"Corrupted cache file found for {dataset_id}. Forcing re-fetch."
            )
            return None
    return None


def _set_cached_lookup(dataset_id: str, data: dict) -> None:
    """Safely writes the lookup to disk using an atomic rename operation."""
    file_path = os.path.join(_CACHE_DIR, f"{dataset_id}_lookup.json")
    temp_path = f"{file_path}.tmp"
    with open(temp_path, "w") as f:
        json.dump(data, f)
    os.rename(temp_path, file_path)


async def fetch_lookup_dict(
    dataset_id: str, storage_base_url: str, data_reader: DataReader
) -> dict:
    """
    Fetches the temporal lookup dictionary. Checks the shared local disk cache first.
    If missing, fetches from origin via the provided DataReader, validates, and caches it.
    """
    cached_data = _get_cached_lookup(dataset_id)
    if cached_data:
        return cached_data

    lookup_uri = f"{storage_base_url.rstrip('/')}/{dataset_id}/lookup.json"

    try:
        lookup_data = await data_reader.read_json(lookup_uri)
    except Exception as e:
        logger.error(f"Failed to fetch lookup dict for '{dataset_id}': {e}")
        raise ValueError(f"Failed to retrieve lookup dictionary from origin: {e}")

    # Schema Validation
    try:
        first_var = next(iter(lookup_data.values()), {})
        first_timestep = next(iter(first_var.values()), {}) if first_var else {}
        if "file" not in first_timestep or "bidx" not in first_timestep:
            raise KeyError()
    except (StopIteration, AttributeError, KeyError):
        logger.critical(f"Lookup schema invalid or empty for '{dataset_id}'.")
        raise ValueError("Data origin schema validation failed or data is empty.")

    # Keys' Time Format Validation and Order Verification
    for var_id, var_data in lookup_data.items():
        previous_time = None
        for time_key in var_data.keys():
            if not ISO_TIME_PATTERN.match(time_key):
                logger.critical(f"Invalid time key '{time_key}' in '{dataset_id}'.")
                raise ValueError(f"Upstream time format violation: {time_key}")

            if previous_time is not None and time_key <= previous_time:
                logger.critical(
                    f"Data prep error for '{dataset_id}': '{time_key}' is out of order (came after '{previous_time}')."
                )
                raise ValueError(
                    "Upstream data prep error: Temporal keys are not sorted chronologically."
                )

            previous_time = time_key

    _set_cached_lookup(dataset_id, lookup_data)
    logger.info(
        f"Successfully fetched, validated, and cached lookup dict to disk for '{dataset_id}'."
    )

    return lookup_data
