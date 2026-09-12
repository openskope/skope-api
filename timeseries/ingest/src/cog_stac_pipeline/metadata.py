import os
import yaml
from dateutil.relativedelta import relativedelta

from .datetime_utils import get_iso_key, singular_to_plural_for_relativedelta


def load_and_verify_metadata(metadata_file_path, dataset_name):
    """Loads YAML metadata and verifies the dataset exists."""
    print(f"Loading metadata from {metadata_file_path}")
    if not os.path.exists(metadata_file_path):
        raise ValueError(f"Metadata file not found at: {metadata_file_path}")

    with open(metadata_file_path, "r") as f:
        yaml_content = yaml.safe_load(f) or []

    meta_by_id = {item.get("id"): item for item in yaml_content if isinstance(item, dict)} if isinstance(yaml_content, list) else {}
    ds_meta = meta_by_id.get(dataset_name)

    if ds_meta is None:
        raise ValueError(f"Dataset '{dataset_name}' not found in metadata file. Cannot proceed.")

    return yaml_content, ds_meta


def validate_else_add_timespan(ds_meta, start_dt, time_delta):
    """Validates user-provided time variables against metadata, adding to metadata if missing."""
    updated = False

    timespan = ds_meta.setdefault("timespan", {})
    period = timespan.setdefault("period", {})
    resolution = timespan.setdefault("resolution", {})

    # 1. Validate Start Time (gte)
    expected_gte = get_iso_key(start_dt, time_delta)
    meta_gte = period.get("gte")

    if meta_gte is None:
        print(f"Metadata missing start time (gte). Adding user-provided start time: {expected_gte}")
        period["gte"] = expected_gte
        updated = True
    elif str(meta_gte) != expected_gte:
        raise ValueError(f"Start time mismatch! User: {expected_gte}, Meta: {meta_gte}")

    # 2. Validate Time Delta (resolution)
    if not resolution:
        print(f"Metadata missing resolution. Adding user-provided value.")
        for k, v in time_delta.items():
            resolution[k] = v
        updated = True
    else:
        user_delta = relativedelta(**time_delta)
        plural_meta_res = singular_to_plural_for_relativedelta(resolution)
        meta_delta = relativedelta(**plural_meta_res)

        if user_delta != meta_delta:
            raise ValueError(f"Time delta mismatch! User: {time_delta}, Meta: {resolution}")

    return updated


def validate_else_add_temporal_end(
    ds_meta, dataset_name, band_counts, start_dt, time_delta
):
    """Validate each variable's band-derived endpoint against metadata."""
    period = ds_meta.setdefault("timespan", {}).setdefault("period", {})
    declared_end = period.get("lte")
    step = relativedelta(**time_delta)
    coverage = {
        variable_id: (
            band_count,
            get_iso_key(start_dt + step * (band_count - 1), time_delta),
        )
        for variable_id, band_count in band_counts.items()
    }

    derived_ends = {derived_end for _, derived_end in coverage.values()}
    if len(derived_ends) != 1:
        details = "; ".join(
            _temporal_coverage_detail(
                dataset_name, variable_id, band_count, derived_end, declared_end
            )
            for variable_id, (band_count, derived_end) in coverage.items()
        )
        raise ValueError(f"Inconsistent temporal coverage: {details}")

    derived_end = next(iter(derived_ends))
    if declared_end is None:
        print(f"Metadata missing end time (lte). Adding derived end time: {derived_end}")
        period["lte"] = derived_end
        return True

    mismatches = [
        _temporal_coverage_detail(
            dataset_name, variable_id, band_count, variable_end, declared_end
        )
        for variable_id, (band_count, variable_end) in coverage.items()
        if variable_end != str(declared_end)
    ]
    if mismatches:
        raise ValueError("Temporal endpoint mismatch: " + "; ".join(mismatches))

    return False


def _temporal_coverage_detail(
    dataset_name, variable_id, band_count, derived_end, declared_end
):
    declared = "<missing>" if declared_end is None else str(declared_end)
    return (
        f"dataset '{dataset_name}', variable '{variable_id}', "
        f"band count {band_count}, derived endpoint '{derived_end}', "
        f"declared endpoint '{declared}'"
    )


def validate_else_add_extracted_info(ds_meta, var_name, c_extra):
    """Validates actual data extracted from the COG against metadata, adding if missing."""
    updated = False

    # 1. CRS
    m_crs = ds_meta.get("crs")
    extracted_crs = f"EPSG:{c_extra.get('proj:epsg')}" if c_extra.get("proj:epsg") else None
    if m_crs is None and extracted_crs:
        print("Metadata missing CRS. Adding extracted CRS.")
        ds_meta["crs"] = extracted_crs
        updated = True
    elif m_crs and extracted_crs and m_crs.upper() != extracted_crs.upper():
        raise ValueError(f"CRS mismatch! Meta: {m_crs}, Data: {extracted_crs}")

    # 2. Transform (Compare first 6 elements rounded to 5 decimals)
    m_trans = ds_meta.get("transform")
    c_trans = c_extra.get("proj:transform")
    if m_trans is None and c_trans:
        print("Metadata missing Transform. Adding extracted Transform.")
        ds_meta["transform"] = c_trans
        updated = True
    elif m_trans and c_trans:
        if not all(round(m, 5) == round(d, 5) for m, d in zip(m_trans[:6], c_trans[:6])):
            raise ValueError(f"Transform mismatch!\nMeta: {m_trans[:6]}\nData: {c_trans[:6]}")

    # 3. Min/Max
    ds_meta.setdefault("variables", [])
    var_meta = next((v for v in ds_meta["variables"] if v.get("id") == var_name), None)

    if var_meta is None:
        raise ValueError(f"Metadata missing variable entry for '{var_name}'.")

    c_min, c_max = c_extra.get("titiler:min"), c_extra.get("titiler:max")

    if var_meta.get("min") is None and c_min is not None:
        print(f"Metadata missing min for {var_name}. Adding extracted min.")
        var_meta["min"] = float(c_min)
        updated = True
    elif var_meta.get("min") is not None and c_min is not None and abs(var_meta["min"] - c_min) > 0.001:
        raise ValueError(f"Min mismatch for {var_name}! Meta: {var_meta['min']}, Data: {c_min}")

    if var_meta.get("max") is None and c_max is not None:
        print(f"Metadata missing max for {var_name}. Adding extracted max.")
        var_meta["max"] = float(c_max)
        updated = True
    elif var_meta.get("max") is not None and c_max is not None and abs(var_meta["max"] - c_max) > 0.001:
        raise ValueError(f"Max mismatch for {var_name}! Meta: {var_meta['max']}, Data: {c_max}")

    return updated
