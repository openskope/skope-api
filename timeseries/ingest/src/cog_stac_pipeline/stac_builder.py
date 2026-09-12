import os
import pystac
from osgeo import gdal
from dataclasses import dataclass
from dateutil.relativedelta import relativedelta
from rio_stac.stac import create_stac_item

from . import cog_builder, fs_utils
from .datetime_utils import get_iso_key, format_stac_datetime, generate_date_range


@dataclass(frozen=True)
class VariablePaths:
    input_path: str
    cogs_var_dir: str
    partial_path_base: str
    var_name: str


def populate_slice_lookup(lookup_dict, var_name, partial_file_path, start_dt, end_dt, step, time_delta):
    """Fills lookup_dict[var_name][iso_key] = {file, bidx} for every timestep in a slice.

    Band indices (bidx) are 1-based and local to the slice file.
    """
    for i, current_dt in enumerate(generate_date_range(start_dt, end_dt, step)):
        iso_key = get_iso_key(current_dt, time_delta)
        lookup_dict[var_name][iso_key] = {
            "file": partial_file_path,
            "bidx": i + 1,
        }


def create_stac_item_for_slice(cog_file_path, var_name, slice_idx, start_dt, end_dt, step):
    """Creates a pystac.Item for a single COG slice.

    Sets start_datetime / end_datetime properties and resolves the asset href
    relative to where the item JSON will be saved.

    Returns:
        pystac.Item
    """
    item = create_stac_item(
        source=cog_file_path,
        id=f"{var_name}_{slice_idx}",
        asset_name="data",
        asset_roles=["data"],
        asset_media_type=pystac.MediaType.COG,
        with_proj=True,
        with_raster=True,
    )

    item.properties["start_datetime"] = format_stac_datetime(start_dt)
    almost_one_step = step - relativedelta(seconds=1)
    item.properties["end_datetime"] = format_stac_datetime(end_dt + almost_one_step)
    item.datetime = None

    # Resolve relative path from the item JSON location to the COG asset
    cog_filename = os.path.basename(cog_file_path)
    item_dir_path = cog_file_path.replace("cogs", "stac").replace(".tif", "")
    item_json_path = os.path.join(item_dir_path, cog_filename.replace(".tif", ".json"))
    item.assets["data"].href = pystac.utils.make_relative_href(cog_file_path, item_json_path)

    return item


def build_collection(var_name, start_dt):
    """Returns a pystac.Collection with placeholder extents (updated later via update_extent_from_items)."""
    return pystac.Collection(
        id=var_name,
        description=f"Chunked COGs for the {var_name} variable.",
        extent=pystac.Extent(
            spatial=pystac.SpatialExtent([[-180.0, -90.0, 180.0, 90.0]]),
            temporal=pystac.TemporalExtent([[
                start_dt,
                start_dt + relativedelta(years=1),  # dummy, replaced by update_extent_from_items
            ]]),
        ),
    )


class _S3StacIO(pystac.StacIO):
    """pystac StacIO implementation that reads/writes catalog JSON via boto3."""

    def __init__(self):
        import boto3
        self._s3 = boto3.client("s3")

    def write_text_method(self, dest, txt):
        bucket, key = dest[5:].split("/", 1)
        self._s3.put_object(Bucket=bucket, Key=key, Body=txt.encode())

    def read_text_method(self, source):
        if isinstance(source, pystac.Link):
            source = source.get_href()
        if fs_utils.is_s3(source):
            bucket, key = source[5:].split("/", 1)
            return self._s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode()
        return pystac.StacIO.default().read_text_method(source)


def save_catalog(catalog, stac_dir):
    """Normalizes hrefs, makes all asset hrefs relative, and saves the catalog."""
    catalog.normalize_hrefs(stac_dir)
    catalog.make_all_asset_hrefs_relative()
    stac_io = _S3StacIO() if fs_utils.is_s3(stac_dir) else None
    catalog.save(dest_href=stac_dir, catalog_type=pystac.CatalogType.SELF_CONTAINED, stac_io=stac_io)


def process_variable(paths, stac_collection, lookup_dict, window, trunc, start_dt, time_delta):
    """Orchestrates COG slicing, lookup dict population, and STAC item generation for one variable.

    For each band slice:
      1. Computes the date range covered by the slice.
      2. Populates lookup_dict entries for every timestep in the slice.
      3. Creates the COG file on disk if it does not already exist.
      4. Generates a STAC item and attaches it to stac_collection.

    Updates stac_collection.extra_fields with global min/max and projection info.
    """
    step = relativedelta(**time_delta)
    lookup_dict[paths.var_name] = {}

    with gdal.Open(fs_utils.to_vsi(paths.input_path)) as ds:
        tot_bands = ds.RasterCount

    n = -(-tot_bands // window)  # ceiling division
    print(f"Dividing raster into {n} slices (max {window} bands each)")

    global_min = float("inf")
    global_max = float("-inf")

    for s in range(n):
        cog_filename = f"{paths.var_name}_{s + 1}.tif"
        cog_file_path = os.path.join(paths.cogs_var_dir, cog_filename)
        partial_file_path = os.path.join(paths.partial_path_base, cog_filename)

        start_idx = s * window
        end_idx = min((s + 1) * window, tot_bands) - 1
        slice_start_dt = start_dt + step * start_idx
        slice_end_dt = start_dt + step * end_idx

        print(f"\nProcessing slice {s + 1}")
        print(f"Populating lookup dict for {slice_start_dt} to {slice_end_dt}")
        populate_slice_lookup(
            lookup_dict, paths.var_name, partial_file_path,
            slice_start_dt, slice_end_dt, step, time_delta,
        )

        if not fs_utils.path_exists(cog_file_path):
            print(f"Creating slice {s + 1} with bands {start_idx + 1} to {end_idx + 1}")
            print(f"Mapped to time steps: {slice_start_dt} to {slice_end_dt}")
            cog_builder.build_cog_slice(
                paths.input_path, cog_file_path,
                start_band=start_idx + 1,
                end_band=end_idx + 1,
                trunc=trunc,
            )

        print(f"Generating STAC Item for {cog_filename}")
        item = create_stac_item_for_slice(
            cog_file_path, paths.var_name, s + 1,
            slice_start_dt, slice_end_dt, step,
        )
        stac_collection.add_item(item)
        print("STAC Item attached to collection.")

        for b in item.assets["data"].extra_fields["raster:bands"]:
            global_min = min(global_min, b["statistics"]["minimum"])
            global_max = max(global_max, b["statistics"]["maximum"])

    # Propagate proj/nodata/range info from the last item (consistent across all slices)
    nodata_val = item.assets["data"].extra_fields["raster:bands"][0]["nodata"]
    proj_props = item.properties
    stac_collection.extra_fields.update({
        "titiler:nodata": nodata_val,
        "titiler:min": global_min,
        "titiler:max": global_max,
        "proj:epsg": proj_props["proj:epsg"],
        "proj:geometry": proj_props["proj:geometry"],
        "proj:shape": proj_props["proj:shape"],
        "proj:transform": proj_props["proj:transform"],
    })
