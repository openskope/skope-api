import os
import json
import yaml
import pystac
from osgeo import gdal

from .config import PipelineConfig
from . import fs_utils, metadata, stac_builder
from .datetime_utils import singular_to_plural_for_relativedelta
from .manifest import (
    ManifestVariable,
    load_input_manifest,
    validate_manifest_metadata,
)


def run_pipeline(config: PipelineConfig) -> None:
    configure_gdal()
    dataset_time_delta = singular_to_plural_for_relativedelta(config.dataset_time_delta)

    yaml_content, ds_meta = metadata.load_and_verify_metadata(config.metadata_file_path, config.dataset_name)
    any_metadata_updated = metadata.validate_else_add_timespan(
        ds_meta, config.dataset_start_datetime, dataset_time_delta
    )

    input_variables = resolve_input_variables(config)
    if config.input_manifest_path:
        validate_manifest_metadata(input_variables, ds_meta)

    band_counts = read_input_band_counts(input_variables, config.dataset_name)
    any_metadata_updated = metadata.validate_else_add_temporal_end(
        ds_meta,
        config.dataset_name,
        band_counts,
        config.dataset_start_datetime,
        dataset_time_delta,
    ) or any_metadata_updated

    if config.preflight_only:
        print(f"Preflight passed for dataset: {config.dataset_name}")
        return

    output_dir = config.resolved_output_dir
    root_cogs_dir = os.path.join(output_dir, "cogs")
    stac_dir = os.path.join(output_dir, "stac")
    fs_utils.makedirs(root_cogs_dir)
    fs_utils.makedirs(stac_dir)

    catalog = pystac.Catalog(
        id="skope-catalog",
        description=ds_meta.get("description", f"STAC Catalog for {config.dataset_name} dataset."),
    )
    lookup_dict = {}

    for input_variable in input_variables:
        input_path = input_variable.uri
        var_name = input_variable.id
        print(f"\nProcessing variable: {var_name}")

        cogs_var_dir = os.path.join(root_cogs_dir, var_name)
        partial_path_base = os.path.join(config.dataset_name, "cogs", var_name)
        fs_utils.makedirs(cogs_var_dir)

        collection = stac_builder.build_collection(var_name, config.dataset_start_datetime)

        stac_builder.process_variable(
            paths=stac_builder.VariablePaths(
                input_path=input_path,
                cogs_var_dir=cogs_var_dir,
                partial_path_base=partial_path_base,
                var_name=var_name,
            ),
            stac_collection=collection,
            lookup_dict=lookup_dict,
            window=config.max_bands_per_slice,
            trunc=config.trunc_to_uint16,
            start_dt=config.dataset_start_datetime,
            time_delta=dataset_time_delta,
        )

        updated_extracted = metadata.validate_else_add_extracted_info(ds_meta, var_name, collection.extra_fields)
        any_metadata_updated = any_metadata_updated or updated_extracted

        collection.update_extent_from_items()
        catalog.add_child(collection)

    print("\nSaving STAC Catalog")
    stac_builder.save_catalog(catalog, stac_dir)

    print("Saving lookup dictionary")
    lookup_file_path = os.path.join(output_dir, "lookup.json")
    fs_utils.write_text(lookup_file_path, json.dumps(lookup_dict, indent=2))

    if any_metadata_updated:
        print(f"Saving updated metadata back to {config.metadata_file_path}")
        with open(config.metadata_file_path, "w") as f:
            yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)

    print("\nDone!")
    print(f"  STAC catalog: {stac_dir}")
    print(f"  COG slices:   {root_cogs_dir}")
    print(f"  Lookup dict:  {lookup_file_path}")


def resolve_input_variables(config: PipelineConfig) -> list[ManifestVariable]:
    if config.input_manifest_path:
        return load_input_manifest(config.input_manifest_path, config.dataset_name)

    return [
        ManifestVariable(id=os.path.basename(path).split(".")[0], uri=path)
        for path in fs_utils.list_tif_files(config.input_dir)
        if not path.endswith("_cogd.tif")
    ]


def read_input_band_counts(
    input_variables: list[ManifestVariable], dataset_name: str
) -> dict[str, int]:
    """Read raster band counts before the pipeline creates any output."""
    band_counts = {}
    for variable in input_variables:
        try:
            with gdal.Open(fs_utils.to_vsi(variable.uri)) as dataset:
                if dataset is None:
                    raise RuntimeError("GDAL returned no dataset")
                band_count = dataset.RasterCount
        except Exception as exc:
            raise ValueError(
                f"Cannot inspect raster for dataset '{dataset_name}', variable "
                f"'{variable.id}': {variable.uri}: {exc}"
            ) from exc

        if band_count < 1:
            raise ValueError(
                f"Raster for dataset '{dataset_name}', variable '{variable.id}' "
                f"has invalid band count {band_count}: {variable.uri}"
            )
        band_counts[variable.id] = band_count

    return band_counts


def configure_gdal() -> None:
    gdal.UseExceptions()
    gdal.SetCacheMax(4608 * 1024 * 1024)  # 4.5 GB
    gdal.SetConfigOption("GDAL_NUM_THREADS", "ALL_CPUS")


def main() -> None:
    run_pipeline(PipelineConfig.from_env())


if __name__ == "__main__":
    main()
