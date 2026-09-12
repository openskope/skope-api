import os
import tempfile
from osgeo import gdal

from . import fs_utils


def build_cog_slice(input_path, cog_file_path, start_band, end_band, trunc=False):
    """Selects a band range from input_path, writes a temp GeoTiff, then converts it to a COG.

    Args:
        input_path:    Path to the source multi-band GeoTiff.
        cog_file_path: Destination path for the output COG file.
        start_band:    First band to extract (1-indexed, inclusive).
        end_band:      Last band to extract (1-indexed, inclusive).
        trunc:         If True, cast values to UInt16 and set nodata=65535.
    """
    selected_bands = ",".join(str(b) for b in range(start_band, end_band + 1))
    trunc_step = "set-type --datatype=UInt16 ! edit --nodata 65535 ! " if trunc else ""

    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        slice_file = tmp.name

    try:
        print("Selecting bands and writing GeoTiff file")
        pipe_str = (
            f"read {fs_utils.to_vsi(input_path)} ! select --band={selected_bands} ! {trunc_step}"
            f"write {slice_file} --format GTiff --co COMPRESS=ZSTD --co TILED=YES "
            f"--co BLOCKXSIZE=128 --co BLOCKYSIZE=128 --overwrite"
        )
        gdal.Run(
            "raster", "pipeline",
            pipeline=pipe_str,
            progress=lambda p, o, d: print(f"{p*100:.0f}% ", end="", flush=True),
        )

        print("\nConverting file to COG")
        gdal.Run(
            "raster", "convert",
            input=slice_file,
            output=fs_utils.to_vsi(cog_file_path),
            output_format="COG",
            creation_option=[
                "BLOCKSIZE=128",
                "COMPRESS=ZSTD",
                "PREDICTOR=2",
                "OVERVIEWS=IGNORE_EXISTING",
                "INTERLEAVE=TILE",
                "SPARSE_OK=TRUE",
            ],
            overwrite=True,
        )
    finally:
        if os.path.exists(slice_file):
            os.unlink(slice_file)
