"""Prepare owda_v1's `pdsi` source from NOAA's Old World Drought Atlas NetCDF (META-013).

Fetch runs it as:

    prepare_owda_v1.py pdsi <output.tif> <owda.nc>

It changes how the values are stored, never the values:
- year 0, the file's first timestep, is left out, so band i holds year i CE,
  described by its four-digit year (OBS-006);
- the NetCDF's (lon, lat, time) array is stored one band per year, with rows
  running north to south; the grid edges are the cell centres +/- 0.25 degree;
- the CRS is WGS 84 (EPSG:4326), which the NetCDF does not declare;
- NaN becomes nodata -9999;
- values stay float64, as published: they do not fit float32 exactly.

Before exiting, it reads the output back and checks every value against the
NetCDF.
"""

import sys

import numpy as np
from netCDF4 import Dataset
from osgeo import gdal, osr

VARIABLE = "pdsi"
NODATA = -9999.0
FIRST_YEAR = 1
STEP = 0.5


def read(source):
    nc = Dataset(source)
    nc.set_auto_mask(False)
    time, lat, lon = nc["time"][:], nc["lat"][:], nc["lon"][:]
    pdsi = nc["pdsi"]
    if pdsi.dimensions != ("lon", "lat", "time"):
        sys.exit(f"expected pdsi(lon, lat, time), found {pdsi.dimensions}")
    if list(time) != list(range(len(time))):
        sys.exit("expected consecutive years from 0")
    for name, axis in (("lat", lat), ("lon", lon)):
        if not np.allclose(np.diff(axis), STEP):
            sys.exit(f"{name} is not a {STEP}-degree grid")
    cube = np.transpose(pdsi[:], (2, 1, 0))  # (time, lat, lon), lat south to north
    years = list(range(FIRST_YEAR, int(time[-1]) + 1))
    return cube[FIRST_YEAR:, ::-1, :], years, float(lon[0]) - STEP / 2, float(lat[-1]) + STEP / 2


def write(output, cube, years, west, north):
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    _, height, width = cube.shape
    out = gdal.GetDriverByName("GTiff").Create(
        output, width, height, len(years), gdal.GDT_Float64,
        options=["TILED=YES", "BLOCKXSIZE=128", "BLOCKYSIZE=128", "INTERLEAVE=BAND",
                 "COMPRESS=DEFLATE", "PREDICTOR=3", "BIGTIFF=NO"],
    )
    out.SetGeoTransform((west, STEP, 0.0, north, 0.0, -STEP))
    out.SetSpatialRef(srs)
    out.SetMetadataItem("AREA_OR_POINT", "Area")
    for i, year in enumerate(years, start=1):
        band = out.GetRasterBand(i)
        band.SetNoDataValue(NODATA)
        band.SetDescription(f"{year:04d}")
        band.WriteArray(np.where(np.isfinite(cube[i - 1]), cube[i - 1], NODATA))
    out = None


def check(output, cube, years):
    """Every valid value is the NetCDF's, bit for bit, and every other cell is nodata."""
    ds = gdal.Open(output)
    if ds.RasterCount != len(years):
        sys.exit(f"output has {ds.RasterCount} bands, not {len(years)}")
    for i, year in enumerate(years, start=1):
        band = ds.GetRasterBand(i)
        values, expected = band.ReadAsArray(), cube[i - 1]
        valid = np.isfinite(expected)
        if band.GetDescription() != f"{year:04d}" or band.GetNoDataValue() != NODATA:
            sys.exit(f"band {i}: wrong description or nodata")
        if not np.array_equal(values[valid].view(np.uint64), expected[valid].view(np.uint64)):
            sys.exit(f"band {i} ({year:04d}): a valid value differs from the NetCDF")
        if not np.all(values[~valid] == NODATA):
            sys.exit(f"band {i} ({year:04d}): a missing cell is not nodata")


def main(variable, output, source):
    if variable != VARIABLE:
        sys.exit(f"owda_v1 has no variable {variable!r}")
    gdal.UseExceptions()
    cube, years, west, north = read(source)
    if np.any(cube[np.isfinite(cube)] == NODATA):
        sys.exit(f"a valid value equals the nodata value {NODATA}")
    write(output, cube, years, west, north)
    check(output, cube, years)


if __name__ == "__main__":
    main(*sys.argv[1:])
