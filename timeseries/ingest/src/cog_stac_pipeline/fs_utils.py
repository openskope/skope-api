import os
from osgeo import gdal


def is_s3(path: str) -> bool:
    return str(path).startswith("s3://")


def to_vsi(path: str) -> str:
    """Converts s3://bucket/key to /vsis3/bucket/key for raw GDAL calls.

    rasterio-based calls (e.g. create_stac_item) handle s3:// natively and
    do not need this conversion. Only explicit gdal.* calls require it.
    """
    return path.replace("s3://", "/vsis3/", 1) if is_s3(path) else path


def list_tif_files(directory: str) -> list[str]:
    """Lists .tif file paths in a local directory or an S3 prefix."""
    if is_s3(directory):
        names = gdal.ReadDir(to_vsi(directory)) or []
        base = directory.rstrip("/")
        return [f"{base}/{n}" for n in names if n.endswith(".tif")]
    return [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith(".tif")
    ]


def path_exists(path: str) -> bool:
    """Returns True if the file exists locally or as an S3 object."""
    if is_s3(path):
        return gdal.VSIStatL(to_vsi(path)) is not None
    return os.path.isfile(path)


def makedirs(path: str) -> None:
    """Creates local directories. No-op for S3 (S3 has no real directories)."""
    if not is_s3(path):
        os.makedirs(path, exist_ok=True)


def write_text(path: str, content: str) -> None:
    """Writes a string to a local file or an S3 object."""
    if is_s3(path):
        import boto3
        bucket, key = path[5:].split("/", 1)
        boto3.client("s3").put_object(Bucket=bucket, Key=key, Body=content.encode())
    else:
        with open(path, "w") as f:
            f.write(content)
