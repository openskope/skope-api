import json
import anyio
import aioboto3
from abc import ABC, abstractmethod
from urllib.parse import urlparse


class DataReader(ABC):
    @abstractmethod
    async def read_json(self, uri: str) -> dict: ...


class LocalDataReader(DataReader):
    async def read_json(self, uri: str) -> dict:
        def _read():
            with open(uri) as f:
                return json.load(f)

        return await anyio.to_thread.run_sync(_read)


class S3DataReader(DataReader):
    async def read_json(self, uri: str) -> dict:
        parsed = urlparse(uri)
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
        session = aioboto3.Session()
        async with session.client("s3") as s3:
            obj = await s3.get_object(Bucket=bucket, Key=key)
            body = await obj["Body"].read()
        return json.loads(body)


def get_data_reader(storage_base_url: str) -> DataReader:
    if storage_base_url.startswith("s3://"):
        return S3DataReader()
    return LocalDataReader()
