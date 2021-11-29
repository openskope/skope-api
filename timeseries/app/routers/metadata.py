import yaml

from fastapi import APIRouter

from ..stores import load_api_metadata

router = APIRouter(prefix="/timeseries-service/api/v2", tags=['metadata'])

_metadata = load_api_metadata()

@router.get("/metadata")
def metadata():
    return list(_metadata.values())
