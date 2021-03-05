import yaml

from fastapi import APIRouter

router = APIRouter(prefix="/timeseries-service/api/v2", tags=['metadata'])


def load_metadata():
    with open("metadata.yml") as f:
        datasets = yaml.safe_load(f)
    result = {}
    for dataset in datasets:
        result[dataset['id']] = dataset
    return result


_metadata = load_metadata()


@router.get("/metadata")
def metadata():
    return list(_metadata.values())