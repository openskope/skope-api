import yaml

from fastapi import APIRouter

router = APIRouter(prefix="/metadata", tags=['metadata'])


def load_metadata():
    with open("metadata.yml") as f:
        datasets = yaml.safe_load(f)
    result = {}
    for dataset in datasets:
        result[dataset['id']] = dataset
    return result


_metadata = load_metadata()


@router.get("/")
def metadata():
    return list(_metadata.values())