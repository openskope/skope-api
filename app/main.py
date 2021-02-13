from fastapi import FastAPI
import logging.config
import yaml
from app.routers import datasets, metadata

with open('deploy/logging.yml') as f:
    logging.config.dictConfig(yaml.safe_load(f))
logger = logging.getLogger(__name__)


class OutOfBoundsError(ValueError):
    pass


app = FastAPI()
app.include_router(datasets.router)
app.include_router(metadata.router)
