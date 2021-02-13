from fastapi import FastAPI
import logging
from app.routers import datasets, metadata

logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')


class OutOfBoundsError(ValueError):
    pass


app = FastAPI()
app.include_router(datasets.router)
app.include_router(metadata.router)
