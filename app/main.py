from fastapi import FastAPI
import logging
from app.settings import settings
from app.routers import datasets, metadata

settings.init()
logger = logging.getLogger(__name__)


app = FastAPI()
app.include_router(datasets.router)
app.include_router(metadata.router)
