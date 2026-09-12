"""COG and STAC ingest pipeline for timeseries datasets."""

from .config import PipelineConfig

__all__ = ["PipelineConfig", "run_pipeline"]


def __getattr__(name):
    if name == "run_pipeline":
        from .main import run_pipeline

        return run_pipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
