import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass(frozen=True)
class PipelineConfig:
    input_dir: str = "/data/skope/cog-input"
    input_manifest_path: str | None = None
    output_dir: str | None = None
    dataset_name: str = "paleocar_v3"
    metadata_file_path: str = "metadata.yml"
    trunc_to_uint16: bool = True
    preflight_only: bool = False
    max_bands_per_slice: int = 100
    dataset_start_datetime: datetime = datetime(103, 1, 1, tzinfo=timezone.utc)
    dataset_time_delta: dict[str, int] = field(default_factory=lambda: {"years": 1})

    @classmethod
    def from_env(cls) -> "PipelineConfig":
        return cls(
            input_dir=os.environ.get("INPUT_DIR", cls.input_dir),
            input_manifest_path=os.environ.get("INPUT_MANIFEST_PATH"),
            output_dir=os.environ.get("OUTPUT_DIR"),
            dataset_name=os.environ.get("DATASET_NAME", cls.dataset_name),
            metadata_file_path=os.environ.get(
                "METADATA_FILE_PATH", cls.metadata_file_path
            ),
            trunc_to_uint16=_env_bool("TRUNC_TO_UINT16", cls.trunc_to_uint16),
            preflight_only=_env_bool("PREFLIGHT_ONLY", cls.preflight_only),
            max_bands_per_slice=int(
                os.environ.get("MAX_BANDS_PER_SLICE", cls.max_bands_per_slice)
            ),
            dataset_start_datetime=_env_datetime(
                "DATASET_START_DATETIME", cls.dataset_start_datetime
            ),
            dataset_time_delta=_env_json_dict("DATASET_TIME_DELTA", {"years": 1}),
        )

    @property
    def resolved_output_dir(self) -> str:
        return self.output_dir or os.path.join(self.input_dir, self.dataset_name)


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _env_datetime(name: str, default: datetime) -> datetime:
    value = os.environ.get(name)
    if value is None:
        return default
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return datetime.fromisoformat(value)


def _env_json_dict(name: str, default: dict[str, int]) -> dict[str, int]:
    value = os.environ.get(name)
    if value is None:
        return default
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError(f"{name} must be a JSON object.")
    return parsed
