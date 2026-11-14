"""Temporal axes and the time-to-band rule (OBS-004, API-002 to API-004).

This module has no GDAL dependency so the API can use the same rule when it
resolves tiles and extractions (release consumption specification PROTO-006).

    timestep(i) = key(origin + step * i)          # regular
                = values[i]                       # enumerated
    chunk = index // chunk_size; bidx = index % chunk_size + 1
    file  = cogs/<variable>/<dataset>--<fname(start)>--<fname(end)>.tif
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Literal

Precision = Literal["year", "month", "day", "datetime"]
_KEY_FORMATS = {
    "year": re.compile(r"^(\d{4})$"),
    "month": re.compile(r"^(\d{4})-(\d{2})$"),
    "day": re.compile(r"^(\d{4})-(\d{2})-(\d{2})$"),
    "datetime": re.compile(r"^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})Z$"),
}
_STEP = re.compile(r"^P(?:(\d+)([YMD])|T(\d+)([HMS]))$")
# Step units each precision can express without inventing a finer component.
_ALLOWED_UNITS = {
    "year": {"Y"},
    "month": {"Y", "M"},
    "day": {"Y", "M", "D"},
    "datetime": {"Y", "M", "D", "H", "MIN", "S"},
}


class AxisError(ValueError):
    """A timestep or axis definition violates the axis rules."""


class MalformedTimestep(AxisError):
    """The key is not a canonical ISO timestep at the dataset's precision (PROTO-009: 422)."""


class UnknownTimestep(AxisError):
    """The key is well formed but names no timestep of the axis (PROTO-009: 404)."""


def parse_key(key: str, precision: Precision) -> datetime:
    match = _KEY_FORMATS[precision].match(key)
    if not match:
        raise MalformedTimestep(f"{key!r} is not a {precision}-precision ISO timestep")
    parts = [int(p) for p in match.groups()]
    parts += [1] * (3 - min(len(parts), 3)) + [0] * (6 - max(len(parts), 3))
    try:
        return datetime(*parts[:6], tzinfo=timezone.utc)
    except ValueError as exc:
        raise MalformedTimestep(f"{key!r} is not a valid date: {exc}") from exc


def format_key(instant: datetime, precision: Precision) -> str:
    if precision == "year":
        return f"{instant.year:04d}"
    if precision == "month":
        return f"{instant.year:04d}-{instant.month:02d}"
    if precision == "day":
        return f"{instant.year:04d}-{instant.month:02d}-{instant.day:02d}"
    return f"{instant.year:04d}-{instant.month:02d}-{instant.day:02d}T{instant:%H:%M:%S}Z"


def parse_step(step: str) -> tuple[int, str]:
    """Return (count, unit) with unit one of Y, M, D, H, MIN, S."""
    match = _STEP.match(step)
    if not match:
        raise AxisError(f"step {step!r} must be a single-unit ISO 8601 duration such as P1Y or PT6H")
    if match.group(1):
        count, unit = int(match.group(1)), match.group(2)
    else:
        count, unit = int(match.group(3)), {"H": "H", "M": "MIN", "S": "S"}[match.group(4)]
    if count < 1:
        raise AxisError("a regular step must be positive (OBS-004)")
    return count, unit


def add_steps(instant: datetime, count: int, unit: str, n: int) -> datetime:
    """Calendar arithmetic: `instant + n * (count unit)`."""
    if unit in {"Y", "M"}:
        months = instant.year * 12 + (instant.month - 1) + n * count * (12 if unit == "Y" else 1)
        year, month = divmod(months, 12)
        try:
            return instant.replace(year=year, month=month + 1)
        except ValueError as exc:
            raise AxisError(f"{instant:%Y-%m-%d} plus {n * count}{unit} is not a calendar date") from exc
    seconds = {"D": 86400, "H": 3600, "MIN": 60, "S": 1}[unit]
    return instant + timedelta(seconds=n * count * seconds)


def end_of_period(instant: datetime, step: str) -> datetime:
    """Last second of the aggregation period that starts at `instant`."""
    count, unit = parse_step(step)
    return add_steps(instant, count, unit, 1) - timedelta(seconds=1)


def fname(key: str) -> str:
    """Filename form of a timestep key (Section 5.3)."""
    return key.replace(":", "-")


@dataclass(frozen=True)
class Axis:
    kind: Literal["regular", "enumerated"]
    precision: Precision
    count: int
    origin: str
    step: str | None = None
    values: tuple[str, ...] | None = None

    @classmethod
    def regular(cls, origin: str, step: str, end: str, precision: Precision) -> "Axis":
        start = parse_key(origin, precision)
        last = parse_key(end, precision)
        count_, unit = parse_step(step)
        if unit not in _ALLOWED_UNITS[precision]:
            raise AxisError(f"step {step} is finer than the axis precision {precision}")
        if last < start:
            raise AxisError(f"end {end} precedes origin {origin} (OBS-004)")
        # Count whole steps; the sequence must terminate exactly at `end` (OBS-004).
        if unit in {"Y", "M"}:
            span = (last.year * 12 + last.month) - (start.year * 12 + start.month)
            estimate = span // (count_ * (12 if unit == "Y" else 1))
        else:
            seconds = {"D": 86400, "H": 3600, "MIN": 60, "S": 1}[unit] * count_
            estimate = int((last - start).total_seconds() // seconds)
        if add_steps(start, count_, unit, estimate) != last:
            raise AxisError(f"origin {origin} plus whole steps of {step} never reaches end {end} (OBS-004)")
        return cls("regular", precision, estimate + 1, origin, step=step)

    @classmethod
    def enumerated(cls, values: list[str] | tuple[str, ...], precision: Precision) -> "Axis":
        if not values:
            raise AxisError("an enumerated axis needs at least one value")
        instants = [parse_key(v, precision) for v in values]
        for prev, cur, key in zip(instants, instants[1:], values[1:]):
            if cur <= prev:
                raise AxisError(f"enumerated values must be unique and strictly increasing at {key} (OBS-004)")
        return cls("enumerated", precision, len(values), values[0], values=tuple(values))

    def key(self, index: int) -> str:
        if not 0 <= index < self.count:
            raise UnknownTimestep(f"index {index} is outside the axis [0, {self.count})")
        if self.values is not None:
            return self.values[index]
        count, unit = parse_step(self.step)
        return format_key(add_steps(parse_key(self.origin, self.precision), count, unit, index), self.precision)

    def keys(self) -> tuple[str, ...]:
        if self.values is not None:
            return self.values
        return tuple(self.key(i) for i in range(self.count))

    def index(self, key: str) -> int:
        """Exact lookup: malformed keys, off-axis keys and between-step keys are rejected."""
        instant = parse_key(key, self.precision)
        if self.values is not None:
            try:
                return self.values.index(key)
            except ValueError:
                raise UnknownTimestep(f"{key} is not on the axis") from None
        start = parse_key(self.origin, self.precision)
        count, unit = parse_step(self.step)
        if unit in {"Y", "M"}:
            months = (instant.year * 12 + instant.month) - (start.year * 12 + start.month)
            candidate = months // (count * (12 if unit == "Y" else 1))
        else:
            seconds = {"D": 86400, "H": 3600, "MIN": 60, "S": 1}[unit] * count
            candidate = int((instant - start).total_seconds() // seconds)
        if not 0 <= candidate < self.count or self.key(candidate) != key:
            raise UnknownTimestep(f"{key} is not on the axis (origin {self.origin}, step {self.step})")
        return candidate

    def select(self, lower: str, upper: str) -> list[int]:
        """Indices of the timesteps within [lower, upper] (PROTO-007)."""
        lo, hi = parse_key(lower, self.precision), parse_key(upper, self.precision)
        return [i for i, k in enumerate(self.keys()) if lo <= parse_key(k, self.precision) <= hi]


@dataclass(frozen=True)
class ChunkLayout:
    """Chunking in index space (API-002, ORG-008, ORG-009)."""

    dataset_id: str
    axis: Axis
    chunk_size: int

    def __post_init__(self):
        if self.chunk_size < 1:
            raise AxisError("chunk_size must be at least 1")

    @property
    def chunk_count(self) -> int:
        return -(-self.axis.count // self.chunk_size)

    def bounds(self, chunk: int) -> tuple[int, int]:
        """Inclusive first and last index of a chunk."""
        if not 0 <= chunk < self.chunk_count:
            raise AxisError(f"chunk {chunk} is outside [0, {self.chunk_count})")
        first = chunk * self.chunk_size
        return first, min(first + self.chunk_size, self.axis.count) - 1

    def item_id(self, chunk: int) -> str:
        first, last = self.bounds(chunk)
        return f"{self.dataset_id}--{fname(self.axis.key(first))}--{fname(self.axis.key(last))}"

    def cog_path(self, variable_id: str, chunk: int) -> str:
        return f"cogs/{variable_id}/{self.item_id(chunk)}.tif"

    def locate(self, index: int) -> tuple[int, int]:
        """(chunk, 1-based band) of an axis index."""
        if not 0 <= index < self.axis.count:
            raise UnknownTimestep(f"index {index} is outside the axis")
        return index // self.chunk_size, index % self.chunk_size + 1

    def resolve(self, variable_id: str, key: str) -> tuple[str, int]:
        """Release-relative COG path and 1-based band of one timestep."""
        chunk, band = self.locate(self.axis.index(key))
        return self.cog_path(variable_id, chunk), band

    def chunk_keys(self, chunk: int) -> tuple[str, ...]:
        first, last = self.bounds(chunk)
        keys = self.axis.keys()
        return keys[first : last + 1]
