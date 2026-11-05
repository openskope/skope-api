"""Compose the app registry from the pinned releases (PIN-001 to PIN-006).

At startup the API reads the pin file, checks every pinned release (PIN-004)
and composes the registry from the releases' overviews, unchanged (PIN-005).
Any failed check refuses the whole registry: no dataset is served on its own.
Each failure is logged as one structured error, and the caller exits.

`verify_releases(full=True)` adds the full byte check that promotion runs
before a deploy (release spec TXN-011).
"""

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml
from pydantic import ValidationError

from app.registry.models import (
    Overview,
    OverviewTimeRegular,
    Pin,
    PinnedRelease,
    ReleaseManifest,
)
from app.vendor.timeaxis import Axis, AxisError, ChunkLayout, MalformedTimestep

logger = logging.getLogger(__name__)

MANIFEST = "release-manifest.json"
OVERVIEW = "overview.yml"


@dataclass(frozen=True)
class Refusal:
    """One failed startup check (PIN-004): what was expected and what was found."""

    requirement: str
    dataset: str
    release_id: str
    check: str
    expected: str
    observed: str


class ReleaseRefused(Exception):
    def __init__(self, refusals: list[Refusal]):
        self.refusals = refusals
        super().__init__(f"refusing to serve: {len(refusals)} release check(s) failed")


def log_refusals(refusals: list[Refusal]) -> None:
    for refusal in refusals:
        logger.error(
            "release check failed: %s", json.dumps(asdict(refusal), sort_keys=True)
        )


# ---------------------------------------------------------------------------
# The served release


@dataclass(frozen=True)
class ServedRelease:
    """One pinned, verified release and the time-to-band rule for its files."""

    path: Path
    overview: Overview
    layout: ChunkLayout

    @property
    def dataset_id(self) -> str:
        return self.overview.dataset.id

    @property
    def axis(self) -> Axis:
        return self.layout.axis

    def has_variable(self, variable_id: str) -> bool:
        return variable_id in self.overview.variables

    def resolve(self, variable_id: str, key: str) -> tuple[str, int]:
        """Absolute COG path and 1-based band of one timestep (PROTO-006).

        `key` must be a canonical key at the dataset's precision; a malformed
        key raises MalformedTimestep and an off-axis key UnknownTimestep.
        """
        relative, band = self.layout.resolve(variable_id, key)
        return str(self.path / relative), band

    def select(
        self, variable_id: str, lower: str, upper: str
    ) -> tuple[dict[str, list[int]], list[str]]:
        """COG paths with their bands, and the timesteps, in [lower, upper] (PROTO-007).

        The bounds may be finer than the dataset's precision when the extra
        components are the canonical start of the period ("0103-01-01" for a
        yearly dataset). Files and bands come out in axis order.
        """
        lower = normalize_key(lower, self.axis.precision)
        upper = normalize_key(upper, self.axis.precision)
        indices = self.axis.select(lower, upper)
        keys = self.axis.keys()
        mapping: dict[str, list[int]] = {}
        for index in indices:
            chunk, band = self.layout.locate(index)
            path = str(self.path / self.layout.cog_path(variable_id, chunk))
            mapping.setdefault(path, []).append(band)
        return mapping, [keys[i] for i in indices]


@dataclass(frozen=True)
class AppRegistry:
    releases: dict[str, ServedRelease]

    def get(self, dataset_id: str) -> ServedRelease | None:
        return self.releases.get(dataset_id)


# ---------------------------------------------------------------------------
# Timestep keys

_PRECISION_LENGTH = {"year": 4, "month": 7, "day": 10, "datetime": 20}
_CANONICAL_START = "0000-01-01T00:00:00Z"


def normalize_key(key: str, precision: str) -> str:
    """Truncate a finer key to `precision` when the dropped part is the period start.

    "0103-01-01" becomes "0103" for a yearly dataset; "0103-06-01" and the
    coarser "0103" for a monthly dataset are malformed (PROTO-009: 422).
    """
    length = _PRECISION_LENGTH[precision]
    if len(key) > length and key[length:] == _CANONICAL_START[length : len(key)]:
        return key[:length]
    if len(key) != length:
        raise MalformedTimestep(f"{key!r} is not a {precision}-precision ISO timestep")
    return key


# ---------------------------------------------------------------------------
# Loading and checking


def load_pin(path: Path) -> Pin:
    return Pin.model_validate(yaml.safe_load(path.read_text("utf-8")))


def multihash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return "1220" + digest.hexdigest()


class _Checker:
    def __init__(self, pinned: PinnedRelease):
        self.pinned = pinned
        self.refusals: list[Refusal] = []

    def refuse(self, requirement: str, check: str, expected, observed) -> None:
        self.refusals.append(
            Refusal(
                requirement,
                self.pinned.dataset,
                self.pinned.release_id,
                check,
                str(expected),
                str(observed),
            )
        )

    def same(self, requirement: str, check: str, expected, observed) -> bool:
        if expected != observed:
            self.refuse(requirement, check, expected, observed)
            return False
        return True


def _layout(overview: Overview) -> ChunkLayout:
    time = overview.dataset.time
    if isinstance(time, OverviewTimeRegular):
        provisional = Axis(
            "regular", time.precision, time.count, time.origin, time.step
        )
        axis = Axis.regular(
            time.origin, time.step, provisional.key(time.count - 1), time.precision
        )
    else:
        axis = Axis.enumerated(time.values, time.precision)
    if axis.count != time.count:
        raise AxisError(
            f"the axis has {axis.count} timesteps, the overview says {time.count}"
        )
    return ChunkLayout(overview.dataset.id, axis, time.chunk_size)


def _check_release(
    pinned: PinnedRelease, release_root: Path, full: bool
) -> tuple[ServedRelease | None, list[Refusal]]:
    c = _Checker(pinned)
    path = release_root / pinned.release_id
    manifest_path = path / MANIFEST
    if not manifest_path.is_file():
        c.refuse("PIN-004", "release manifest exists", str(manifest_path), "missing")
        return None, c.refusals

    manifest_bytes = manifest_path.read_bytes()
    c.same(
        "PIN-004",
        "manifest digest",
        pinned.manifest_sha256,
        hashlib.sha256(manifest_bytes).hexdigest(),
    )
    try:
        manifest = ReleaseManifest.model_validate_json(manifest_bytes)
    except ValidationError as exc:
        c.refuse("PIN-004", "release manifest is valid", "schema 1.0.0", exc)
        return None, c.refusals
    c.same("PIN-004", "manifest release ID", pinned.release_id, manifest.release_id)
    c.same("PIN-004", "manifest dataset", pinned.dataset, manifest.dataset.id)

    # Every inventoried file exists with its recorded size; promotion also
    # compares every checksum (TXN-011), logging progress since it reads
    # every byte of the release.
    files = {f.path: f for f in manifest.files}
    total_gb = sum(f.size for f in manifest.files) / 1e9
    checked, checked_gb = 0, 0.0
    if full:
        logger.info(
            "%s: checking the checksum of %d files, %.1f GB",
            pinned.release_id,
            len(manifest.files),
            total_gb,
        )
    for entry in manifest.files:
        file_path = path / entry.path
        if not file_path.is_file():
            c.refuse("PIN-004", f"file exists: {entry.path}", entry.size, "missing")
            continue
        if not c.same(
            "PIN-004", f"file size: {entry.path}", entry.size, file_path.stat().st_size
        ):
            continue
        if full:
            c.same(
                "TXN-011",
                f"file checksum: {entry.path}",
                entry.checksum,
                multihash(file_path),
            )
            checked += 1
            checked_gb += entry.size / 1e9
            logger.info(
                "%s: %d of %d files checked, %.1f of %.1f GB",
                pinned.release_id,
                checked,
                len(manifest.files),
                checked_gb,
                total_gb,
            )

    overview_entry = files.get(OVERVIEW)
    overview_path = path / OVERVIEW
    if overview_entry is None or not overview_path.is_file():
        c.refuse("PIN-004", "overview in the manifest", OVERVIEW, "missing")
        return None, c.refusals
    overview_bytes = overview_path.read_bytes()
    c.same(
        "PIN-004",
        "overview checksum",
        overview_entry.checksum,
        "1220" + hashlib.sha256(overview_bytes).hexdigest(),
    )
    try:
        overview = Overview.model_validate(yaml.safe_load(overview_bytes))
    except (ValidationError, yaml.YAMLError) as exc:
        c.refuse("PIN-004", "overview is valid", "schema 1.0.0", exc)
        return None, c.refusals
    c.same("PIN-004", "overview release ID", pinned.release_id, overview.release_id)
    c.same(
        "PIN-004",
        "overview declaration digest",
        pinned.declaration_digest,
        overview.declaration_digest,
    )
    c.same("PIN-004", "overview dataset", pinned.dataset, overview.dataset.id)

    # Not supported yet; serving these without them would be wrong.
    if overview.dataset.profile != "TemporalCubeDataset":
        c.refuse(
            "PROTO-008",
            "static datasets are not served yet",
            "TemporalCubeDataset",
            overview.dataset.profile,
        )
        return None, c.refusals
    for variable_id, variable in overview.variables.items():
        if (variable.scale, variable.offset) != (1.0, 0.0):
            c.refuse(
                "PROTO-010",
                f"{variable_id}: scale and offset (unscaling is not implemented yet)",
                (1.0, 0.0),
                (variable.scale, variable.offset),
            )

    try:
        layout = _layout(overview)
    except AxisError as exc:
        c.refuse("API-002", "time axis", "a valid axis", exc)
        return None, c.refusals
    # The time-to-band rule must name exactly the COGs the manifest holds.
    expected_cogs = {
        layout.cog_path(variable_id, chunk)
        for variable_id in overview.variables
        for chunk in range(layout.chunk_count)
    }
    listed_cogs = {p for p in files if p.startswith("cogs/")}
    c.same(
        "API-002",
        "COGs named by the time axis",
        sorted(expected_cogs),
        sorted(listed_cogs),
    )

    if c.refusals:
        return None, c.refusals
    return ServedRelease(path, overview, layout), []


def verify_releases(pin: Pin, release_root: Path, *, full: bool = False) -> AppRegistry:
    """Check every pinned release and compose the registry, or raise ReleaseRefused."""
    refusals: list[Refusal] = []
    releases: dict[str, ServedRelease] = {}
    seen: set[str] = set()
    for pinned in pin.releases:
        if pinned.dataset in seen:
            refusals.append(
                Refusal(
                    "PIN-005",
                    pinned.dataset,
                    pinned.release_id,
                    "one release per dataset",
                    "1",
                    "2 or more",
                )
            )
            continue
        seen.add(pinned.dataset)
        served, failed = _check_release(pinned, release_root, full)
        refusals.extend(failed)
        if served is not None:
            releases[served.dataset_id] = served
    if refusals:
        raise ReleaseRefused(refusals)
    return AppRegistry(releases)
