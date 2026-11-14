"""Structured validation findings (VAL-001, VAL-002)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Iterable, Literal

Severity = Literal["error", "warning"]


@dataclass(frozen=True)
class Finding:
    """One validation result, tied to the requirement it enforces."""

    requirement: str
    severity: Severity
    message: str
    path: str | None = None
    dataset: str | None = None
    variable: str | None = None
    chunk: str | None = None
    band: int | None = None

    def to_text(self) -> str:
        context = [
            f"{name}={value}"
            for name, value in (
                ("dataset", self.dataset),
                ("variable", self.variable),
                ("chunk", self.chunk),
                ("band", self.band),
                ("path", self.path),
            )
            if value is not None
        ]
        suffix = f" [{', '.join(context)}]" if context else ""
        return f"{self.severity.upper():7} {self.requirement}: {self.message}{suffix}"


@dataclass
class Report:
    """Findings for one validation pass or a whole build (VAL-001)."""

    findings: list[Finding] = field(default_factory=list)

    def add(self, requirement: str, message: str, *, severity: Severity = "error", **context) -> None:
        self.findings.append(Finding(requirement, severity, message, **context))

    def warn(self, requirement: str, message: str, **context) -> None:
        self.add(requirement, message, severity="warning", **context)

    def extend(self, findings: Iterable[Finding]) -> None:
        self.findings.extend(findings)

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "error"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_if_errors(self, stage: str) -> None:
        if self.errors:
            raise ReleaseError(stage, self)

    def to_text(self) -> str:
        return "\n".join(f.to_text() for f in self.findings)

    def to_json(self) -> str:
        return json.dumps([asdict(f) for f in self.findings], indent=2, sort_keys=True)


class ReleaseError(Exception):
    """A validation pass failed; carries the findings that failed it."""

    def __init__(self, stage: str, report: Report):
        self.stage = stage
        self.report = report
        first = report.errors[0].to_text() if report.errors else "no findings"
        super().__init__(f"{stage} failed with {len(report.errors)} error(s); first: {first}")
