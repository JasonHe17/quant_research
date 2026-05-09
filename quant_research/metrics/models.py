"""Metric and report models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class MetricResult:
    """One named metric value with optional metadata."""

    name: str
    value: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("metric name is required")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class MetricsReport:
    """Collection of metric results for one research object."""

    name: str
    metrics: tuple[MetricResult, ...]
    metadata: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("report name is required")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "metrics": [metric.to_dict() for metric in self.metrics],
            "metadata": dict(self.metadata),
            "artifacts": dict(self.artifacts),
        }

    def with_artifacts(self, artifacts: dict[str, str]) -> "MetricsReport":
        return MetricsReport(
            name=self.name,
            metrics=self.metrics,
            metadata=dict(self.metadata),
            artifacts={**self.artifacts, **artifacts},
        )
