"""Experiment configuration models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Reproducible research experiment configuration."""

    name: str
    data_snapshot: str
    parameters: dict[str, Any] = field(default_factory=dict)
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("experiment name is required")
        if not self.data_snapshot:
            raise ValueError("data_snapshot is required")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "data_snapshot": self.data_snapshot,
            "parameters": dict(self.parameters),
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExperimentConfig":
        return cls(
            name=str(payload["name"]),
            data_snapshot=str(payload["data_snapshot"]),
            parameters=dict(payload.get("parameters", {})),
            tags=tuple(str(item) for item in payload.get("tags", ())),
        )
