"""Experiment configuration models."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Reproducible research experiment configuration."""

    name: str
    data_snapshot: str
    parameters: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("experiment name is required")
        if not self.data_snapshot:
            raise ValueError("data_snapshot is required")
