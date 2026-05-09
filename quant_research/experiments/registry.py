"""Experiment registry scaffold."""

from __future__ import annotations

from quant_research.experiments.config import ExperimentConfig


class ExperimentRegistry:
    """In-memory experiment registry."""

    def __init__(self) -> None:
        self._items: dict[str, ExperimentConfig] = {}

    def register(self, config: ExperimentConfig) -> None:
        self._items[config.name] = config

    def get(self, name: str) -> ExperimentConfig:
        return self._items[name]

    def list(self) -> tuple[ExperimentConfig, ...]:
        return tuple(self._items[name] for name in sorted(self._items))
