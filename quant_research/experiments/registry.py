"""Experiment registry scaffold."""

from __future__ import annotations

import itertools
from typing import Any

from quant_research.experiments.config import ExperimentConfig


class ExperimentRegistry:
    """In-memory experiment registry."""

    def __init__(self) -> None:
        self._items: dict[str, ExperimentConfig] = {}

    def register(self, config: ExperimentConfig) -> None:
        self._items[config.name] = config

    def register_grid(
        self,
        *,
        name_prefix: str,
        data_snapshot: str,
        parameter_grid: dict[str, tuple[Any, ...] | list[Any]],
        base_parameters: dict[str, Any] | None = None,
        tags: tuple[str, ...] = ("grid",),
    ) -> tuple[ExperimentConfig, ...]:
        """Register one config for each parameter-grid combination."""

        keys = tuple(parameter_grid)
        configs: list[ExperimentConfig] = []
        for index, values in enumerate(
            itertools.product(*(tuple(parameter_grid[key]) for key in keys)),
            start=1,
        ):
            parameters = {**dict(base_parameters or {}), **dict(zip(keys, values))}
            config = ExperimentConfig(
                name=f"{name_prefix}-{index:03d}",
                data_snapshot=data_snapshot,
                parameters=parameters,
                tags=tags,
            )
            self.register(config)
            configs.append(config)
        return tuple(configs)

    def register_walk_forward(
        self,
        *,
        name_prefix: str,
        data_snapshot: str,
        windows: tuple[dict[str, Any], ...],
        base_parameters: dict[str, Any] | None = None,
        tags: tuple[str, ...] = ("walk_forward",),
    ) -> tuple[ExperimentConfig, ...]:
        """Register one config per walk-forward window."""

        configs: list[ExperimentConfig] = []
        for index, window in enumerate(windows, start=1):
            window_name = str(window.get("name", f"{index:03d}"))
            parameters = {**dict(base_parameters or {}), "window": dict(window)}
            config = ExperimentConfig(
                name=f"{name_prefix}-{window_name}",
                data_snapshot=data_snapshot,
                parameters=parameters,
                tags=tags,
            )
            self.register(config)
            configs.append(config)
        return tuple(configs)

    def get(self, name: str) -> ExperimentConfig:
        return self._items[name]

    def list(self) -> tuple[ExperimentConfig, ...]:
        return tuple(self._items[name] for name in sorted(self._items))
