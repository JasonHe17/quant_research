"""Experiment runner scaffold."""

from __future__ import annotations

from quant_research.experiments.config import ExperimentConfig


class ExperimentRunner:
    """Runs reproducible experiment configurations."""

    def run(self, config: ExperimentConfig) -> object:
        raise NotImplementedError(f"experiment {config.name!r} is not implemented yet")
