"""Experiment runner scaffold."""

from __future__ import annotations

from quant_research.experiments.config import ExperimentConfig
from quant_research.experiments.run import ExperimentRun, ExperimentRunStore


class ExperimentRunner:
    """Runs reproducible experiment configurations."""

    def __init__(self, *, run_store: ExperimentRunStore | None = None) -> None:
        self.run_store = run_store

    def create_run(self, config: ExperimentConfig) -> ExperimentRun:
        run = ExperimentRun.create(config=config)
        if self.run_store is not None:
            self.run_store.write(run)
        return run

    def complete_run(
        self,
        run: ExperimentRun,
        *,
        artifacts: dict[str, str] | None = None,
        metrics: dict[str, float] | None = None,
        cache_manifest_ids: tuple[str, ...] | None = None,
    ) -> ExperimentRun:
        completed = run.complete(
            artifacts=artifacts,
            metrics=metrics,
            cache_manifest_ids=cache_manifest_ids,
        )
        if self.run_store is not None:
            self.run_store.write(completed)
        return completed
