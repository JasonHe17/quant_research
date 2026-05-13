"""Experiment runner scaffold."""

from __future__ import annotations

from quant_research.experiments.config import ExperimentConfig
from quant_research.experiments.provenance import (
    artifact_hashes,
    collect_experiment_provenance,
)
from quant_research.experiments.run import ExperimentRun, ExperimentRunStore


class ExperimentRunner:
    """Runs reproducible experiment configurations."""

    def __init__(self, *, run_store: ExperimentRunStore | None = None) -> None:
        self.run_store = run_store

    def create_run(
        self,
        config: ExperimentConfig,
        *,
        catalog_path: str | None = None,
        command_line: tuple[str, ...] | None = None,
    ) -> ExperimentRun:
        run = ExperimentRun.create(
            config=config,
            provenance=collect_experiment_provenance(
                data_snapshot=config.data_snapshot,
                catalog_path=catalog_path,
                command_line=command_line,
            ),
        )
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
        catalog_path: str | None = None,
        command_line: tuple[str, ...] | None = None,
    ) -> ExperimentRun:
        next_artifacts = dict(artifacts or {})
        completed = run.complete(
            artifacts=next_artifacts,
            metrics=metrics,
            cache_manifest_ids=cache_manifest_ids,
            provenance=collect_experiment_provenance(
                data_snapshot=run.config.data_snapshot,
                catalog_path=catalog_path,
                command_line=command_line,
            ),
            artifact_hashes=artifact_hashes(next_artifacts),
        )
        if self.run_store is not None:
            self.run_store.write(completed)
        return completed
