from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from quant_research.experiments import (
    ExperimentConfig,
    ExperimentRegistry,
    ExperimentRun,
    ExperimentRunner,
    ExperimentRunStore,
    collect_experiment_provenance,
)


def test_experiment_config_round_trips_dict() -> None:
    config = ExperimentConfig(
        name="factor-smoke",
        data_snapshot="2026-05-09",
        parameters={"symbols": ["600000.SH"]},
        tags=("smoke", "factor"),
    )

    loaded = ExperimentConfig.from_dict(config.to_dict())

    assert loaded == config


def test_experiment_run_store_round_trips_json(tmp_path: Path) -> None:
    config = ExperimentConfig(name="factor-smoke", data_snapshot="2026-05-09")
    run = ExperimentRun.create(
        config=config,
        started_at=datetime(2026, 5, 9, tzinfo=timezone.utc),
        artifacts={"factor": "research_store/factors/close_return.parquet"},
        metrics={"ic": 0.03},
        cache_manifest_ids=("manifest-1",),
    )
    store = ExperimentRunStore(root=tmp_path)

    path = store.write(run)
    loaded = store.read(run.run_id)

    assert path == store.run_path(run.run_id)
    assert loaded == run
    assert path.read_text(encoding="utf-8").endswith("\n")


def test_experiment_runner_creates_and_completes_runs(tmp_path: Path) -> None:
    store = ExperimentRunStore(root=tmp_path)
    runner = ExperimentRunner(run_store=store)
    config = ExperimentConfig(name="factor-smoke", data_snapshot="2026-05-09")
    artifact_path = tmp_path / "factor.parquet"
    artifact_path.write_text("artifact", encoding="utf-8")

    created = runner.create_run(config, command_line=("pytest",))
    completed = runner.complete_run(
        created,
        artifacts={"factor": str(artifact_path)},
        metrics={"total_return": 0.12},
        cache_manifest_ids=("manifest-1", "manifest-2"),
        command_line=("pytest",),
    )
    loaded = store.read(created.run_id)

    assert created.status == "created"
    assert completed.status == "completed"
    assert completed.finished_at is not None
    assert loaded == completed
    assert loaded.metrics["total_return"] == 0.12
    assert loaded.provenance["data_snapshot"] == "2026-05-09"
    assert "environment_hash" in loaded.provenance
    assert "factor" in loaded.artifact_hashes
    assert store.verify_artifacts(loaded) == {"factor": True}


def test_experiment_run_store_lists_by_status(tmp_path: Path) -> None:
    store = ExperimentRunStore(root=tmp_path)
    first = ExperimentRun.create(
        config=ExperimentConfig(name="first", data_snapshot="2026-05-09"),
        started_at=datetime(2026, 5, 9, 1, tzinfo=timezone.utc),
    )
    second = ExperimentRun.create(
        config=ExperimentConfig(name="second", data_snapshot="2026-05-09"),
        started_at=datetime(2026, 5, 9, 2, tzinfo=timezone.utc),
    ).complete(metrics={"sharpe": 1.2})
    _ = store.write(first)
    _ = store.write(second)

    assert store.list() == (first, second)
    assert store.list(status="completed") == (second,)


def test_experiment_run_store_writes_immutable_run_dirs_and_summary(tmp_path: Path) -> None:
    store = ExperimentRunStore(root=tmp_path)
    run = ExperimentRun.create(
        config=ExperimentConfig(name="immutable", data_snapshot="2026-05-09"),
        started_at=datetime(2026, 5, 9, tzinfo=timezone.utc),
    ).complete(metrics={"ic": 0.1})

    path = store.write_immutable(run)
    same_path = store.write_immutable(run)
    _ = store.write(run)
    summary = store.summary_frame()

    assert path == same_path
    assert path.name == "run.json"
    assert summary.loc[0, "metric:ic"] == 0.1


def test_experiment_registry_lists_configs_by_name() -> None:
    registry = ExperimentRegistry()
    second = ExperimentConfig(name="z-exp", data_snapshot="2026-05-09")
    first = ExperimentConfig(name="a-exp", data_snapshot="2026-05-09")

    registry.register(second)
    registry.register(first)

    assert registry.list() == (first, second)


def test_experiment_registry_registers_grid_and_walk_forward_configs() -> None:
    registry = ExperimentRegistry()

    grid = registry.register_grid(
        name_prefix="grid",
        data_snapshot="2026-05-09",
        parameter_grid={"lookback": [1, 3], "top_n": [10]},
    )
    walk = registry.register_walk_forward(
        name_prefix="wf",
        data_snapshot="2026-05-09",
        windows=({"name": "2025", "train_end": "2025-06-30"},),
    )

    assert [config.name for config in grid] == ["grid-001", "grid-002"]
    assert walk[0].parameters["window"]["name"] == "2025"
    assert len(registry.list()) == 3


def test_collect_experiment_provenance_records_environment() -> None:
    provenance = collect_experiment_provenance(
        data_snapshot="2026-05-09",
        command_line=("cmd",),
        packages=("pandas", "missing-package-for-test"),
    )

    assert provenance["data_snapshot"] == "2026-05-09"
    assert provenance["command_line"] == ["cmd"]
    assert provenance["package_versions"]["missing-package-for-test"] is None
    assert isinstance(provenance["environment_hash"], str)
