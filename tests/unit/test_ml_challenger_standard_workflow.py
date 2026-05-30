from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def _load_module():
    path = Path("examples/run_ml_challenger_standard_workflow.py")
    spec = importlib.util.spec_from_file_location("run_ml_challenger_standard_workflow", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_standard_workflow_plan_includes_strict_no_leak_stages(tmp_path) -> None:
    module = _load_module()
    dataset_dir = tmp_path / "dataset"
    primary_score_dir = tmp_path / "primary_scores"
    baseline_backtest_dir = tmp_path / "baseline_backtest"
    dataset_dir.mkdir()
    primary_score_dir.mkdir()
    baseline_backtest_dir.mkdir()
    admission_report = tmp_path / "admission.json"
    admission_report.write_text("{}", encoding="utf-8")

    args = module._parse_args(
        [
            "--dataset-dir",
            str(dataset_dir),
            "--admission-report",
            str(admission_report),
            "--primary-score-dir",
            str(primary_score_dir),
            "--baseline-backtest-dir",
            str(baseline_backtest_dir),
            "--output-dir",
            str(tmp_path / "workflow"),
            "--include-features",
            "alpha_a",
            "alpha_b",
            "--train-start",
            "2023-01-01T00:00:00+08:00",
            "--history-train-end",
            "2024-12-31T23:59:59+08:00",
            "--history-test-start",
            "2025-01-01T00:00:00+08:00",
            "--history-test-end",
            "2025-12-31T23:59:59+08:00",
            "--live-train-end",
            "2025-12-31T23:59:59+08:00",
            "--live-start",
            "2026-01-01T00:00:00+08:00",
            "--live-end",
            "2026-05-15T15:00:00+08:00",
            "--primary-blend-weights",
            "0.5",
            "--selection-lookback-days",
            "63",
        ]
    )

    stages = module.build_standard_workflow_plan(args)
    commands = [item for stage in stages for item in stage.command]

    assert [stage.name for stage in stages] == [
        "01_train_walk_forward_ml_challenger",
        "02_build_primary_pool_blends",
        "03_backtest_history_primary_w050",
        "04_backtest_live_primary_w050",
        "05_build_adaptive_selector_lb063",
        "06_backtest_live_adaptive_lb063",
    ]
    assert "--allow-label-derived-features" not in commands
    assert "--policy-force-source-transition-exits" in commands
    assert "--fold" in commands
    assert "primary_pool_rerank" in commands
    assert "--commission-bps" in commands
