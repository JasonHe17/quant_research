from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from examples.run_framework_v1_benchmark import (
    BacktestJob,
    _can_launch_backtest_job,
    _run_command,
)


def test_framework_v1_benchmark_dry_run_writes_reproducible_plan(
    tmp_path: Path,
) -> None:
    script = Path("examples/run_framework_v1_benchmark.py")
    output_dir = tmp_path / "benchmark"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--output-dir",
            str(output_dir),
            "--start",
            "2024-01-02T09:35:00+08:00",
            "--end",
            "2024-01-03T15:00:00+08:00",
            "--max-symbols",
            "2",
            "--profile",
            "standard",
            "--label-horizon-bars",
            "48",
            "240",
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    summary = json.loads((output_dir / "benchmark_summary.json").read_text())
    commands = json.loads((output_dir / "commands.json").read_text())
    assert summary["status"] == "dry_run"
    assert summary["benchmark"] == "framework_v1"
    assert set(commands) == {
        "dataset",
        "factor_evaluation",
        "backtest_full_base",
        "backtest_year_2024_base",
        "backtest_full_high_cost",
    }
    assert summary["config"]["profile"] == "standard"
    assert summary["config"]["label_horizon_bars"] == [48, 240]
    assert summary["config"]["evaluation_workers"] == 8
    assert summary["config"]["backtest_workers"] == 2
    assert summary["config"]["backtest_memory_budget_gb"] is None
    assert "acceptance_plan" in summary
    assert "full_high_cost" in summary["backtests"]
    assert "build_baseline_a_alpha_dataset.py" in commands["dataset"][1]
    assert commands["dataset"][
        commands["dataset"].index("--horizon-bars") + 1 : commands["dataset"].index(
            "--entry-lag-bars"
        )
    ] == ["48", "240"]
    assert commands["factor_evaluation"][
        commands["factor_evaluation"].index("--label-column") + 1
    ] == "forward_return_48b"
    assert commands["factor_evaluation"][
        commands["factor_evaluation"].index("--horizon-label-columns") + 1
    ] == "forward_return_240b"
    assert "run_baseline_a_real_backtest.py" in commands["backtest_full_base"][1]
    assert "--streaming-chunk" in commands["backtest_full_base"]


def test_framework_v1_benchmark_profiles_define_expected_scenarios(
    tmp_path: Path,
) -> None:
    script = Path("examples/run_framework_v1_benchmark.py")

    quick = _dry_run(script, tmp_path / "quick", profile="quick")
    robust = _dry_run(script, tmp_path / "robust", profile="robust")

    assert set(quick["commands"]) == {
        "dataset",
        "factor_evaluation",
        "backtest_full_base",
    }
    assert "backtest_full_zero_cost" in robust["commands"]
    assert "backtest_full_trade_filter_stress" in robust["commands"]
    assert "backtest_full_high_cost" in robust["commands"]
    assert robust["acceptance_plan"]["profile"] == "robust"


def test_framework_v1_benchmark_can_plan_candidate_policy_validation(
    tmp_path: Path,
) -> None:
    script = Path("examples/run_framework_v1_benchmark.py")
    output_dir = tmp_path / "benchmark"
    admission = tmp_path / "factor_admission_report.json"
    admission.write_text("{}", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--output-dir",
            str(output_dir),
            "--start",
            "2024-01-02T09:35:00+08:00",
            "--end",
            "2024-01-03T15:00:00+08:00",
            "--max-symbols",
            "2",
            "--profile",
            "quick",
            "--candidate-admission-report",
            str(admission),
            "--candidate-policy-validation-methods",
            "decorrelated",
            "equal",
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    commands = json.loads((output_dir / "commands.json").read_text())
    command = commands["candidate_policy_validation"]
    assert "run_candidate_policy_validation.py" in command[1]
    assert command[command.index("--dataset-dir") + 1] == str(output_dir / "alpha_dataset")
    assert command[command.index("--admission-report") + 1] == str(admission)
    assert command[
        command.index("--methods") + 1 : command.index("--primary-method")
    ] == ["decorrelated", "equal"]
    assert command[command.index("--policy") + 1] == "partial_rebalance_daily"


def test_framework_v1_benchmark_command_failures_raise(tmp_path: Path) -> None:
    log_path = tmp_path / "logs" / "failed.log"

    with pytest.raises(RuntimeError, match="benchmark command failed with code 7"):
        _run_command(
            [sys.executable, "-c", "import sys; sys.exit(7)"],
            log_path=log_path,
        )

    assert log_path.exists()
    assert "sys.exit(7)" in log_path.read_text(encoding="utf-8")


def test_backtest_memory_gate_blocks_jobs_over_budget(
    tmp_path: Path,
) -> None:
    job = BacktestJob(
        stage_name="backtest_full_base",
        scenario_name="full_base",
        command=[sys.executable, "-c", "pass"],
        log_path=tmp_path / "log.txt",
        memory_estimate_gb=8.0,
    )

    assert not _can_launch_backtest_job(
        job,
        running_memory_gb=0.0,
        memory_budget_gb=4.0,
        running_count=0,
    )
    assert not _can_launch_backtest_job(
        job,
        running_memory_gb=6.0,
        memory_budget_gb=10.0,
        running_count=1,
    )
    assert _can_launch_backtest_job(
        job,
        running_memory_gb=2.0,
        memory_budget_gb=10.0,
        running_count=1,
    )


def _dry_run(script: Path, output_dir: Path, *, profile: str) -> dict[str, object]:
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--output-dir",
            str(output_dir),
            "--start",
            "2024-01-02T09:35:00+08:00",
            "--end",
            "2024-01-03T15:00:00+08:00",
            "--max-symbols",
            "2",
            "--profile",
            profile,
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    return json.loads((output_dir / "benchmark_summary.json").read_text())
