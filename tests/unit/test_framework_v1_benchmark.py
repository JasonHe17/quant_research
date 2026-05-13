from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from examples.run_framework_v1_benchmark import _run_command


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
    assert summary["config"]["evaluation_workers"] == 4
    assert "acceptance_plan" in summary
    assert "full_high_cost" in summary["backtests"]
    assert "build_baseline_a_alpha_dataset.py" in commands["dataset"][1]
    assert "run_baseline_a_real_backtest.py" in commands["backtest_full_base"][1]


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


def test_framework_v1_benchmark_command_failures_raise(tmp_path: Path) -> None:
    log_path = tmp_path / "logs" / "failed.log"

    with pytest.raises(RuntimeError, match="benchmark command failed with code 7"):
        _run_command(
            [sys.executable, "-c", "import sys; sys.exit(7)"],
            log_path=log_path,
        )

    assert log_path.exists()
    assert "sys.exit(7)" in log_path.read_text(encoding="utf-8")


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
