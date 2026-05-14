from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_real_data_smoke_example_has_cli_help() -> None:
    script = Path("examples/real_data_smoke.py")

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--workspace-root" in result.stdout
    assert "--quant-dataset-root" in result.stdout


def test_framework_v1_benchmark_example_has_cli_help() -> None:
    script = Path("examples/run_framework_v1_benchmark.py")

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--output-dir" in result.stdout
    assert "--dry-run" in result.stdout


def test_framework_v1_acceptance_analysis_example_has_cli_help() -> None:
    script = Path("examples/analyze_framework_v1_acceptance.py")

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--benchmark-summary" in result.stdout
    assert "--enforce-candidates" in result.stdout


def test_candidate_factor_portfolios_example_has_cli_help() -> None:
    script = Path("examples/run_candidate_factor_portfolios.py")

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--admission-report" in result.stdout
    assert "--run-backtests" in result.stdout


def test_candidate_policy_validation_example_has_cli_help() -> None:
    script = Path("examples/run_candidate_policy_validation.py")

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--profile" in result.stdout
    assert "--policy" in result.stdout


def test_candidate_policy_regime_analysis_example_has_cli_help() -> None:
    script = Path("examples/analyze_candidate_policy_regime.py")

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--validation-dir" in result.stdout
    assert "--scenario" in result.stdout
