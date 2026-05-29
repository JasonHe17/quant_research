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
    assert "--factor-health-mode" in result.stdout
    assert "--factor-max-weight" in result.stdout
    assert "--factor-max-contribution-share" in result.stdout
    assert "--forecast-calibration-mode" in result.stdout
    assert "--include-features" in result.stdout


def test_ml_factor_challenger_example_has_cli_help() -> None:
    script = Path("examples/run_ml_factor_challenger.py")

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--dataset-dir" in result.stdout
    assert "--admission-report" in result.stdout
    assert "--fold" in result.stdout
    assert "--redundancy-sample-rows" in result.stdout


def test_primary_pool_score_blends_example_has_cli_help() -> None:
    script = Path("examples/build_primary_pool_score_blends.py")

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--primary-score-dir" in result.stdout
    assert "--ml-pool-score-dir" in result.stdout
    assert "--primary-blend-weights" in result.stdout


def test_ml_challenger_attribution_example_has_cli_help() -> None:
    script = Path("examples/analyze_ml_challenger_attribution.py")

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--baseline-backtest-dir" in result.stdout
    assert "--challenger-backtest-dir" in result.stdout
    assert "--baseline-score-dir" in result.stdout
    assert "--challenger-score-dir" in result.stdout


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
    assert "--backtest-policy-set" in result.stdout
    assert "--forecast-calibration-mode" in result.stdout
    assert "--factor-risk-gate-feature" in result.stdout
    assert "--scenario-workers" in result.stdout
    assert "--include-features" in result.stdout


def test_factor_registry_validation_example_has_cli_help() -> None:
    script = Path("examples/validate_factor_registry.py")

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--registry" in result.stdout
    assert "--enforce-clean" in result.stdout


def test_allocator_registry_validation_example_has_cli_help() -> None:
    script = Path("examples/validate_allocator_registry.py")

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--registry" in result.stdout
    assert "--factor-registry" in result.stdout
    assert "--enforce-clean" in result.stdout


def test_allocator_validation_example_has_cli_help() -> None:
    script = Path("examples/run_allocator_validation.py")

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--allocator-id" in result.stdout
    assert "--profile" in result.stdout
    assert "--dry-run" in result.stdout


def test_allocator_monitoring_report_example_has_cli_help() -> None:
    script = Path("examples/generate_allocator_monitoring_report.py")

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--allocator-id" in result.stdout
    assert "--enforce-no-failures" in result.stdout
    assert "--append-history" in result.stdout


def test_allocator_daily_monitoring_example_has_cli_help() -> None:
    script = Path("examples/run_allocator_daily_monitoring.py")

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--allocator-id" in result.stdout
    assert "--output-root" in result.stdout
    assert "--enforce-no-sustained-warnings" in result.stdout


def test_factor_candidate_review_example_has_cli_help() -> None:
    script = Path("examples/run_factor_candidate_review.py")

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--factor-id" in result.stdout
    assert "--admission-report" in result.stdout


def test_factor_research_memory_check_example_has_cli_help() -> None:
    script = Path("examples/check_factor_research_memory.py")

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--factor-id" in result.stdout
    assert "--family" in result.stdout
    assert "--enforce-no-blocking" in result.stdout


def test_factor_failure_atlas_example_has_cli_help() -> None:
    script = Path("examples/build_factor_failure_atlas.py")

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--registry" in result.stdout
    assert "--output-dir" in result.stdout


def test_factor_opportunity_map_example_has_cli_help() -> None:
    script = Path("examples/build_factor_opportunity_map.py")

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--registry" in result.stdout
    assert "--min-positive-years" in result.stdout


def test_candidate_alpha_queue_review_example_has_cli_help() -> None:
    script = Path("examples/review_candidate_alpha_queue.py")

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--opportunity-map" in result.stdout
    assert "--validation-output-root" in result.stdout


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
    assert "--include-features" in result.stdout


def test_event_state_regime_analysis_example_has_cli_help() -> None:
    script = Path("examples/analyze_event_state_regime.py")

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--dataset-dir" in result.stdout
    assert "--validation-dir" in result.stdout
    assert "--event-feature-columns" in result.stdout
    assert "--max-z-score" in result.stdout


def test_event_state_exposure_schedule_builder_example_has_cli_help() -> None:
    script = Path("examples/build_event_state_exposure_schedule.py")

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--event-states-path" in result.stdout
    assert "--lag-windows" in result.stdout
    assert "--blocked-states" in result.stdout


def test_joined_alpha_dataset_builder_example_has_cli_help() -> None:
    script = Path("examples/build_joined_alpha_dataset.py")

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--base-dataset-dir" in result.stdout
    assert "--source" in result.stdout
    assert "--overwrite" in result.stdout


def test_joined_selection_residual_risk_analysis_example_has_cli_help() -> None:
    script = Path("examples/analyze_joined_selection_residual_risk.py")

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--validation-dir" in result.stdout
    assert "--event-state-summary" in result.stdout
    assert "--exposure-schedule" in result.stdout


def test_policy_regime_gate_builder_example_has_cli_help() -> None:
    script = Path("examples/build_policy_regime_gate.py")

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--scores-path" in result.stdout
    assert "--lookback-windows" in result.stdout
    assert "--label-lag-windows" in result.stdout
    assert "--state-confirmation-windows" in result.stdout
    assert "--gate-mode" in result.stdout
    assert "--budget-min-scale" in result.stdout
    assert "--scale-change-deadband" in result.stdout


def test_factor_risk_gate_builder_example_has_cli_help() -> None:
    script = Path("examples/build_factor_risk_gate.py")

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--feature" in result.stdout
    assert "--high-quantile" in result.stdout
    assert "--base-schedule" in result.stdout


def test_policy_backtest_comparison_example_has_cli_help() -> None:
    script = Path("examples/compare_policy_backtests.py")

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--baseline-dir" in result.stdout
    assert "--candidate-dir" in result.stdout
