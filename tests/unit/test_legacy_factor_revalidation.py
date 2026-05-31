from __future__ import annotations

import json
from pathlib import Path

from examples.run_legacy_factor_revalidation import (
    _admission_eligible_jobs,
    _collect_factor_results,
    _factor_jobs,
    _primary_label_column,
    _recommended_action,
    _selected_factors,
    _shared_benchmark_command,
    run_legacy_factor_revalidation,
)


def test_legacy_factor_revalidation_dry_run_plans_shared_and_factor_jobs(
    tmp_path: Path,
) -> None:
    registry_path = _registry_path(tmp_path)
    output_dir = tmp_path / "revalidation"
    args = _args(
        registry=str(registry_path),
        output_dir=str(output_dir),
        dry_run=True,
    )

    summary = run_legacy_factor_revalidation(args)

    assert summary["status"] == "dry_run"
    assert summary["factor_count"] == 2
    commands = json.loads((output_dir / "commands.json").read_text())
    assert "run_framework_v1_benchmark.py" in commands["shared_benchmark"][1]
    assert "--auto-factor-admission" in commands["shared_benchmark"]
    assert commands["shared_benchmark"][
        commands["shared_benchmark"].index("--factor-registry") + 1
    ] == str(registry_path)
    assert set(commands["factor_revalidations"]) == {"alpha_a", "alpha_b"}
    alpha_command = commands["factor_revalidations"]["alpha_a"]
    assert "run_candidate_policy_validation.py" in alpha_command[1]
    assert alpha_command[
        alpha_command.index("--registry-statuses") + 1 : alpha_command.index(
            "--admission-statuses"
        )
    ] == ["candidate", "promoted"]
    assert alpha_command[
        alpha_command.index("--admission-statuses") + 1 : alpha_command.index(
            "--evaluation-roles"
        )
    ] == ["candidate", "watchlist"]
    assert alpha_command[
        alpha_command.index("--evaluation-roles") + 1 : alpha_command.index(
            "--output-dir"
        )
    ] == ["alpha_rank"]
    assert alpha_command[alpha_command.index("--score-transform") + 1] == "rank"
    assert alpha_command[
        alpha_command.index("--factor-health-mode") + 1 : alpha_command.index(
            "--include-features"
        )
    ] == ["monitor"]
    assert alpha_command[
        alpha_command.index("--include-features") + 1 : alpha_command.index("--top-n")
    ] == ["alpha_a_feature"]
    assert alpha_command[
        alpha_command.index("--backtest-policies") + 1 : alpha_command.index("--resume-existing")
    ] == ["partial_rebalance_daily", "cost_aware_optimizer_daily"]
    assert alpha_command[
        alpha_command.index("--years") + 1 : alpha_command.index("--backtest-policies")
    ] == ["2024"]
    assert "--backtest-memory-estimate-gb" not in alpha_command
    assert alpha_command[alpha_command.index("--full-backtest-memory-gb") + 1] == "5.0"
    assert alpha_command[alpha_command.index("--yearly-backtest-memory-gb") + 1] == "5.0"


def test_legacy_factor_revalidation_filters_status_and_factor_ids(tmp_path: Path) -> None:
    registry_path = _registry_path(tmp_path)
    args = _args(
        registry=str(registry_path),
        statuses=["promoted"],
        factor_ids=["alpha_a", "alpha_b"],
    )

    factors = _selected_factors(args)

    assert [factor.factor_id for factor in factors] == ["alpha_b"]


def test_legacy_factor_revalidation_filters_evaluation_roles(tmp_path: Path) -> None:
    registry_path = _registry_path(tmp_path)
    args = _args(
        registry=str(registry_path),
        statuses=["candidate", "promoted"],
        evaluation_roles=["risk_penalty"],
    )

    factors = _selected_factors(args)

    assert [factor.factor_id for factor in factors] == ["risk_a"]


def test_legacy_factor_revalidation_primary_label_uses_first_horizon() -> None:
    assert _primary_label_column(_args(label_horizon_bars=[48])) == "forward_return"
    assert _primary_label_column(_args(label_horizon_bars=[240, 960])) == (
        "forward_return_240b"
    )


def test_legacy_factor_revalidation_shared_command_carries_parallel_controls(
    tmp_path: Path,
) -> None:
    args = _args(
        registry=str(_registry_path(tmp_path)),
        output_dir=str(tmp_path / "out"),
        dataset_workers=3,
        evaluation_workers=4,
        shared_backtest_workers=5,
    )

    command = _shared_benchmark_command(args)

    assert command[command.index("--dataset-workers") + 1] == "3"
    assert command[command.index("--evaluation-workers") + 1] == "4"
    assert command[command.index("--backtest-workers") + 1] == "5"


def test_legacy_factor_revalidation_factor_jobs_use_memory_estimate(
    tmp_path: Path,
) -> None:
    args = _args(
        registry=str(_registry_path(tmp_path)),
        output_dir=str(tmp_path / "out"),
        factor_job_memory_gb=7.5,
    )

    jobs = _factor_jobs(args, _selected_factors(args))

    assert jobs[0].memory_estimate_gb == 7.5
    assert jobs[0].summary_path.name == "validation_summary.json"


def test_legacy_factor_revalidation_skips_admission_filtered_jobs(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "out"
    args = _args(
        registry=str(_registry_path(tmp_path)),
        output_dir=str(output_dir),
        statuses=["candidate", "promoted"],
        admission_statuses=["candidate", "watchlist"],
    )
    factors = _selected_factors(args)
    jobs = _factor_jobs(args, factors)
    admission_dir = output_dir / "shared_benchmark" / "factor_admission"
    admission_dir.mkdir(parents=True)
    (admission_dir / "factor_admission_report.json").write_text(
        json.dumps(
            {
                "factors": [
                    {"feature": "alpha_a_feature", "admission_status": "reject"},
                    {"feature": "alpha_b_feature", "admission_status": "candidate"},
                ]
            }
        ),
        encoding="utf-8",
    )

    eligible = _admission_eligible_jobs(args, factors, jobs)
    results = _collect_factor_results(args, factors, jobs)

    assert [job.factor_id for job in eligible] == ["alpha_b"]
    alpha_a = next(row for row in results if row["factor_id"] == "alpha_a")
    assert alpha_a["new_admission_status"] == "reject"
    assert alpha_a["validation_status"] == "admission_filtered"


def test_legacy_factor_revalidation_recommendation_flags_weak_warn_candidate() -> None:
    action = _recommended_action(
        legacy_status="candidate",
        new_status="candidate",
        best_policy={
            "full_base_return": 0.007,
            "full_high_cost_return": 0.001,
        },
        validation={"overall_status": "warn"},
    )

    assert action == "weak_or_unstable_candidate_review"


def test_legacy_factor_revalidation_recommendation_confirms_strong_warn_candidate() -> None:
    action = _recommended_action(
        legacy_status="candidate",
        new_status="candidate",
        best_policy={
            "full_base_return": 0.20,
            "full_high_cost_return": 0.19,
        },
        validation={"overall_status": "warn"},
    )

    assert action == "confirmed_with_stability_warning"


def _registry_path(tmp_path: Path) -> Path:
    path = tmp_path / "factor_registry.json"
    payload = {
        "registry_name": "test",
        "version": 1,
        "entries": [
            _entry("alpha_a", "alpha_a_feature", "candidate"),
            _entry("alpha_b", "alpha_b_feature", "promoted"),
            _entry(
                "risk_a",
                "risk_a_feature",
                "candidate",
                evaluation_role="risk_penalty",
            ),
            _entry("alpha_old", "alpha_old_feature", "deprecated"),
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _entry(
    factor_id: str,
    feature: str,
    status: str,
    *,
    evaluation_role: str = "alpha_rank",
) -> dict[str, object]:
    return {
        "factor_id": factor_id,
        "display_name": factor_id,
        "family": "momentum",
        "status": status,
        "evaluation_role": evaluation_role,
        "expected_direction": "long",
        "feature_columns": [feature],
        "required_inputs": ["close_price"],
        "frequency": "5m",
        "description": "test",
        "hypothesis": "test",
        "evaluation": {"admission_status": status},
        "a_share_constraints": {
            "long_only": True,
            "price_limit_aware": True,
            "st_aware": True,
            "t_plus_one_safe": True,
        },
        "point_in_time_safe": True,
        "live_available": True,
    }


def _args(**overrides: object) -> object:
    defaults = {
        "registry": "configs/factors/factor_registry.json",
        "output_dir": "runs/legacy_factor_revalidation/current",
        "statuses": ["candidate", "promoted"],
        "admission_statuses": ["candidate", "watchlist"],
        "evaluation_roles": ["alpha_rank"],
        "factor_ids": None,
        "max_factors": None,
        "profile": "quick",
        "start": "2024-01-02T09:35:00+08:00",
        "end": "2024-01-03T15:00:00+08:00",
        "catalog_path": None,
        "data_snapshot": "2026-05-09",
        "max_symbols": 2,
        "years": [2024],
        "label_horizon_bars": [48, 240],
        "methods": ["decorrelated", "equal"],
        "score_transform": "rank",
        "backtest_policies": [
            "partial_rebalance_daily",
            "cost_aware_optimizer_daily",
        ],
        "primary_policy": "partial_rebalance_daily",
        "factor_health_mode": "monitor",
        "top_n": 50,
        "commission_bps": 3.0,
        "slippage_bps": 1.0,
        "sell_stamp_tax_bps": 5.0,
        "min_commission": 5.0,
        "cost_stress_multiplier": 2.0,
        "dataset_workers": 1,
        "dataset_worker_memory_estimate_gb": 10.0,
        "evaluation_workers": 6,
        "shared_backtest_workers": 6,
        "full_backtest_memory_gb": 8.0,
        "yearly_backtest_memory_gb": 6.0,
        "factor_workers": 2,
        "factor_backtest_workers": 1,
        "factor_job_memory_gb": 5.0,
        "factor_memory_budget_gb": 0.0,
        "data_access_mode": "fast_parquet",
        "streaming_chunk": "month",
        "streaming_chunk_padding_days": 10,
        "skip_shared_benchmark": False,
        "resume_existing": True,
        "dry_run": False,
    }
    defaults.update(overrides)
    return type("Args", (), defaults)()
