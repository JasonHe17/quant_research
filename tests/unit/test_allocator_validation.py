from __future__ import annotations

import argparse
from pathlib import Path

from examples.run_allocator_validation import (
    _policy_validation_args_from_allocator,
    run_allocator_validation,
)
from quant_research.portfolio import load_allocator_registry


def test_allocator_validation_dry_run_generates_registered_commands(tmp_path: Path) -> None:
    args = argparse.Namespace(
        registry="configs/allocators/candidate_allocator_registry.json",
        factor_registry="configs/factors/factor_registry.json",
        allocator_id="event_limit_diffusion_complementary_health_shrink_48b",
        output_dir=str(tmp_path),
        profile="registered",
        scenario_workers=1,
        scenario_memory_budget_gb=0.0,
        backtest_workers=2,
        backtest_memory_budget_gb=12.0,
        resume_existing=True,
        dry_run=True,
        enforce_gates=False,
    )

    summary = run_allocator_validation(args)

    command = summary["commands"]["full_base"]
    assert summary["status"] == "dry_run"
    assert summary["allocator"]["profile"] == "robust"
    assert summary["allocator"]["validation_status"] == "pass"
    assert command[command.index("--dataset-dir") + 1] == (
        "runs/factor_research/event_shock_proxy_2026_05_24/joined_selection_alpha_dataset"
    )
    assert "--profile" not in command
    assert set(summary["commands"]) == {
        "full_base",
        "full_high_cost",
        "full_zero_cost",
        "year_2023_base",
        "year_2024_base",
        "year_2025_base",
    }
    assert command[command.index("--policy-gross-exposure-scale-path") + 1] == (
        "runs/candidate_factor_portfolios/"
        "event_limit_diffusion_2026_05_24_event_state_block_limit_standard/"
        "event_state_exposure_gate/gross_exposure_schedule.csv"
    )
    assert command[command.index("--factor-health-mode") + 1] == "shrink"
    assert command[command.index("--policy-set-rebalance-every-n-bars") + 1] == "48"
    assert "--resume-existing" in command
    assert (tmp_path / "commands.json").exists()
    assert (tmp_path / "allocator_validation_summary.json").exists()


def test_allocator_validation_args_use_allocator_execution_config(tmp_path: Path) -> None:
    registry = load_allocator_registry("configs/allocators/candidate_allocator_registry.json")
    allocator = registry.get("event_limit_diffusion_complementary_health_shrink_48b")

    args = _policy_validation_args_from_allocator(
        allocator.to_dict(),
        output_dir=tmp_path,
        profile="standard",
        dry_run=True,
        enforce_gates=True,
        resume_existing=True,
        scenario_workers=1,
        scenario_memory_budget_gb=0.0,
        backtest_workers=3,
        backtest_memory_budget_gb=18.0,
    )

    assert args.profile == "standard"
    assert args.methods == ["decorrelated"]
    assert args.include_features == [
        "intraday_sell_pressure_absorption_5m_w48",
        "intraday_volatility_5m_w6",
        "intraday_event_limit_diffusion_resilience_5m_w48",
        "intraday_efficiency_ratio_5m_w48",
        "intraday_amihud_5m",
    ]
    assert args.factor_health_mode == "shrink"
    assert args.policy_set_rebalance_every_n_bars == 48
    assert args.policy_set_partial_rebalance_rate == 0.5
    assert args.policy_gross_exposure_scale_path.endswith(
        "event_state_exposure_gate/gross_exposure_schedule.csv"
    )
    assert args.cost_stress_multiplier == 1.5
    assert args.backtest_workers == 3
    assert args.backtest_memory_budget_gb == 18.0
    assert args.dry_run is True
    assert args.enforce_gates is True
