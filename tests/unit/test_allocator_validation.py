from __future__ import annotations

import argparse
import json
from pathlib import Path

from examples.run_allocator_validation import (
    _policy_validation_args_from_allocator,
    _score_overlay_validation_args_from_allocator,
    run_allocator_validation,
)
from quant_research.portfolio import load_allocator_registry


def test_allocator_validation_dry_run_generates_registered_commands(
    tmp_path: Path,
) -> None:
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


def test_allocator_validation_args_use_allocator_execution_config(
    tmp_path: Path,
) -> None:
    registry = load_allocator_registry(
        "configs/allocators/candidate_allocator_registry.json"
    )
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


def test_score_overlay_allocator_validation_dry_run(tmp_path: Path) -> None:
    primary = tmp_path / "primary"
    satellite = tmp_path / "satellite"
    primary.mkdir()
    satellite.mkdir()
    (primary / "score_2023_01.parquet").touch()
    (satellite / "score_2023_01.parquet").touch()
    schedule = tmp_path / "condition.csv"
    schedule.write_text("timestamp,risk_state\n2023-01-03T15:00:00+08:00,reduced\n")
    registry_path = tmp_path / "allocator_registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "registry_name": "test_allocator_registry",
                "version": 1,
                "allocators": [
                    _overlay_allocator_payload(
                        primary_score_dir=str(primary),
                        satellite_score_dir=str(satellite),
                        condition_schedule=str(schedule),
                    )
                ],
            }
        )
    )
    args = argparse.Namespace(
        registry=str(registry_path),
        factor_registry="configs/factors/factor_registry.json",
        allocator_id="lottery_overlay_test_allocator",
        output_dir=str(tmp_path / "validation"),
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

    assert summary["status"] == "dry_run"
    assert summary["allocator"]["profile"] == "robust"
    assert summary["methods"]["lottery_overlay_test_w10"]["overlay_mode"] == "blend"
    command = summary["commands"]["lottery_overlay_test_w10:full_base"]
    assert command[command.index("--predictions-path") + 1].endswith(
        "scores/lottery_overlay_test_w10/score_*.parquet"
    )
    assert summary["params"]["condition"]["active_values"] == ["reduced", "blocked"]
    assert (tmp_path / "validation" / "commands.json").exists()
    assert (tmp_path / "validation" / "allocator_validation_summary.json").exists()


def test_score_overlay_validation_args_use_allocator_config(tmp_path: Path) -> None:
    allocator = _overlay_allocator_payload(
        primary_score_dir=str(tmp_path / "primary"),
        satellite_score_dir=str(tmp_path / "satellite"),
        condition_schedule=str(tmp_path / "condition.csv"),
    )

    args = _score_overlay_validation_args_from_allocator(
        allocator,
        output_dir=tmp_path / "validation",
        profile="standard",
        dry_run=True,
        resume_existing=True,
        job_workers=3,
    )

    assert args.primary_score_dir.endswith("primary")
    assert args.satellite_score_dir.endswith("satellite")
    assert args.method_prefix == "lottery_overlay_test"
    assert args.overlay_weights == [0.1]
    assert args.condition_values == ["reduced", "blocked"]
    assert args.trade_policy == "cost_aware_optimizer"
    assert args.policy_gross_exposure_scale == 0.4
    assert args.policy_max_gross_turnover_per_rebalance == 0.005
    assert args.policy_total_gross_turnover_budget == 155.0
    assert args.policy_cost_pressure_reduced_scale == 1.0
    assert args.optimizer_target_cap_mode == "clip"
    assert args.job_workers == 3
    assert args.dry_run is True


def test_score_overlay_validation_args_include_score_postprocessor(
    tmp_path: Path,
) -> None:
    allocator = _overlay_allocator_payload(
        primary_score_dir=str(tmp_path / "primary"),
        satellite_score_dir=str(tmp_path / "satellite"),
        condition_schedule=str(tmp_path / "condition.csv"),
    )
    allocator["score"]["postprocessors"] = [
        {
            "type": "optimizer_risk_penalty_join",
            "penalty_dir": str(tmp_path / "penalty"),
            "penalty_column": "liq_penalty_bps",
            "fill_value": 0.25,
        }
    ]

    args = _score_overlay_validation_args_from_allocator(
        allocator,
        output_dir=tmp_path / "validation",
        profile="standard",
        dry_run=True,
        resume_existing=True,
        job_workers=3,
    )

    assert args.optimizer_risk_penalty_dir.endswith("penalty")
    assert args.optimizer_risk_penalty_column == "liq_penalty_bps"
    assert args.optimizer_risk_penalty_fill_value == 0.25


def test_score_overlay_validation_args_include_target_weight_cap_postprocessor(
    tmp_path: Path,
) -> None:
    allocator = _overlay_allocator_payload(
        primary_score_dir=str(tmp_path / "primary"),
        satellite_score_dir=str(tmp_path / "satellite"),
        condition_schedule=str(tmp_path / "condition.csv"),
    )
    allocator["score"]["postprocessors"] = [
        {
            "type": "target_weight_cap_join",
            "cap_dir": str(tmp_path / "cap"),
            "cap_column": "liq_cap",
            "fill_value": 1.0,
        }
    ]

    args = _score_overlay_validation_args_from_allocator(
        allocator,
        output_dir=tmp_path / "validation",
        profile="standard",
        dry_run=True,
        resume_existing=True,
        job_workers=3,
    )

    assert args.target_weight_cap_dir.endswith("cap")
    assert args.target_weight_cap_column == "liq_cap"
    assert args.target_weight_cap_fill_value == 1.0


def test_score_overlay_validation_args_include_optimizer_target_cap_mode(
    tmp_path: Path,
) -> None:
    allocator = _overlay_allocator_payload(
        primary_score_dir=str(tmp_path / "primary"),
        satellite_score_dir=str(tmp_path / "satellite"),
        condition_schedule=str(tmp_path / "condition.csv"),
    )
    allocator["execution_policy"]["optimizer_target_cap_mode"] = "replace"

    args = _score_overlay_validation_args_from_allocator(
        allocator,
        output_dir=tmp_path / "validation",
        profile="standard",
        dry_run=True,
        resume_existing=True,
        job_workers=3,
    )

    assert args.optimizer_target_cap_mode == "replace"


def _overlay_allocator_payload(
    *,
    primary_score_dir: str,
    satellite_score_dir: str,
    condition_schedule: str,
) -> dict[str, object]:
    return {
        "allocator_id": "lottery_overlay_test_allocator",
        "display_name": "Lottery overlay test allocator",
        "status": "candidate",
        "owner": "quant_research",
        "description": "A test score-overlay allocator.",
        "hypothesis": "A score-level overlay can be governed from the allocator registry.",
        "score": {
            "construction": "score_overlay",
            "primary_score_dir": primary_score_dir,
            "satellite_score_dir": satellite_score_dir,
            "method_prefix": "lottery_overlay_test",
            "overlay_weights": [0.1],
            "overlay_mode": "blend",
            "rank_normalize": True,
            "condition": {
                "schedule_path": condition_schedule,
                "column": "risk_state",
                "values": ["reduced", "blocked"],
            },
        },
        "risk_controls": {},
        "execution_policy": {
            "policy": "cost_aware_optimizer_budget155_cost_pressure_cap0010_turnover005_gross040_daily",
            "trade_policy": "cost_aware_optimizer",
            "top_n": 50,
            "entry_rank": 50,
            "exit_rank": 150,
            "max_entries_per_rebalance": 10,
            "max_exits_per_rebalance": 10,
            "rebalance_every_n_bars": 48,
            "partial_rebalance_rate": 0.5,
            "min_hold_bars": 0,
            "no_trade_weight_band": 0.002,
            "estimated_cost_bps": 13.0,
            "gross_exposure_scale": 0.4,
            "max_gross_turnover_per_rebalance": 0.005,
            "total_gross_turnover_budget": 155.0,
            "turnover_budget_period": "path",
            "turnover_budget_pacing": 0.0,
            "cost_pressure_threshold_bps": 1000.0,
            "cost_pressure_reduced_scale": 1.0,
            "cost_pressure_max_gross_turnover_per_rebalance": 0.01,
            "optimizer_weighting": "utility",
            "optimizer_target_cap_mode": "clip",
            "optimizer_score_to_edge_bps": 100.0,
            "optimizer_min_net_edge_bps": 0.0,
            "optimizer_risk_penalty_multiplier": 1.0,
            "min_trade_weight": 0.0005,
            "exclude_st": True,
            "lot_size": 100,
            "limit_up_bps": 980.0,
            "limit_down_bps": 980.0,
        },
        "cost_model": {
            "commission_bps": 3.0,
            "slippage_bps": 1.0,
            "sell_stamp_tax_bps": 5.0,
            "min_commission": 5.0,
            "high_cost_multiplier": 2.0,
        },
        "data": {
            "partition_start": "2023_01",
            "partition_end": "2023_12",
            "data_access_mode": "fast_parquet",
            "streaming_chunk": "month",
            "streaming_chunk_padding_days": 10,
            "catalog_path": "../quant_dataset/canonical_store/catalog/quant_research.duckdb",
        },
        "validation": {
            "status": "pass",
            "standard_validation": "pyproject.toml",
            "robust_validation": "pyproject.toml",
        },
        "governance": {"decision": "candidate_allocator"},
        "references": ["pyproject.toml"],
        "tags": ["score_overlay"],
    }
