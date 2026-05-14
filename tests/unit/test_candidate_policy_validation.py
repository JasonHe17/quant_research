from __future__ import annotations

from pathlib import Path

from examples.run_candidate_policy_validation import (
    _infer_full_years,
    _scenario_command,
    _validation_checks,
    _validation_scenarios,
)


def test_candidate_policy_validation_builds_standard_scenarios() -> None:
    args = _validation_args(profile="standard")

    scenarios = _validation_scenarios(args, years=[2023, 2024])

    assert [scenario.name for scenario in scenarios] == [
        "full_base",
        "year_2023_base",
        "year_2024_base",
        "full_high_cost",
    ]
    assert scenarios[0].partition_start == "2023_01"
    assert scenarios[0].partition_end == "2024_12"
    assert scenarios[-1].commission_bps == 6.0
    assert scenarios[-1].sell_stamp_tax_bps == 10.0


def test_candidate_policy_validation_command_uses_selected_policy(tmp_path: Path) -> None:
    args = _validation_args(
        output_dir=str(tmp_path),
        methods=["decorrelated", "equal"],
        backtest_memory_budget_gb=12.0,
    )
    scenario = _validation_scenarios(args, years=[2023])[0]

    command = _scenario_command(args, scenario)

    assert command[command.index("--backtest-policies") + 1] == "partial_rebalance_daily"
    assert command[command.index("--methods") + 1 : command.index("--partition-start")] == [
        "decorrelated",
        "equal",
    ]
    assert command[command.index("--partition-start") + 1] == "2023_01"
    assert command[command.index("--backtest-memory-budget-gb") + 1] == "12.0"
    assert "--resume-existing" in command


def test_candidate_policy_validation_infers_full_years(tmp_path: Path) -> None:
    for month in range(1, 13):
        (tmp_path / f"dataset_2023_{month:02d}.parquet").touch()
    for month in range(1, 12):
        (tmp_path / f"dataset_2024_{month:02d}.parquet").touch()

    assert _infer_full_years(tmp_path) == [2023]


def test_candidate_policy_validation_checks_primary_policy() -> None:
    args = _validation_args()
    rows = [
        {
            "scenario": "full_base",
            "method": "decorrelated",
            "policy": "partial_rebalance_daily",
            "total_return": 0.1,
            "max_drawdown": -0.1,
            "gross_turnover": 120.0,
            "trade_count": 10,
            "total_transaction_cost": 100.0,
            "final_equity": 1_100_000.0,
        },
        {
            "scenario": "full_high_cost",
            "method": "decorrelated",
            "policy": "partial_rebalance_daily",
            "total_return": 0.05,
            "max_drawdown": -0.2,
            "gross_turnover": 118.0,
            "trade_count": 10,
            "total_transaction_cost": 200.0,
            "final_equity": 1_050_000.0,
        },
        {
            "scenario": "year_2023_base",
            "method": "decorrelated",
            "policy": "partial_rebalance_daily",
            "total_return": 0.03,
            "max_drawdown": -0.1,
            "gross_turnover": 40.0,
            "trade_count": 10,
            "total_transaction_cost": 50.0,
            "final_equity": 1_030_000.0,
        },
    ]

    checks = _validation_checks(args, rows)

    assert checks["overall_status"] == "pass"


def _validation_args(**overrides: object) -> object:
    defaults = {
        "dataset_dir": "dataset",
        "admission_report": "admission.json",
        "factor_correlation": "correlation.csv",
        "output_dir": "runs/validation",
        "profile": "standard",
        "years": None,
        "methods": ["decorrelated", "equal", "ic_weighted"],
        "primary_method": "decorrelated",
        "policy": "partial_rebalance_daily",
        "top_n": 50,
        "initial_cash": 1_000_000.0,
        "commission_bps": 3.0,
        "slippage_bps": 1.0,
        "sell_stamp_tax_bps": 5.0,
        "min_commission": 5.0,
        "cost_stress_multiplier": 2.0,
        "lot_size": 100,
        "min_trade_weight": 0.0005,
        "exclude_st": True,
        "limit_up_bps": 980.0,
        "limit_down_bps": 980.0,
        "policy_no_trade_weight_band": 0.002,
        "policy_set_drop_count": 10,
        "policy_set_exit_rank": 150,
        "policy_set_rebalance_every_n_bars": 48,
        "policy_set_partial_rebalance_rate": 0.5,
        "data_access_mode": "fast_parquet",
        "streaming_chunk": "month",
        "streaming_chunk_padding_days": 10,
        "backtest_workers": 2,
        "backtest_memory_budget_gb": 12.0,
        "full_backtest_memory_gb": 5.0,
        "yearly_backtest_memory_gb": 5.0,
        "max_full_turnover": 160.0,
        "resume_existing": True,
        "dry_run": False,
        "enforce_gates": False,
    }
    defaults.update(overrides)
    return type("Args", (), defaults)()
