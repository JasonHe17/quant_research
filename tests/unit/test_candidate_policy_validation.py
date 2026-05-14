from __future__ import annotations

from pathlib import Path

import pytest

from examples.run_candidate_policy_validation import (
    _collect_factor_contribution_summary_rows,
    _collect_factor_health_summary_rows,
    _infer_full_years,
    _monthly_summary_rows_for_backtest,
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
        factor_max_weight=0.4,
        factor_max_contribution_share=0.5,
        factor_health_mode="shrink",
        factor_health_lookback_windows=10,
    )
    scenario = _validation_scenarios(args, years=[2023])[0]

    command = _scenario_command(args, scenario)

    assert command[command.index("--backtest-policies") + 1] == "partial_rebalance_daily"
    assert command[command.index("--methods") + 1 : command.index("--partition-start")] == [
        "decorrelated",
        "equal",
    ]
    assert command[command.index("--partition-start") + 1] == "2023_01"
    assert command[command.index("--policy-gross-exposure-scale") + 1] == "1.0"
    assert command[command.index("--backtest-memory-budget-gb") + 1] == "12.0"
    assert command[command.index("--factor-max-weight") + 1] == "0.4"
    assert command[command.index("--factor-max-contribution-share") + 1] == "0.5"
    assert command[command.index("--factor-health-mode") + 1] == "shrink"
    assert command[command.index("--factor-health-lookback-windows") + 1] == "10"
    assert command[command.index("--score-diagnostics-top-n") + 1] == "50"
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


def test_candidate_policy_validation_builds_monthly_summary(tmp_path: Path) -> None:
    backtest_dir = tmp_path / "backtest"
    backtest_dir.mkdir()
    (backtest_dir / "equity_curve.csv").write_text(
        "\n".join(
            [
                "timestamp,cash,positions_value,equity",
                "2023-12-29T15:00:00+08:00,1000000,0,1000000",
                "2024-01-02T09:35:00+08:00,900000,100000,1000000",
                "2024-01-31T15:00:00+08:00,900000,120000,1020000",
                "2024-02-01T09:35:00+08:00,900000,90000,990000",
                "2024-02-29T15:00:00+08:00,900000,80000,980000",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (backtest_dir / "trades.csv").write_text(
        "\n".join(
            [
                "timestamp,instrument_id,side,shares,price,reference_price,commission,stamp_tax,slippage_cost,total_cost,notional,reference_notional",
                "2024-01-02T09:40:00+08:00,a,buy,100,10,10,5,0,1,6,1000,1000",
                "2024-02-01T09:40:00+08:00,a,sell,100,10,10,5,1,1,7,1000,1000",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    scenario = _validation_scenarios(_validation_args(), years=[2024])[0]

    rows = _monthly_summary_rows_for_backtest(
        backtest_dir,
        scenario=scenario,
        method="decorrelated",
        policy="partial_rebalance_daily",
        initial_cash=1_000_000.0,
    )

    assert [row["month"] for row in rows] == ["2024-01", "2024-02"]
    assert rows[0]["return"] == pytest.approx(0.02)
    assert rows[0]["trade_count"] == 1
    assert rows[1]["total_transaction_cost"] == 7.0


def test_candidate_policy_validation_collects_factor_summaries(tmp_path: Path) -> None:
    args = _validation_args(output_dir=str(tmp_path))
    scenario = _validation_scenarios(args, years=[2024])[0]
    scenario_dir = tmp_path / scenario.name
    scenario_dir.mkdir(parents=True)
    health_path = scenario_dir / "factor_health.csv"
    health_path.write_text(
        "\n".join(
            [
                "timestamp,feature,weight_scale,shrink_reason",
                "t0,alpha_a,1.0,warmup",
                "t1,alpha_a,0.5,lagged_health_shrink",
                "t0,alpha_b,0.75,healthy",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    diagnostics_dir = scenario_dir / "scores" / "decorrelated" / "diagnostics"
    diagnostics_dir.mkdir(parents=True)
    diagnostics_path = diagnostics_dir / "factor_contribution_2024_01.csv"
    diagnostics_path.write_text(
        "\n".join(
            [
                "timestamp,largest_abs_contribution_share,top_two_abs_contribution_share",
                "t0,0.6,0.9",
                "t1,0.4,0.8",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (scenario_dir / "summary.json").write_text(
        (
            "{"
            f"\"factor_health_schedule\": \"{health_path}\","
            "\"methods\": {"
            "\"decorrelated\": {"
            "\"factor_contribution_diagnostics\": ["
            f"\"{diagnostics_path}\""
            "]"
            "}"
            "}"
            "}"
        ),
        encoding="utf-8",
    )

    health_rows = _collect_factor_health_summary_rows(args, [scenario])
    contribution_rows = _collect_factor_contribution_summary_rows(args, [scenario])

    assert health_rows[0]["feature"] == "alpha_a"
    assert health_rows[0]["lagged_health_shrink_count"] == 1
    assert contribution_rows[0]["average_largest_abs_contribution_share"] == pytest.approx(
        0.5
    )


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
        "score_diagnostics_top_n": 50,
        "factor_max_weight": None,
        "factor_max_contribution_share": None,
        "factor_health_mode": "off",
        "factor_health_lookback_windows": 20,
        "factor_health_min_periods": 5,
        "factor_health_label_lag_windows": 48,
        "factor_health_min_scale": 0.25,
        "factor_health_max_scale": 1.0,
        "factor_health_rank_ic_floor": -0.05,
        "factor_health_rank_ic_ceiling": 0.05,
        "factor_health_spread_floor": -0.001,
        "factor_health_spread_ceiling": 0.001,
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
        "policy_gross_exposure_scale": 1.0,
        "policy_gross_exposure_scale_path": None,
        "optimizer_candidate_rank": None,
        "optimizer_score_to_edge_bps": 100.0,
        "optimizer_min_net_edge_bps": 0.0,
        "optimizer_risk_penalty_multiplier": 1.0,
        "optimizer_weighting": "utility",
        "optimizer_max_name_weight": None,
        "optimizer_max_gross_exposure_increase_per_rebalance": None,
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
