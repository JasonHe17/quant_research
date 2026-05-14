from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from examples.run_candidate_factor_portfolios import (
    BacktestPolicySpec,
    _build_factor_health_schedule,
    _backtest_command,
    _backtest_jobs,
    _backtest_policy_specs,
    _backtest_summary_rows,
    _dataset_paths,
    _effective_backtest_memory_budget_gb,
    _summary_params,
)
from quant_research.portfolio import (
    CandidateFactor,
    FactorHealthConfig,
    ScoreForecastCalibrationConfig,
    build_composite_scores,
    build_factor_health_schedule,
    cap_factor_weights,
    factor_contribution_diagnostics,
    factor_combination_weights,
    load_candidate_factors,
    write_score_partitions,
)


def test_load_candidate_factors_uses_admission_direction(tmp_path: Path) -> None:
    path = tmp_path / "admission.json"
    path.write_text(
        json.dumps(
            {
                "factors": [
                    {
                        "feature": "alpha_a",
                        "admission_status": "candidate",
                        "direction": "invert",
                        "spearman_rank_ic_mean": -0.02,
                    },
                    {
                        "feature": "alpha_b",
                        "admission_status": "watchlist",
                        "direction": "long",
                        "spearman_rank_ic_mean": 0.01,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    factors = load_candidate_factors(path)

    assert factors == (CandidateFactor("alpha_a", -1, -0.02),)


def test_factor_combination_weights_support_methods() -> None:
    factors = (
        CandidateFactor("alpha_a", 1, 0.02),
        CandidateFactor("alpha_b", -1, -0.01),
    )
    correlation = pd.DataFrame(
        [[1.0, 0.9], [0.9, 1.0]],
        index=["alpha_a", "alpha_b"],
        columns=["alpha_a", "alpha_b"],
    )

    equal = factor_combination_weights(factors, method="equal")
    ic_weighted = factor_combination_weights(factors, method="ic_weighted")
    decorrelated = factor_combination_weights(
        factors,
        method="decorrelated",
        correlation=correlation,
    )

    assert equal == {"alpha_a": 0.5, "alpha_b": 0.5}
    assert ic_weighted["alpha_a"] == pytest.approx(2 / 3)
    assert sum(decorrelated.values()) == pytest.approx(1.0)


def test_cap_factor_weights_limits_concentrated_static_weights() -> None:
    weights = cap_factor_weights(
        {"alpha_a": 0.8, "alpha_b": 0.15, "alpha_c": 0.05},
        max_weight=0.6,
    )

    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights["alpha_a"] <= 0.6000001
    assert weights["alpha_b"] > 0.15


def test_build_composite_scores_ranks_and_orients_cross_sectionally() -> None:
    frame = pd.DataFrame(
        [
            {"timestamp": "t0", "instrument_id": "a", "alpha_a": 1.0, "alpha_b": 1.0},
            {"timestamp": "t0", "instrument_id": "b", "alpha_a": 2.0, "alpha_b": 0.0},
            {"timestamp": "t0", "instrument_id": "c", "alpha_a": 3.0, "alpha_b": -1.0},
        ]
    )
    factors = (
        CandidateFactor("alpha_a", 1, 0.02),
        CandidateFactor("alpha_b", -1, -0.01),
    )

    scores = build_composite_scores(
        frame,
        candidates=factors,
        weights={"alpha_a": 0.5, "alpha_b": 0.5},
    )

    assert scores.iloc[0]["instrument_id"] == "c"
    assert scores.iloc[0]["score"] > scores.iloc[-1]["score"]


def test_build_composite_scores_applies_factor_health_scales() -> None:
    frame = pd.DataFrame(
        [
            {"timestamp": "t0", "instrument_id": "a", "alpha_a": 1.0, "alpha_b": 3.0},
            {"timestamp": "t0", "instrument_id": "b", "alpha_a": 2.0, "alpha_b": 2.0},
            {"timestamp": "t0", "instrument_id": "c", "alpha_a": 3.0, "alpha_b": 1.0},
        ]
    )
    factors = (
        CandidateFactor("alpha_a", 1, 0.02),
        CandidateFactor("alpha_b", 1, 0.01),
    )
    health = pd.DataFrame(
        [
            {"timestamp": "t0", "feature": "alpha_a", "weight_scale": 1.0},
            {"timestamp": "t0", "feature": "alpha_b", "weight_scale": 0.0},
        ]
    )

    scores = build_composite_scores(
        frame,
        candidates=factors,
        weights={"alpha_a": 0.5, "alpha_b": 0.5},
        factor_health=health,
    )

    assert scores.iloc[0]["instrument_id"] == "c"
    assert scores.iloc[-1]["instrument_id"] == "a"


def test_build_composite_scores_caps_row_level_factor_contributions() -> None:
    frame = pd.DataFrame(
        [
            {"timestamp": "t0", "instrument_id": "a", "alpha_a": 1.0, "alpha_b": 1.0},
            {"timestamp": "t0", "instrument_id": "b", "alpha_a": 2.0, "alpha_b": 1.0},
            {"timestamp": "t0", "instrument_id": "c", "alpha_a": 3.0, "alpha_b": 1.0},
        ]
    )
    factors = (
        CandidateFactor("alpha_a", 1, 0.02),
        CandidateFactor("alpha_b", 1, 0.01),
    )

    uncapped = build_composite_scores(
        frame,
        candidates=factors,
        weights={"alpha_a": 0.9, "alpha_b": 0.1},
    )
    capped = build_composite_scores(
        frame,
        candidates=factors,
        weights={"alpha_a": 0.9, "alpha_b": 0.1},
        max_factor_contribution_share=0.5,
    )

    assert capped.iloc[0]["score"] < uncapped.iloc[0]["score"]


def test_build_factor_health_schedule_uses_only_lagged_labels(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset_2024_01.parquet"
    pd.DataFrame(
        [
            {
                "timestamp": f"t{index}",
                "instrument_id": instrument,
                "alpha_a": value,
                "forward_return": value if index == 0 else -value,
            }
            for index in range(4)
            for instrument, value in (("a", 1.0), ("b", 2.0), ("c", 3.0))
        ]
    ).to_parquet(dataset_path, index=False)

    schedule = build_factor_health_schedule(
        [dataset_path],
        candidates=(CandidateFactor("alpha_a", 1, 0.02),),
        config=FactorHealthConfig(
            lookback_windows=1,
            min_periods=1,
            label_lag_windows=1,
            min_scale=0.25,
            max_scale=1.0,
            rank_ic_floor=-1.0,
            rank_ic_ceiling=1.0,
            spread_floor=-1.0,
            spread_ceiling=1.0,
        ),
        top_n=1,
    )

    assert schedule.loc[0, "shrink_reason"] == "warmup"
    assert schedule.loc[1, "weight_scale"] == pytest.approx(1.0)
    assert schedule.loc[2, "weight_scale"] == pytest.approx(0.25)


def test_factor_contribution_diagnostics_reports_top_concentration() -> None:
    frame = pd.DataFrame(
        [
            {
                "timestamp": "t0",
                "instrument_id": "a",
                "alpha_a": 1.0,
                "alpha_b": 1.0,
                "forward_return": 0.01,
            },
            {
                "timestamp": "t0",
                "instrument_id": "b",
                "alpha_a": 2.0,
                "alpha_b": 1.0,
                "forward_return": 0.02,
            },
            {
                "timestamp": "t0",
                "instrument_id": "c",
                "alpha_a": 3.0,
                "alpha_b": 1.0,
                "forward_return": 0.03,
            },
        ]
    )
    factors = (
        CandidateFactor("alpha_a", 1, 0.02),
        CandidateFactor("alpha_b", 1, 0.01),
    )
    scores = build_composite_scores(
        frame,
        candidates=factors,
        weights={"alpha_a": 0.8, "alpha_b": 0.2},
    )

    diagnostics = factor_contribution_diagnostics(
        frame,
        scores=scores,
        candidates=factors,
        weights={"alpha_a": 0.8, "alpha_b": 0.2},
        factor_health=None,
        max_factor_contribution_share=0.5,
        top_n=2,
    )

    assert diagnostics.loc[0, "largest_contribution_feature"] == "alpha_a"
    assert diagnostics.loc[0, "largest_abs_contribution_share"] <= 0.5


def test_write_score_partitions_writes_one_partition_per_method(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset_2024_01.parquet"
    pd.DataFrame(
        [
            {
                "timestamp": "t0",
                "instrument_id": "a",
                "alpha_a": 1.0,
                "forward_return": 0.01,
            },
            {
                "timestamp": "t0",
                "instrument_id": "b",
                "alpha_a": 2.0,
                "forward_return": 0.02,
            },
        ]
    ).to_parquet(dataset_path, index=False)

    summary = write_score_partitions(
        [dataset_path],
        output_dir=tmp_path / "scores",
        candidates=(CandidateFactor("alpha_a", 1, 0.02),),
        weights_by_method={"equal": {"alpha_a": 1.0}},
        diagnostics_top_n=1,
    )

    assert summary["methods"]["equal"]["row_count"] == 2
    assert Path(tmp_path / "scores" / "equal" / "score_2024_01.parquet").exists()
    assert Path(
        tmp_path
        / "scores"
        / "equal"
        / "diagnostics"
        / "factor_contribution_2024_01.csv"
    ).exists()


def test_write_score_partitions_can_attach_calibrated_forecasts(tmp_path: Path) -> None:
    for partition, rows in {
        "2024_01": [
            {"timestamp": "t0", "instrument_id": "a", "alpha_a": 3.0, "forward_return": 0.02},
            {"timestamp": "t0", "instrument_id": "b", "alpha_a": 1.0, "forward_return": -0.01},
            {"timestamp": "t1", "instrument_id": "a", "alpha_a": 3.0, "forward_return": -0.03},
            {"timestamp": "t1", "instrument_id": "b", "alpha_a": 1.0, "forward_return": 0.01},
        ],
        "2024_02": [
            {"timestamp": "t2", "instrument_id": "a", "alpha_a": 3.0, "forward_return": 0.04},
            {"timestamp": "t2", "instrument_id": "b", "alpha_a": 1.0, "forward_return": -0.02},
        ],
    }.items():
        pd.DataFrame(rows).to_parquet(tmp_path / f"dataset_{partition}.parquet", index=False)

    summary = write_score_partitions(
        [
            tmp_path / "dataset_2024_01.parquet",
            tmp_path / "dataset_2024_02.parquet",
        ],
        output_dir=tmp_path / "scores",
        candidates=(CandidateFactor("alpha_a", 1, 0.02),),
        weights_by_method={"equal": {"alpha_a": 1.0}},
        forecast_calibration_config=ScoreForecastCalibrationConfig(
            lookback_windows=1,
            min_periods=1,
            label_lag_windows=1,
            bucket_count=2,
        ),
    )

    scores = pd.read_parquet(tmp_path / "scores" / "equal" / "score_2024_02.parquet")
    top = scores.sort_values("score", ascending=False).iloc[0]
    assert "expected_edge_bps" in scores.columns
    assert top["expected_edge_bps"] == pytest.approx(-300.0)
    assert summary["methods"]["equal"]["score_forecast_calibration"]


def test_candidate_factor_builds_factor_health_schedule_when_enabled(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset_2024_01.parquet"
    pd.DataFrame(
        [
            {
                "timestamp": f"t{index}",
                "instrument_id": instrument,
                "alpha_a": value,
                "forward_return": value,
            }
            for index in range(2)
            for instrument, value in (("a", 1.0), ("b", 2.0), ("c", 3.0))
        ]
    ).to_parquet(dataset_path, index=False)
    args = _portfolio_args(
        dataset_dir=str(tmp_path),
        factor_health_mode="shrink",
        factor_health_lookback_windows=1,
        factor_health_min_periods=1,
        factor_health_label_lag_windows=1,
        score_diagnostics_top_n=1,
    )

    schedule = _build_factor_health_schedule(
        args,
        [dataset_path],
        (CandidateFactor("alpha_a", 1, 0.02),),
    )

    assert schedule is not None
    assert len(schedule) == 2


def test_candidate_factor_script_filters_dataset_partitions(tmp_path: Path) -> None:
    for partition in ("2023_01", "2023_02", "2023_03", "2023_04"):
        (tmp_path / f"dataset_{partition}.parquet").touch()

    args = type(
        "Args",
        (),
        {
            "dataset_dir": str(tmp_path),
            "partition_start": "2023_02",
            "partition_end": "2023_03",
            "max_partitions": None,
        },
    )()

    assert [path.name for path in _dataset_paths(args)] == [
        "dataset_2023_02.parquet",
        "dataset_2023_03.parquet",
    ]


def test_candidate_factor_policy_set_builds_standard_comparison_specs() -> None:
    args = _portfolio_args(
        backtest_policy_set="comparison",
        top_n=50,
        policy_no_trade_weight_band=0.002,
    )

    specs = _backtest_policy_specs(args)

    assert [spec.name for spec in specs] == [
        "naive_top_n_every_bar",
        "top_k_drop_daily",
        "entry_exit_buffer_every_bar",
        "entry_exit_buffer_daily",
        "partial_rebalance_daily",
    ]
    top_k_drop = specs[1]
    assert top_k_drop.trade_policy == "rank_buffer_drop"
    assert top_k_drop.rebalance_every_n_bars == 48
    assert top_k_drop.policy_entry_rank == 50
    assert top_k_drop.policy_exit_rank == 50
    assert top_k_drop.policy_max_entries_per_rebalance == 10
    assert top_k_drop.policy_max_exits_per_rebalance == 10
    buffered = specs[3]
    assert buffered.policy_exit_rank == 150
    assert buffered.policy_no_trade_weight_band == pytest.approx(0.002)
    assert specs[4].policy_partial_rebalance_rate == pytest.approx(0.5)


def test_candidate_factor_policy_set_filters_named_specs() -> None:
    args = _portfolio_args(
        backtest_policy_set="comparison",
        backtest_policies=["top_k_drop_daily", "partial_rebalance_daily"],
    )

    specs = _backtest_policy_specs(args)

    assert [spec.name for spec in specs] == [
        "top_k_drop_daily",
        "partial_rebalance_daily",
    ]


def test_candidate_factor_policy_set_rejects_unknown_filter() -> None:
    args = _portfolio_args(
        backtest_policy_set="comparison",
        backtest_policies=["missing_policy"],
    )

    with pytest.raises(ValueError, match="unknown backtest policies"):
        _backtest_policy_specs(args)


def test_candidate_factor_backtest_command_includes_policy_args(tmp_path: Path) -> None:
    args = _portfolio_args(max_bar_turnover_participation=0.05)
    spec = BacktestPolicySpec(
        name="entry_exit_buffer_daily",
        trade_policy="rank_buffer_drop",
        rebalance_every_n_bars=48,
        policy_entry_rank=50,
        policy_exit_rank=150,
        policy_max_entries_per_rebalance=10,
        policy_max_exits_per_rebalance=10,
        policy_no_trade_weight_band=0.002,
        policy_partial_rebalance_rate=0.5,
        policy_gross_exposure_scale=0.75,
        policy_gross_exposure_scale_path="gate.csv",
    )

    command = _backtest_command(args, "scores/*.parquet", tmp_path / "bt", spec)

    assert command[command.index("--trade-policy") + 1] == "rank_buffer_drop"
    assert command[command.index("--rebalance-every-n-bars") + 1] == "48"
    assert command[command.index("--policy-entry-rank") + 1] == "50"
    assert command[command.index("--policy-exit-rank") + 1] == "150"
    assert command[command.index("--policy-no-trade-weight-band") + 1] == "0.002"
    assert command[command.index("--policy-partial-rebalance-rate") + 1] == "0.5"
    assert command[command.index("--policy-gross-exposure-scale") + 1] == "0.75"
    assert command[command.index("--policy-gross-exposure-scale-path") + 1] == "gate.csv"
    assert command[command.index("--max-bar-turnover-participation") + 1] == "0.05"


def test_candidate_factor_backtest_jobs_use_nested_policy_paths(tmp_path: Path) -> None:
    args = _portfolio_args(
        output_dir=str(tmp_path),
        backtest_policy_set="comparison",
        backtest_policies=["top_k_drop_daily", "partial_rebalance_daily"],
        backtest_memory_estimate_gb=4.5,
    )
    scores_summary = {
        "methods": {
            "decorrelated": {"path": "scores/decorrelated/*.parquet"},
        }
    }

    jobs = _backtest_jobs(args, scores_summary=scores_summary)

    assert len(jobs) == 2
    assert jobs[0].summary_path == (
        tmp_path
        / "backtests"
        / "decorrelated"
        / "top_k_drop_daily"
        / "summary.json"
    )
    assert jobs[0].log_path == (
        tmp_path / "logs" / "backtest_decorrelated_top_k_drop_daily.log"
    )
    assert jobs[0].memory_estimate_gb == pytest.approx(4.5)


def test_candidate_factor_backtest_summary_rows_flatten_nested_results() -> None:
    rows = _backtest_summary_rows(
        {
            "decorrelated": {
                "entry_exit_buffer_daily": {
                    "params": {
                        "trade_policy": "rank_buffer_drop",
                        "rebalance_every_n_bars": 48,
                        "policy_entry_rank": 50,
                        "policy_exit_rank": 150,
                        "policy_max_entries_per_rebalance": 10,
                        "policy_max_exits_per_rebalance": 10,
                        "policy_no_trade_weight_band": 0.002,
                        "policy_partial_rebalance_rate": 1.0,
                        "policy_gross_exposure_scale": 0.75,
                        "policy_gross_exposure_scale_path": "gate.csv",
                    },
                    "metrics": {
                        "total_return": 0.079,
                        "max_drawdown": -0.06,
                        "gross_turnover": 47.4,
                        "trade_count": 1148,
                        "total_transaction_cost": 33462,
                        "final_equity": 1_079_000,
                    },
                    "signal_count": 1323,
                    "execution_row_count": 904224,
                    "policy_diagnostics": {
                        "planned_gross_turnover": 54.4,
                        "average_target_gross_exposure": 0.74,
                        "gross_exposure_scaled_count": 96,
                        "risk_reduction_count": 24,
                        "order_intent_count": 1232,
                        "entry_count": 590,
                        "exit_count": 564,
                        "hold_count": 655,
                        "no_trade_count": 0,
                    },
                }
            }
        }
    )

    assert rows == [
        {
            "method": "decorrelated",
            "policy": "entry_exit_buffer_daily",
            "trade_policy": "rank_buffer_drop",
            "rebalance_every_n_bars": 48,
            "policy_entry_rank": 50,
            "policy_exit_rank": 150,
            "policy_max_entries_per_rebalance": 10,
            "policy_max_exits_per_rebalance": 10,
            "policy_no_trade_weight_band": 0.002,
            "policy_partial_rebalance_rate": 1.0,
            "policy_gross_exposure_scale": 0.75,
            "policy_gross_exposure_scale_path": "gate.csv",
            "optimizer_candidate_rank": None,
            "optimizer_score_to_edge_bps": None,
            "optimizer_min_net_edge_bps": None,
            "optimizer_risk_penalty_multiplier": None,
            "optimizer_weighting": None,
            "optimizer_max_name_weight": None,
            "optimizer_max_gross_exposure_increase_per_rebalance": None,
            "total_return": 0.079,
            "max_drawdown": -0.06,
            "gross_turnover": 47.4,
            "trade_count": 1148,
            "total_transaction_cost": 33462,
            "final_equity": 1_079_000,
            "signal_count": 1323,
            "execution_row_count": 904224,
            "planned_gross_turnover": 54.4,
            "average_target_gross_exposure": 0.74,
            "gross_exposure_scaled_count": 96,
            "risk_reduction_count": 24,
            "order_intent_count": 1232,
            "entry_count": 590,
            "exit_count": 564,
            "hold_count": 655,
            "no_trade_count": 0,
        }
    ]


def test_candidate_factor_backtest_memory_budget_auto_detects_available() -> None:
    args = _portfolio_args(
        backtest_memory_budget_gb=0.0,
        backtest_memory_estimate_gb=5.0,
    )

    assert _effective_backtest_memory_budget_gb(args) >= 5.0


def test_candidate_factor_summary_params_record_backtest_policy_set() -> None:
    args = _portfolio_args(
        run_backtests=True,
        backtest_policy_set="comparison",
        backtest_policies=["partial_rebalance_daily"],
        policy_set_exit_rank=150,
        backtest_workers=2,
    )

    params = _summary_params(args)

    assert params["run_backtests"] is True
    assert params["backtest"]["backtest_policy_set"] == "comparison"  # type: ignore[index]
    assert params["backtest"]["backtest_policies"] == ["partial_rebalance_daily"]  # type: ignore[index]
    assert params["backtest"]["policy_set_exit_rank"] == 150  # type: ignore[index]
    assert params["backtest"]["backtest_workers"] == 2  # type: ignore[index]


def _portfolio_args(**overrides: object) -> object:
    defaults = {
        "dataset_dir": "dataset",
        "admission_report": "admission.json",
        "factor_correlation": "correlation.csv",
        "methods": ["decorrelated"],
        "statuses": ["candidate"],
        "max_partitions": None,
        "partition_start": None,
        "partition_end": None,
        "run_backtests": False,
        "output_dir": "runs",
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
        "forecast_calibration_mode": "off",
        "forecast_calibration_lookback_windows": 20,
        "forecast_calibration_min_periods": 5,
        "forecast_calibration_label_lag_windows": 48,
        "forecast_calibration_bucket_count": 5,
        "forecast_calibration_risk_multiplier": 1.0,
        "forecast_calibration_max_abs_edge_bps": None,
        "score_diagnostics_top_n": None,
        "backtest_policy_set": "single",
        "backtest_policies": None,
        "trade_policy": "naive_top_n",
        "rebalance_every_n_bars": 1,
        "hold_rank_buffer": None,
        "policy_entry_rank": None,
        "policy_exit_rank": None,
        "policy_max_entries_per_rebalance": None,
        "policy_max_exits_per_rebalance": None,
        "policy_min_hold_bars": 0,
        "policy_min_expected_edge_bps": None,
        "policy_estimated_cost_bps": 0.0,
        "policy_no_trade_weight_band": 0.0,
        "policy_partial_rebalance_rate": 1.0,
        "policy_max_gross_turnover_per_rebalance": None,
        "policy_gross_exposure_scale": 1.0,
        "policy_gross_exposure_scale_path": None,
        "optimizer_candidate_rank": None,
        "optimizer_score_to_edge_bps": 100.0,
        "optimizer_min_net_edge_bps": 0.0,
        "optimizer_risk_penalty_multiplier": 1.0,
        "optimizer_weighting": "utility",
        "optimizer_max_name_weight": None,
        "optimizer_max_gross_exposure_increase_per_rebalance": None,
        "policy_set_drop_count": 10,
        "policy_set_exit_rank": None,
        "policy_set_rebalance_every_n_bars": 48,
        "policy_set_partial_rebalance_rate": 0.5,
        "backtest_workers": 1,
        "backtest_memory_budget_gb": 0.0,
        "backtest_memory_estimate_gb": 5.0,
        "resume_existing": False,
        "catalog_path": "catalog.duckdb",
        "start": "2023-01-03T09:35:00+08:00",
        "end": "2023-03-31T15:00:00+08:00",
        "top_n": 50,
        "initial_cash": 1_000_000.0,
        "commission_bps": 3.0,
        "slippage_bps": 1.0,
        "sell_stamp_tax_bps": 5.0,
        "min_commission": 5.0,
        "lot_size": 100,
        "min_trade_weight": 0.0005,
        "exclude_st": True,
        "limit_up_bps": 980.0,
        "limit_down_bps": 980.0,
        "max_bar_turnover_participation": None,
        "data_access_mode": "fast_parquet",
        "streaming_chunk": "month",
        "streaming_chunk_padding_days": 10,
    }
    defaults.update(overrides)
    return type("Args", (), defaults)()
