from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from examples.run_candidate_factor_portfolios import (
    BacktestPolicySpec,
    _backtest_command,
    _backtest_policy_specs,
    _dataset_paths,
    _summary_params,
)
from quant_research.portfolio import (
    CandidateFactor,
    build_composite_scores,
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


def test_write_score_partitions_writes_one_partition_per_method(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset_2024_01.parquet"
    pd.DataFrame(
        [
            {"timestamp": "t0", "instrument_id": "a", "alpha_a": 1.0},
            {"timestamp": "t0", "instrument_id": "b", "alpha_a": 2.0},
        ]
    ).to_parquet(dataset_path, index=False)

    summary = write_score_partitions(
        [dataset_path],
        output_dir=tmp_path / "scores",
        candidates=(CandidateFactor("alpha_a", 1, 0.02),),
        weights_by_method={"equal": {"alpha_a": 1.0}},
    )

    assert summary["methods"]["equal"]["row_count"] == 2
    assert Path(tmp_path / "scores" / "equal" / "score_2024_01.parquet").exists()


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
    )

    command = _backtest_command(args, "scores/*.parquet", tmp_path / "bt", spec)

    assert command[command.index("--trade-policy") + 1] == "rank_buffer_drop"
    assert command[command.index("--rebalance-every-n-bars") + 1] == "48"
    assert command[command.index("--policy-entry-rank") + 1] == "50"
    assert command[command.index("--policy-exit-rank") + 1] == "150"
    assert command[command.index("--policy-no-trade-weight-band") + 1] == "0.002"
    assert command[command.index("--policy-partial-rebalance-rate") + 1] == "0.5"
    assert command[command.index("--max-bar-turnover-participation") + 1] == "0.05"


def test_candidate_factor_summary_params_record_backtest_policy_set() -> None:
    args = _portfolio_args(
        run_backtests=True,
        backtest_policy_set="comparison",
        policy_set_exit_rank=150,
    )

    params = _summary_params(args)

    assert params["run_backtests"] is True
    assert params["backtest"]["backtest_policy_set"] == "comparison"  # type: ignore[index]
    assert params["backtest"]["policy_set_exit_rank"] == 150  # type: ignore[index]


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
        "backtest_policy_set": "single",
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
        "policy_set_drop_count": 10,
        "policy_set_exit_rank": None,
        "policy_set_rebalance_every_n_bars": 48,
        "policy_set_partial_rebalance_rate": 0.5,
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
