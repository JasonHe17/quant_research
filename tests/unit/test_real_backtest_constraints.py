from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import warnings

import pandas as pd
import pytest

from quant_research.backtest import (
    TargetWeightExecutionConfig,
    TargetWeightExecutionSimulator,
    simulate_target_weight_execution_batches,
)
from examples.build_baseline_a_alpha_dataset import (
    _add_entry_execution_columns,
    _entry_execution_filter_counts,
    _filter_entry_execution_constraints,
)
from examples.run_baseline_a_real_backtest import (
    BacktestParams,
    SimulationState,
    _add_execution_constraint_columns,
    _bar_time_index,
    _build_next_bar_executions,
    _build_reversal_signals,
    _execution_event_constraint_counts,
    _execution_constraint_counts,
    _merge_execution_constraint_counts,
    _load_bars_from_files,
    _minute_bar_files_for_range,
    _simulate,
)
from examples.run_baseline_a_grid import (
    GridRun,
    _run_grid_file_parallel,
    _run_grid_file_sequential,
    _build_reversal_signals_from_features,
    _factor_column,
    _prepare_grid_features,
)


def test_execution_constraint_columns_flag_st_and_limit_states() -> None:
    bars = pd.DataFrame(
        [
            {
                "bar_end_time": "2025-01-02T15:00:00+08:00",
                "instrument_id": "inst-a",
                "open_price": 10.0,
                "close_price": 10.0,
                "volume": 100.0,
                "turnover": 1_000.0,
                "raw_name": "Plain A",
            },
            {
                "bar_end_time": "2025-01-03T09:35:00+08:00",
                "instrument_id": "inst-a",
                "open_price": 11.0,
                "close_price": 11.0,
                "volume": 100.0,
                "turnover": 1_100.0,
                "raw_name": "Plain A",
            },
            {
                "bar_end_time": "2025-01-02T15:00:00+08:00",
                "instrument_id": "inst-b",
                "open_price": 10.0,
                "close_price": 10.0,
                "volume": 100.0,
                "turnover": 1_000.0,
                "raw_name": "Plain B",
            },
            {
                "bar_end_time": "2025-01-03T09:35:00+08:00",
                "instrument_id": "inst-b",
                "open_price": 9.0,
                "close_price": 9.0,
                "volume": 100.0,
                "turnover": 900.0,
                "raw_name": "Plain B",
            },
            {
                "bar_end_time": "2025-01-03T09:35:00+08:00",
                "instrument_id": "inst-st",
                "open_price": 8.0,
                "close_price": 8.0,
                "volume": 100.0,
                "turnover": 800.0,
                "raw_name": "*ST Sample",
            },
        ]
    )

    constrained = _add_execution_constraint_columns(
        bars,
        _params(exclude_st=True, limit_up_bps=980.0, limit_down_bps=980.0),
    )

    limit_up_row = constrained.loc[
        (constrained["instrument_id"] == "inst-a")
        & (constrained["trade_date"] == "2025-01-03")
    ].iloc[0]
    limit_down_row = constrained.loc[
        (constrained["instrument_id"] == "inst-b")
        & (constrained["trade_date"] == "2025-01-03")
    ].iloc[0]
    st_row = constrained.loc[constrained["instrument_id"] == "inst-st"].iloc[0]

    assert bool(limit_up_row["limit_up_open"])
    assert not bool(limit_up_row["buyable_bar"])
    assert bool(limit_down_row["limit_down_open"])
    assert not bool(limit_down_row["sellable_bar"])
    assert bool(st_row["is_st"])
    assert not bool(st_row["tradable_bar"])
    assert bool(limit_up_row["sellable_bar"])
    assert bool(limit_down_row["buyable_bar"])


def test_simulation_blocks_limit_up_buys_and_limit_down_sells() -> None:
    executions = pd.DataFrame(
        [
            {
                "exec_time": "2025-01-03T09:35:00+08:00",
                "instrument_id": "held",
                "canonical_code": "600000.SH",
                "open_price": 10.0,
                "close_price": 10.0,
                "turnover": 1_000_000.0,
                "tradable_bar": True,
                "limit_up_open": False,
                "limit_down_open": True,
                "target_weight": 0.0,
            },
            {
                "exec_time": "2025-01-03T09:35:00+08:00",
                "instrument_id": "buy",
                "canonical_code": "600001.SH",
                "open_price": 10.0,
                "close_price": 10.0,
                "turnover": 1_000_000.0,
                "tradable_bar": True,
                "limit_up_open": True,
                "limit_down_open": False,
                "target_weight": 1.0,
            },
        ]
    )
    state = SimulationState(
        cash=10_000.0,
        lots={"held": [{"shares": 1_000, "date": "2025-01-02", "sellable": True}]},
        previous_date="2025-01-02",
        last_prices={"held": 10.0},
    )

    trades, _, _, next_state = _simulate(executions, _params(), state=state)

    assert trades.empty
    assert next_state.cash == 10_000.0
    assert sum(int(lot["shares"]) for lot in next_state.lots["held"]) == 1_000


def test_simulation_is_independent_of_execution_frame_index() -> None:
    executions = pd.DataFrame(
        [
            {
                "exec_time": "2025-01-03T09:35:00+08:00",
                "instrument_id": "buy-a",
                "canonical_code": "600001.SH",
                "open_price": 10.0,
                "close_price": 11.0,
                "turnover": 1_000_000.0,
                "tradable_bar": True,
                "limit_up_open": False,
                "limit_down_open": False,
                "target_weight": 0.5,
            },
            {
                "exec_time": "2025-01-03T09:35:00+08:00",
                "instrument_id": "buy-b",
                "canonical_code": "600002.SH",
                "open_price": 20.0,
                "close_price": 19.0,
                "turnover": 1_000_000.0,
                "tradable_bar": True,
                "limit_up_open": False,
                "limit_down_open": False,
                "target_weight": 0.5,
            },
            {
                "exec_time": "2025-01-06T09:35:00+08:00",
                "instrument_id": "buy-a",
                "canonical_code": "600001.SH",
                "open_price": 11.0,
                "close_price": 11.5,
                "turnover": 1_000_000.0,
                "tradable_bar": True,
                "limit_up_open": False,
                "limit_down_open": False,
                "target_weight": 0.0,
            },
        ],
        index=[10, 20, 30],
    )
    params = _params(lot_size=1)

    indexed_trades, indexed_equity, indexed_positions, _ = _simulate(
        executions,
        params,
    )
    reset_trades, reset_equity, reset_positions, _ = _simulate(
        executions.reset_index(drop=True),
        params,
    )

    pd.testing.assert_frame_equal(indexed_trades, reset_trades)
    pd.testing.assert_frame_equal(indexed_equity, reset_equity)
    pd.testing.assert_frame_equal(indexed_positions, reset_positions)


def test_baseline_execution_builder_keeps_only_targets_and_tracked_holdings() -> None:
    bars = pd.DataFrame(
        [
            _bar("t0", "inst-a"),
            _bar("t0", "inst-held"),
            _bar("t0", "inst-unused"),
            _bar("t1", "inst-a"),
            _bar("t1", "inst-held"),
            _bar("t1", "inst-unused"),
            _bar("t2", "inst-a"),
            _bar("t2", "inst-held"),
            _bar("t2", "inst-unused"),
        ]
    )
    signals = pd.DataFrame(
        [
            {
                "signal_time": "t0",
                "instrument_id": "inst-a",
                "target_weight": 1.0,
            }
        ]
    )

    executions = _build_next_bar_executions(
        bars,
        signals,
        tracked_instruments={"inst-held"},
    )

    assert set(executions["instrument_id"]) == {"inst-a", "inst-held"}
    assert len(executions) == 2
    target = executions.loc[
        (executions["exec_time"] == "t1") & (executions["instrument_id"] == "inst-a"),
        "target_weight",
    ]
    held = executions.loc[
        (executions["exec_time"] == "t1") & (executions["instrument_id"] == "inst-held"),
        "target_weight",
    ]
    assert target.iloc[0] == pytest.approx(1.0)
    assert pd.isna(held.iloc[0])


def test_baseline_execution_builder_matches_precomputed_time_index() -> None:
    bars = pd.DataFrame(
        [
            _bar("t0", "inst-a"),
            _bar("t0", "inst-held"),
            _bar("t0", "inst-unused"),
            _bar("t1", "inst-a"),
            _bar("t1", "inst-held"),
            _bar("t1", "inst-unused"),
            _bar("t2", "inst-a"),
            _bar("t2", "inst-held"),
            _bar("t2", "inst-unused"),
        ]
    )
    signals = pd.DataFrame(
        [
            {
                "signal_time": "t0",
                "instrument_id": "inst-a",
                "target_weight": 1.0,
            }
        ]
    )

    default = _build_next_bar_executions(
        bars,
        signals,
        tracked_instruments={"inst-held"},
    ).reset_index(drop=True)
    indexed = _build_next_bar_executions(
        bars,
        signals,
        tracked_instruments={"inst-held"},
        bar_time_index=_bar_time_index(bars),
    ).reset_index(drop=True)

    pd.testing.assert_frame_equal(indexed, default)


def test_reversal_signals_select_top_names_with_stable_tie_breaks() -> None:
    bars = pd.DataFrame(
        [
            _bar_with_close("t0", "inst-a", "600001.SH", 10.0),
            _bar_with_close("t0", "inst-b", "600000.SH", 10.0),
            _bar_with_close("t0", "inst-c", "600002.SH", 10.0),
            _bar_with_close("t1", "inst-a", "600001.SH", 9.0),
            _bar_with_close("t1", "inst-b", "600000.SH", 9.0),
            _bar_with_close("t1", "inst-c", "600002.SH", 11.0),
            _bar_with_close("t2", "inst-a", "600001.SH", 8.0),
            _bar_with_close("t2", "inst-b", "600000.SH", 8.0),
            _bar_with_close("t2", "inst-c", "600002.SH", 12.0),
        ]
    )

    signals = _build_reversal_signals(bars, _params(top_n=2, lookback_bars=1))

    first_time = signals.loc[signals["signal_time"] == "t1"]
    assert first_time["instrument_id"].tolist() == ["inst-b", "inst-a"]
    assert first_time["target_weight"].tolist() == [0.5, 0.5]
    assert "inst-c" not in set(first_time["instrument_id"])


def test_grid_reversal_signals_select_top_names_from_precomputed_features() -> None:
    bars = pd.DataFrame(
        [
            _feature_bar("t1", "inst-a", "600001.SH", 0.2),
            _feature_bar("t1", "inst-b", "600000.SH", 0.2),
            _feature_bar("t1", "inst-c", "600002.SH", -0.1),
            _feature_bar("t2", "inst-a", "600001.SH", 0.3),
        ]
    )

    signals = _build_reversal_signals_from_features(
        bars,
        _params(top_n=2, lookback_bars=1),
    )

    first_time = signals.loc[signals["signal_time"] == "t1"]
    assert first_time["instrument_id"].tolist() == ["inst-b", "inst-a"]
    assert first_time["target_weight"].tolist() == [0.5, 0.5]
    assert signals.loc[
        signals["signal_time"] == "t2", "target_weight"
    ].tolist() == [1.0]


def test_grid_parallel_run_matches_sequential_run() -> None:
    bars = pd.DataFrame(
        [
            _bar_with_close("t0", "inst-a", "600001.SH", 10.0),
            _bar_with_close("t0", "inst-b", "600000.SH", 10.0),
            _bar_with_close("t0", "inst-c", "600002.SH", 10.0),
            _bar_with_close("t1", "inst-a", "600001.SH", 9.0),
            _bar_with_close("t1", "inst-b", "600000.SH", 11.0),
            _bar_with_close("t1", "inst-c", "600002.SH", 8.0),
            _bar_with_close("t2", "inst-a", "600001.SH", 8.0),
            _bar_with_close("t2", "inst-b", "600000.SH", 12.0),
            _bar_with_close("t2", "inst-c", "600002.SH", 9.0),
        ]
    )
    sequential_runs = _grid_runs()
    parallel_runs = _grid_runs()

    _prepare_grid_features(bars, sequential_runs)
    _run_grid_file_sequential(
        sequential_runs,
        bars=bars,
        write_run_artifacts=False,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*multi-threaded, use of fork\\(\\) may lead to deadlocks.*",
            category=DeprecationWarning,
        )
        _run_grid_file_parallel(
            parallel_runs,
            bars=bars,
            write_run_artifacts=False,
            workers=2,
        )

    for sequential, parallel in zip(sequential_runs, parallel_runs, strict=True):
        assert sequential.signal_count == parallel.signal_count
        assert sequential.trade_count == parallel.trade_count
        assert sequential.equity_values == parallel.equity_values
        assert sequential.state.cash == pytest.approx(parallel.state.cash)
        assert sequential.state.lots == parallel.state.lots


def test_simulation_caps_trade_size_by_bar_turnover_participation() -> None:
    executions = pd.DataFrame(
        [
            {
                "exec_time": "2025-01-03T09:35:00+08:00",
                "instrument_id": "buy",
                "canonical_code": "600001.SH",
                "open_price": 10.0,
                "close_price": 10.0,
                "turnover": 100_000.0,
                "tradable_bar": True,
                "limit_up_open": False,
                "limit_down_open": False,
                "target_weight": 1.0,
            },
        ]
    )

    execution_events: list[dict[str, object]] = []
    trades, _, _, next_state = _simulate(
        executions,
        _params(
            initial_cash=100_000.0,
            max_bar_turnover_participation=0.05,
            allow_same_bar_capacity=True,
        ),
        diagnostics=execution_events,
    )
    counts = _execution_event_constraint_counts(pd.DataFrame(execution_events))

    assert list(trades["shares"]) == [500]
    assert next_state.cash == 95_000.0
    assert counts["capacity_capped_event_count"] == 1
    assert counts["capacity_limited_event_count"] == 1
    assert counts["capacity_desired_shares"] == 10_000
    assert counts["capacity_executable_shares"] == 500
    assert counts["capacity_unfilled_shares"] == 9_500
    assert counts["capacity_unfilled_notional"] == 95_000.0


def test_open_execution_rejects_same_bar_capacity_without_explicit_policy() -> None:
    with pytest.raises(ValueError, match="same-bar capacity"):
        TargetWeightExecutionConfig(
            initial_cash=100_000.0,
            max_bar_turnover_participation=0.05,
        )


def test_execution_event_counts_capacity_zero_without_duplicate_cap() -> None:
    executions = pd.DataFrame(
        [
            {
                "exec_time": "2025-01-03T09:35:00+08:00",
                "instrument_id": "buy",
                "canonical_code": "600001.SH",
                "open_price": 10.0,
                "close_price": 10.0,
                "turnover": 1_000.0,
                "tradable_bar": True,
                "limit_up_open": False,
                "limit_down_open": False,
                "target_weight": 1.0,
            },
        ]
    )
    execution_events: list[dict[str, object]] = []

    trades, _, _, next_state = _simulate(
        executions,
        _params(
            initial_cash=100_000.0,
            max_bar_turnover_participation=0.05,
            allow_same_bar_capacity=True,
        ),
        diagnostics=execution_events,
    )
    counts = _execution_event_constraint_counts(pd.DataFrame(execution_events))

    assert trades.empty
    assert next_state.cash == 100_000.0
    assert "capacity_capped_event_count" not in counts
    assert counts["capacity_zero_event_count"] == 1
    assert counts["capacity_limited_event_count"] == 1
    assert counts["capacity_desired_shares"] == 10_000
    assert counts["capacity_executable_shares"] == 0
    assert counts["capacity_unfilled_shares"] == 10_000
    assert counts["capacity_unfilled_notional"] == 100_000.0


def test_simulation_sizes_open_execution_without_same_bar_close() -> None:
    executions = pd.DataFrame(
        [
            {
                "exec_time": "2025-01-03T09:35:00+08:00",
                "instrument_id": "held",
                "canonical_code": "600000.SH",
                "open_price": 10.0,
                "close_price": 100.0,
                "turnover": 1_000_000.0,
                "tradable_bar": True,
                "limit_up_open": False,
                "limit_down_open": False,
                "target_weight": 0.5,
            },
        ]
    )
    state = SimulationState(
        cash=0.0,
        lots={"held": [{"shares": 100, "date": "2025-01-02", "sellable": True}]},
        previous_date="2025-01-02",
        last_prices={"held": 10.0},
    )

    trades, _, final_positions, _ = _simulate(
        executions,
        _params(initial_cash=1_000.0, lot_size=1),
        state=state,
    )

    assert trades.loc[0, "side"] == "sell"
    assert trades.loc[0, "shares"] == 50
    assert final_positions.loc[0, "shares"] == 50


def test_target_weight_execution_simulator_keeps_state_between_batches() -> None:
    simulator = TargetWeightExecutionSimulator(
        TargetWeightExecutionConfig(initial_cash=20_000.0, lot_size=100)
    )
    first_batch = pd.DataFrame(
        [
            {
                "exec_time": "2025-01-03T09:35:00+08:00",
                "instrument_id": "buy",
                "canonical_code": "600001.SH",
                "open_price": 10.0,
                "close_price": 10.0,
                "turnover": 1_000_000.0,
                "tradable_bar": True,
                "limit_up_open": False,
                "limit_down_open": False,
                "target_weight": 0.5,
            }
        ]
    )
    second_batch = first_batch.assign(
        exec_time="2025-01-06T09:35:00+08:00",
        target_weight=0.0,
    )

    first_trades, _, _ = simulator.run(first_batch)
    second_trades, _, positions = simulator.run(second_batch)

    assert first_trades.loc[0, "side"] == "buy"
    assert second_trades.loc[0, "side"] == "sell"
    assert positions.empty


def test_target_weight_batch_execution_api_keeps_continuous_state() -> None:
    config = TargetWeightExecutionConfig(initial_cash=20_000.0, lot_size=100)
    first_batch = pd.DataFrame(
        [
            {
                "exec_time": "2025-01-03T09:35:00+08:00",
                "instrument_id": "buy",
                "canonical_code": "600001.SH",
                "open_price": 10.0,
                "close_price": 10.0,
                "turnover": 1_000_000.0,
                "tradable_bar": True,
                "limit_up_open": False,
                "limit_down_open": False,
                "target_weight": 0.5,
            }
        ]
    )
    second_batch = first_batch.assign(
        exec_time="2025-01-06T09:35:00+08:00",
        target_weight=0.0,
    )

    trades, equity, positions, diagnostics, state = (
        simulate_target_weight_execution_batches(
            [first_batch, second_batch],
            config,
        )
    )

    assert trades["side"].tolist() == ["buy", "sell"]
    assert len(equity) == 2
    assert positions.empty
    assert state.cash == 20_000.0
    assert diagnostics["batch_index"].tolist() == [0, 1]


def test_target_weight_execution_simulator_reports_diagnostics() -> None:
    simulator = TargetWeightExecutionSimulator(
        TargetWeightExecutionConfig(initial_cash=20_000.0, lot_size=100)
    )
    executions = pd.DataFrame(
        [
            {
                "exec_time": "2025-01-03T09:35:00+08:00",
                "instrument_id": "buy",
                "canonical_code": "600001.SH",
                "open_price": 10.0,
                "close_price": 10.0,
                "turnover": 1_000_000.0,
                "tradable_bar": True,
                "limit_up_open": False,
                "limit_down_open": False,
                "target_weight": 0.5,
            },
            {
                "exec_time": "2025-01-03T09:35:00+08:00",
                "instrument_id": "blocked",
                "canonical_code": "600002.SH",
                "open_price": 10.0,
                "close_price": 10.0,
                "turnover": 1_000_000.0,
                "tradable_bar": False,
                "limit_up_open": False,
                "limit_down_open": False,
                "target_weight": 0.5,
            },
        ]
    )

    trades, _, _, diagnostics = simulator.run_with_diagnostics(executions)

    assert len(trades) == 1
    assert diagnostics.loc[0, "trade_count"] == 1
    assert diagnostics.loc[0, "positive_target_non_tradable_row_count"] == 1
    assert diagnostics.loc[0, "non_tradable_event_count"] == 1
    assert diagnostics.loc[0, "gross_traded_notional"] == 10_000.0
    assert simulator.last_execution_events["reason"].tolist() == ["non_tradable"]


def test_execution_constraint_counts_include_targeted_limit_rows() -> None:
    executions = pd.DataFrame(
        [
            {
                "tradable_bar": True,
                "limit_up_open": True,
                "limit_down_open": False,
                "target_weight": 0.5,
            },
            {
                "tradable_bar": False,
                "limit_up_open": False,
                "limit_down_open": True,
                "target_weight": 0.0,
            },
            {
                "tradable_bar": False,
                "limit_up_open": False,
                "limit_down_open": False,
                "target_weight": 0.5,
            },
        ]
    )

    counts = _execution_constraint_counts(executions)
    _merge_execution_constraint_counts(
        counts,
        {
            "capacity_limited_event_count": 2,
            "capacity_unfilled_notional": 123.5,
        },
    )

    assert counts["execution_row_count"] == 3
    assert counts["non_tradable_row_count"] == 2
    assert counts["limit_up_open_row_count"] == 1
    assert counts["limit_down_open_row_count"] == 1
    assert counts["positive_target_row_count"] == 2
    assert counts["positive_target_non_tradable_row_count"] == 1
    assert counts["positive_target_limit_up_open_row_count"] == 1
    assert counts["positive_target_limit_down_open_row_count"] == 0
    assert counts["capacity_limited_event_count"] == 2
    assert counts["capacity_unfilled_notional"] == 123.5
    assert {
        key: counts[key]
        for key in (
            "execution_row_count",
            "non_tradable_row_count",
            "limit_up_open_row_count",
            "limit_down_open_row_count",
            "positive_target_row_count",
            "positive_target_non_tradable_row_count",
            "positive_target_limit_up_open_row_count",
            "positive_target_limit_down_open_row_count",
        )
    } == {
        "execution_row_count": 3,
        "non_tradable_row_count": 2,
        "limit_up_open_row_count": 1,
        "limit_down_open_row_count": 1,
        "positive_target_row_count": 2,
        "positive_target_non_tradable_row_count": 1,
        "positive_target_limit_up_open_row_count": 1,
        "positive_target_limit_down_open_row_count": 0,
    }


def test_entry_execution_filter_removes_unbuyable_training_labels() -> None:
    labels = pd.DataFrame(
        [
            {
                "timestamp": "t0",
                "instrument_id": "buyable",
                "entry_timestamp": "t1",
                "forward_return": 0.01,
            },
            {
                "timestamp": "t0",
                "instrument_id": "limit-up",
                "entry_timestamp": "t1",
                "forward_return": 0.10,
            },
            {
                "timestamp": "t0",
                "instrument_id": "halted",
                "entry_timestamp": "t1",
                "forward_return": 0.20,
            },
        ]
    )
    bars = pd.DataFrame(
        [
            {
                "instrument_id": "buyable",
                "bar_end_time": "t1",
                "tradable_bar": True,
                "limit_up_open": False,
                "limit_down_open": False,
            },
            {
                "instrument_id": "limit-up",
                "bar_end_time": "t1",
                "tradable_bar": True,
                "limit_up_open": True,
                "limit_down_open": False,
            },
            {
                "instrument_id": "halted",
                "bar_end_time": "t1",
                "tradable_bar": False,
                "limit_up_open": False,
                "limit_down_open": False,
            },
        ]
    )

    enriched = _add_entry_execution_columns(labels, bars)
    counts = _entry_execution_filter_counts(enriched)
    filtered = _filter_entry_execution_constraints(
        enriched,
        SimpleNamespace(filter_entry_tradable=True, filter_entry_limit_up=True),
    )

    assert counts["entry_non_tradable_label_count"] == 1
    assert counts["entry_limit_up_label_count"] == 1
    assert filtered["instrument_id"].tolist() == ["buyable"]


def test_minute_bar_files_include_baostock_update_for_2026(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "canonical_store" / "v1" / "market" / "records=minute_bar"
    data_dir.mkdir(parents=True)
    full_2025 = (
        data_dir
        / "market_cn_equity_full__a股_分时数据_沪深__5分钟_按年汇总__2025.parquet"
    )
    update_5m = data_dir / "market_baostock_cn_equity_update__5m__5m.parquet"
    full_2025.write_text("", encoding="utf-8")
    update_5m.write_text("", encoding="utf-8")
    params = _params(
        catalog_path=tmp_path
        / "canonical_store"
        / "catalog"
        / "quant_research.duckdb",
        start="2026-01-02T09:35:00+08:00",
        end="2026-01-02T15:00:00+08:00",
    )

    files = _minute_bar_files_for_range(
        params,
        start="2026-01-02T09:35:00+08:00",
        end="2026-01-02T15:00:00+08:00",
    )

    assert files == [update_5m]


def test_load_bars_from_files_deduplicates_full_and_update_rows(
    tmp_path: Path,
) -> None:
    full_path = tmp_path / "full.parquet"
    update_path = tmp_path / "update.parquet"
    shared_key = {
        "instrument_id": "inst_600000",
        "canonical_code": "600000.SH",
        "market": "CN",
        "asset_type": "equity",
        "source_family": "cn_equity_minute",
        "frequency": "5m",
        "bar_end_time": "2025-12-26T09:35:00+08:00",
        "high_price": 10.2,
        "low_price": 9.8,
        "volume": 1000.0,
        "turnover": 10000.0,
        "raw_name": "plain",
    }
    pd.DataFrame(
        [
            {
                **shared_key,
                "open_price": 10.0,
                "close_price": 10.1,
                "raw_file_path": "量化数据/A股分时数据/full.csv",
            }
        ]
    ).to_parquet(full_path, index=False)
    pd.DataFrame(
        [
            {
                **shared_key,
                "open_price": 20.0,
                "close_price": 20.1,
                "raw_file_path": "baostock_sync/output/update.csv",
            },
            {
                "instrument_id": "inst_600000",
                "canonical_code": "600000.SH",
                "market": "CN",
                "asset_type": "equity",
                "source_family": "cn_equity_minute",
                "frequency": "5m",
                "bar_end_time": "2026-01-05T09:35:00+08:00",
                "open_price": 11.0,
                "high_price": 11.2,
                "low_price": 10.8,
                "close_price": 11.1,
                "volume": 1200.0,
                "turnover": 13200.0,
                "raw_name": "plain",
                "raw_file_path": "baostock_sync/output/update.csv",
            },
        ]
    ).to_parquet(update_path, index=False)
    params = _params(
        start="2025-12-26T09:35:00+08:00",
        end="2026-01-05T09:35:00+08:00",
    )

    bars = _load_bars_from_files(params, [full_path, update_path])

    assert bars[["bar_end_time", "open_price"]].to_dict("records") == [
        {
            "bar_end_time": "2025-12-26T09:35:00+08:00",
            "open_price": 10.0,
        },
        {
            "bar_end_time": "2026-01-05T09:35:00+08:00",
            "open_price": 11.0,
        },
    ]


def _params(**overrides: object) -> BacktestParams:
    values = {
        "catalog_path": Path("dummy.duckdb"),
        "start": "2025-01-03T09:35:00+08:00",
        "end": "2025-01-03T15:00:00+08:00",
        "top_n": 1,
        "initial_cash": 10_000.0,
        "lookback_bars": 1,
        "min_avg_turnover": None,
        "liquidity_window_bars": 1,
        "commission_bps": 0.0,
        "slippage_bps": 0.0,
        "lot_size": 100,
        "max_symbols": None,
        "output_dir": None,
    }
    values.update(overrides)
    return BacktestParams(**values)


def _bar(timestamp: str, instrument_id: str) -> dict[str, object]:
    return {
        "bar_end_time": timestamp,
        "instrument_id": instrument_id,
        "canonical_code": instrument_id,
        "open_price": 10.0,
        "close_price": 10.0,
        "turnover": 1_000_000.0,
        "tradable_bar": True,
        "limit_up_open": False,
        "limit_down_open": False,
    }


def _bar_with_close(
    timestamp: str,
    instrument_id: str,
    canonical_code: str,
    close_price: float,
) -> dict[str, object]:
    row = _bar(timestamp, instrument_id)
    row["canonical_code"] = canonical_code
    row["close_price"] = close_price
    row["open_price"] = close_price
    return row


def _feature_bar(
    timestamp: str,
    instrument_id: str,
    canonical_code: str,
    factor_value: float,
) -> dict[str, object]:
    row = _bar(timestamp, instrument_id)
    row["canonical_code"] = canonical_code
    row[_factor_column(1)] = factor_value
    return row


def _grid_runs() -> list[GridRun]:
    return [
        GridRun(
            name="lb1_top1",
            params=_params(top_n=1, lookback_bars=1, initial_cash=10_000.0),
            state=SimulationState(
                cash=10_000.0,
                lots={},
                previous_date=None,
                last_prices={},
            ),
            equity_values=[10_000.0],
        ),
        GridRun(
            name="lb1_top2",
            params=_params(top_n=2, lookback_bars=1, initial_cash=10_000.0),
            state=SimulationState(
                cash=10_000.0,
                lots={},
                previous_date=None,
                last_prices={},
            ),
            equity_values=[10_000.0],
        ),
    ]
