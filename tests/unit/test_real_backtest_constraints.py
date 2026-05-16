from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

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
    _build_next_bar_executions,
    _build_reversal_signals,
    _execution_constraint_counts,
    _simulate,
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

    trades, _, _, next_state = _simulate(
        executions,
        _params(
            initial_cash=100_000.0,
            max_bar_turnover_participation=0.05,
            allow_same_bar_capacity=True,
        ),
    )

    assert list(trades["shares"]) == [500]
    assert next_state.cash == 95_000.0


def test_open_execution_rejects_same_bar_capacity_without_explicit_policy() -> None:
    with pytest.raises(ValueError, match="same-bar capacity"):
        TargetWeightExecutionConfig(
            initial_cash=100_000.0,
            max_bar_turnover_participation=0.05,
        )


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

    assert counts == {
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
