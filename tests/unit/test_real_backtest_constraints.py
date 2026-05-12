from __future__ import annotations

from pathlib import Path

import pandas as pd

from examples.run_baseline_a_real_backtest import (
    BacktestParams,
    SimulationState,
    _add_execution_constraint_columns,
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
    assert bool(limit_down_row["limit_down_open"])
    assert bool(st_row["is_st"])
    assert not bool(st_row["tradable_bar"])


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
        _params(initial_cash=100_000.0, max_bar_turnover_participation=0.05),
    )

    assert list(trades["shares"]) == [500]
    assert next_state.cash == 95_000.0


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
