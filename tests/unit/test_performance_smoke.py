from __future__ import annotations

import time

import pandas as pd

from quant_research.backtest import (
    TargetWeightExecutionConfig,
    simulate_target_weight_execution_batches,
)


def test_target_weight_execution_performance_smoke_for_five_minute_slice() -> None:
    executions = _five_minute_slice(symbol_count=200, bar_count=48)
    config = TargetWeightExecutionConfig(initial_cash=1_000_000.0, lot_size=100)

    started = time.perf_counter()
    trades, equity, positions, diagnostics, _ = simulate_target_weight_execution_batches(
        [executions],
        config,
    )
    elapsed = time.perf_counter() - started

    assert elapsed < 5.0
    assert len(equity) == 48
    assert len(trades) > 0
    assert len(positions) > 0
    assert diagnostics.loc[0, "execution_row_count"] == 9_600


def _five_minute_slice(*, symbol_count: int, bar_count: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    timestamps = pd.date_range(
        "2025-01-03T09:35:00+08:00",
        periods=bar_count,
        freq="5min",
    )
    for bar_index, timestamp in enumerate(timestamps):
        for symbol_index in range(symbol_count):
            price = 10.0 + symbol_index * 0.01 + bar_index * 0.001
            rows.append(
                {
                    "exec_time": timestamp.isoformat(),
                    "instrument_id": f"inst-{symbol_index:04d}",
                    "open_price": price,
                    "close_price": price * 1.001,
                    "turnover": 10_000_000.0,
                    "tradable_bar": True,
                    "limit_up_open": False,
                    "limit_down_open": False,
                    "target_weight": 1.0 / symbol_count,
                }
            )
    return pd.DataFrame(rows)
