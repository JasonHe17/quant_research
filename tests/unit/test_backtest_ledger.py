from __future__ import annotations

import pandas as pd
import pytest

from quant_research.backtest import (
    FillRow,
    LedgerSnapshotRow,
    OrderRow,
    PositionLotRow,
    fills_from_trades,
    fills_to_frame,
    ledger_to_frame,
    lots_to_frame,
    orders_to_frame,
    validate_fill_frame,
    validate_ledger_frame,
    validate_lot_frame,
    validate_order_frame,
)


def test_standard_ledger_tables_validate() -> None:
    orders = orders_to_frame(
        [
            OrderRow(
                timestamp="2025-01-03T09:35:00+08:00",
                instrument_id="inst-1",
                side="buy",
                quantity=100,
                target_weight=0.5,
            )
        ]
    )
    fills = fills_to_frame(
        [
            FillRow(
                timestamp="2025-01-03T09:35:00+08:00",
                instrument_id="inst-1",
                side="buy",
                quantity=100,
                price=10.0,
                commission=1.0,
            )
        ]
    )
    lots = lots_to_frame(
        [
            PositionLotRow(
                instrument_id="inst-1",
                shares=100,
                acquired_date="2025-01-03",
                sellable=False,
                cost_price=10.0,
            )
        ]
    )
    ledger = ledger_to_frame(
        [
            LedgerSnapshotRow(
                timestamp="2025-01-03T09:35:00+08:00",
                cash=9_000.0,
                positions_value=1_000.0,
                equity=10_000.0,
            )
        ]
    )

    validate_order_frame(orders)
    validate_fill_frame(fills)
    validate_lot_frame(lots)
    validate_ledger_frame(ledger)
    assert fills.loc[0, "notional"] == 1_000.0


def test_fills_from_legacy_trades() -> None:
    trades = pd.DataFrame(
        [
            {
                "timestamp": "2025-01-03T09:35:00+08:00",
                "instrument_id": "inst-1",
                "side": "sell",
                "shares": 50,
                "price": 10.0,
                "commission": 0.5,
                "stamp_tax": 0.1,
                "slippage_cost": 0.2,
            }
        ]
    )

    fills = fills_from_trades(trades)

    validate_fill_frame(fills)
    assert fills.loc[0, "quantity"] == 50.0
    assert fills.loc[0, "notional"] == 500.0


def test_ledger_validators_reject_missing_columns() -> None:
    with pytest.raises(ValueError, match="orders"):
        validate_order_frame(pd.DataFrame([{"timestamp": "t0"}]))
