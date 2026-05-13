"""Standard backtest ledger table schemas."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quant_research.schemas import validate_standard_table


ORDER_COLUMNS = (
    "timestamp",
    "instrument_id",
    "side",
    "quantity",
    "order_type",
    "target_weight",
    "reason",
)
FILL_COLUMNS = (
    "timestamp",
    "instrument_id",
    "side",
    "quantity",
    "price",
    "commission",
    "stamp_tax",
    "slippage_cost",
    "notional",
    "order_id",
)
LOT_COLUMNS = (
    "instrument_id",
    "shares",
    "acquired_date",
    "sellable",
    "cost_price",
)
POSITION_COLUMNS = (
    "timestamp",
    "instrument_id",
    "quantity",
    "market_value",
    "sellable_quantity",
)
LEDGER_COLUMNS = (
    "timestamp",
    "cash",
    "positions_value",
    "equity",
)


@dataclass(frozen=True, slots=True)
class OrderRow:
    """One standard order table row."""

    timestamp: object
    instrument_id: str
    side: str
    quantity: float
    order_type: str = "market"
    target_weight: float | None = None
    reason: str = "rebalance"


@dataclass(frozen=True, slots=True)
class FillRow:
    """One standard fill table row."""

    timestamp: object
    instrument_id: str
    side: str
    quantity: float
    price: float
    commission: float = 0.0
    stamp_tax: float = 0.0
    slippage_cost: float = 0.0
    order_id: str | None = None

    @property
    def notional(self) -> float:
        return self.quantity * self.price


@dataclass(frozen=True, slots=True)
class PositionLotRow:
    """One standard position lot row."""

    instrument_id: str
    shares: int
    acquired_date: str
    sellable: bool
    cost_price: float | None = None


@dataclass(frozen=True, slots=True)
class LedgerSnapshotRow:
    """One standard account ledger snapshot row."""

    timestamp: object
    cash: float
    positions_value: float
    equity: float


def orders_to_frame(rows: list[OrderRow] | tuple[OrderRow, ...]) -> pd.DataFrame:
    """Convert order rows to the standard order frame."""

    if not rows:
        return empty_order_frame()
    return pd.DataFrame([_order_to_dict(row) for row in rows]).loc[:, ORDER_COLUMNS]


def fills_to_frame(rows: list[FillRow] | tuple[FillRow, ...]) -> pd.DataFrame:
    """Convert fill rows to the standard fill frame."""

    if not rows:
        return empty_fill_frame()
    return pd.DataFrame([_fill_to_dict(row) for row in rows]).loc[:, FILL_COLUMNS]


def lots_to_frame(
    rows: list[PositionLotRow] | tuple[PositionLotRow, ...]
) -> pd.DataFrame:
    """Convert lot rows to the standard lot frame."""

    if not rows:
        return empty_lot_frame()
    return pd.DataFrame([_lot_to_dict(row) for row in rows]).loc[:, LOT_COLUMNS]


def ledger_to_frame(
    rows: list[LedgerSnapshotRow] | tuple[LedgerSnapshotRow, ...]
) -> pd.DataFrame:
    """Convert account snapshots to the standard ledger frame."""

    if not rows:
        return empty_ledger_frame()
    return pd.DataFrame([_ledger_to_dict(row) for row in rows]).loc[:, LEDGER_COLUMNS]


def fills_from_trades(trades: pd.DataFrame) -> pd.DataFrame:
    """Map legacy trade rows to the standard fill table."""

    if trades.empty:
        return empty_fill_frame()
    _require_columns(
        trades,
        ("timestamp", "instrument_id", "side", "shares", "price"),
        name="trades",
    )
    output = pd.DataFrame()
    output["timestamp"] = trades["timestamp"]
    output["instrument_id"] = trades["instrument_id"]
    output["side"] = trades["side"]
    output["quantity"] = trades["shares"].astype(float)
    output["price"] = trades["price"].astype(float)
    output["commission"] = _optional_numeric(trades, "commission")
    output["stamp_tax"] = _optional_numeric(trades, "stamp_tax")
    output["slippage_cost"] = _optional_numeric(trades, "slippage_cost")
    output["notional"] = output["quantity"] * output["price"]
    output["order_id"] = trades["order_id"] if "order_id" in trades.columns else None
    return output.loc[:, FILL_COLUMNS]


def empty_order_frame() -> pd.DataFrame:
    """Return an empty standard order frame."""

    return pd.DataFrame(columns=ORDER_COLUMNS)


def empty_fill_frame() -> pd.DataFrame:
    """Return an empty standard fill frame."""

    return pd.DataFrame(columns=FILL_COLUMNS)


def empty_lot_frame() -> pd.DataFrame:
    """Return an empty standard lot frame."""

    return pd.DataFrame(columns=LOT_COLUMNS)


def empty_position_frame() -> pd.DataFrame:
    """Return an empty standard position frame."""

    return pd.DataFrame(columns=POSITION_COLUMNS)


def empty_ledger_frame() -> pd.DataFrame:
    """Return an empty standard ledger frame."""

    return pd.DataFrame(columns=LEDGER_COLUMNS)


def validate_order_frame(frame: pd.DataFrame) -> None:
    """Validate the standard order frame columns."""

    validate_standard_table("orders", frame)


def validate_fill_frame(frame: pd.DataFrame) -> None:
    """Validate the standard fill frame columns."""

    validate_standard_table("fills", frame)


def validate_lot_frame(frame: pd.DataFrame) -> None:
    """Validate the standard lot frame columns."""

    validate_standard_table("lots", frame)


def validate_position_frame(frame: pd.DataFrame) -> None:
    """Validate the standard position frame columns."""

    validate_standard_table("positions", frame)


def validate_ledger_frame(frame: pd.DataFrame) -> None:
    """Validate the standard account ledger frame columns."""

    validate_standard_table("ledger", frame)


def _order_to_dict(row: OrderRow) -> dict[str, object]:
    return {
        "timestamp": row.timestamp,
        "instrument_id": row.instrument_id,
        "side": row.side,
        "quantity": row.quantity,
        "order_type": row.order_type,
        "target_weight": row.target_weight,
        "reason": row.reason,
    }


def _fill_to_dict(row: FillRow) -> dict[str, object]:
    return {
        "timestamp": row.timestamp,
        "instrument_id": row.instrument_id,
        "side": row.side,
        "quantity": row.quantity,
        "price": row.price,
        "commission": row.commission,
        "stamp_tax": row.stamp_tax,
        "slippage_cost": row.slippage_cost,
        "notional": row.notional,
        "order_id": row.order_id,
    }


def _lot_to_dict(row: PositionLotRow) -> dict[str, object]:
    return {
        "instrument_id": row.instrument_id,
        "shares": row.shares,
        "acquired_date": row.acquired_date,
        "sellable": row.sellable,
        "cost_price": row.cost_price,
    }


def _ledger_to_dict(row: LedgerSnapshotRow) -> dict[str, object]:
    return {
        "timestamp": row.timestamp,
        "cash": row.cash,
        "positions_value": row.positions_value,
        "equity": row.equity,
    }


def _optional_numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(0.0, index=frame.index)
    return frame[column].fillna(0.0).astype(float)


def _require_columns(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    *,
    name: str,
) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")
