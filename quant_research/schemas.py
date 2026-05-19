"""Standard table schema validation helpers."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True, slots=True)
class ColumnSpec:
    """One required column in a standard table schema."""

    name: str
    kind: str | None = None
    nullable: bool = True


@dataclass(frozen=True, slots=True)
class TableSchema:
    """A lightweight schema for framework DataFrame contracts."""

    name: str
    columns: tuple[ColumnSpec, ...]
    allow_extra_columns: bool = True

    @property
    def required_columns(self) -> tuple[str, ...]:
        return tuple(column.name for column in self.columns)


FACTOR_SCHEMA = TableSchema(
    name="factor",
    columns=(
        ColumnSpec("factor_name", "string", nullable=False),
        ColumnSpec("instrument_id", "string", nullable=False),
        ColumnSpec("factor_value", "numeric"),
    ),
)
SIGNAL_SCHEMA = TableSchema(
    name="signal",
    columns=(
        ColumnSpec("timestamp", nullable=False),
        ColumnSpec("instrument_id", "string", nullable=False),
        ColumnSpec("signal", "numeric", nullable=False),
    ),
)
PORTFOLIO_TARGET_WEIGHTS_SCHEMA = TableSchema(
    name="portfolio_target_weights",
    columns=(
        ColumnSpec("timestamp", nullable=False),
        ColumnSpec("instrument_id", "string", nullable=False),
        ColumnSpec("target_weight", "numeric", nullable=False),
    ),
)
PORTFOLIO_REBALANCE_ORDERS_SCHEMA = TableSchema(
    name="portfolio_rebalance_orders",
    columns=(
        ColumnSpec("timestamp", nullable=False),
        ColumnSpec("instrument_id", "string", nullable=False),
        ColumnSpec("current_weight", "numeric", nullable=False),
        ColumnSpec("target_weight", "numeric", nullable=False),
        ColumnSpec("delta_weight", "numeric", nullable=False),
    ),
)
BACKTEST_TRADES_SCHEMA = TableSchema(
    name="backtest_trades",
    columns=(
        ColumnSpec("timestamp", nullable=False),
        ColumnSpec("instrument_id", "string", nullable=False),
        ColumnSpec("quantity", "numeric", nullable=False),
        ColumnSpec("price", "numeric", nullable=False),
    ),
)
BACKTEST_POSITIONS_SCHEMA = TableSchema(
    name="backtest_positions",
    columns=(
        ColumnSpec("timestamp", nullable=False),
        ColumnSpec("instrument_id", "string", nullable=False),
        ColumnSpec("quantity", "numeric", nullable=False),
        ColumnSpec("market_value", "numeric", nullable=False),
    ),
)
BACKTEST_EQUITY_CURVE_SCHEMA = TableSchema(
    name="backtest_equity_curve",
    columns=(
        ColumnSpec("timestamp", nullable=False),
        ColumnSpec("equity", "numeric", nullable=False),
    ),
)
UNIVERSE_MEMBERS_SCHEMA = TableSchema(
    name="universe_members",
    columns=(ColumnSpec("instrument_id", "string", nullable=False),),
)
ORDERS_SCHEMA = TableSchema(
    name="orders",
    columns=(
        ColumnSpec("timestamp", nullable=False),
        ColumnSpec("instrument_id", "string", nullable=False),
        ColumnSpec("side", "string", nullable=False),
        ColumnSpec("quantity", "numeric", nullable=False),
        ColumnSpec("order_type", "string", nullable=False),
        ColumnSpec("target_weight", "numeric"),
        ColumnSpec("reason", "string", nullable=False),
    ),
)
FILLS_SCHEMA = TableSchema(
    name="fills",
    columns=(
        ColumnSpec("timestamp", nullable=False),
        ColumnSpec("instrument_id", "string", nullable=False),
        ColumnSpec("side", "string", nullable=False),
        ColumnSpec("quantity", "numeric", nullable=False),
        ColumnSpec("price", "numeric", nullable=False),
        ColumnSpec("commission", "numeric", nullable=False),
        ColumnSpec("stamp_tax", "numeric", nullable=False),
        ColumnSpec("slippage_cost", "numeric", nullable=False),
        ColumnSpec("notional", "numeric", nullable=False),
        ColumnSpec("order_id", "string"),
    ),
)
LOTS_SCHEMA = TableSchema(
    name="lots",
    columns=(
        ColumnSpec("instrument_id", "string", nullable=False),
        ColumnSpec("shares", "numeric", nullable=False),
        ColumnSpec("acquired_date", nullable=False),
        ColumnSpec("sellable", "boolean", nullable=False),
        ColumnSpec("cost_price", "numeric"),
    ),
)
POSITIONS_SCHEMA = TableSchema(
    name="positions",
    columns=(
        ColumnSpec("timestamp", nullable=False),
        ColumnSpec("instrument_id", "string", nullable=False),
        ColumnSpec("quantity", "numeric", nullable=False),
        ColumnSpec("market_value", "numeric", nullable=False),
        ColumnSpec("sellable_quantity", "numeric"),
    ),
)
LEDGER_SCHEMA = TableSchema(
    name="ledger",
    columns=(
        ColumnSpec("timestamp", nullable=False),
        ColumnSpec("cash", "numeric", nullable=False),
        ColumnSpec("positions_value", "numeric", nullable=False),
        ColumnSpec("equity", "numeric", nullable=False),
    ),
)

STANDARD_TABLE_SCHEMAS = {
    schema.name: schema
    for schema in (
        FACTOR_SCHEMA,
        SIGNAL_SCHEMA,
        PORTFOLIO_TARGET_WEIGHTS_SCHEMA,
        PORTFOLIO_REBALANCE_ORDERS_SCHEMA,
        BACKTEST_TRADES_SCHEMA,
        BACKTEST_POSITIONS_SCHEMA,
        BACKTEST_EQUITY_CURVE_SCHEMA,
        UNIVERSE_MEMBERS_SCHEMA,
        ORDERS_SCHEMA,
        FILLS_SCHEMA,
        LOTS_SCHEMA,
        POSITIONS_SCHEMA,
        LEDGER_SCHEMA,
    )
}


def validate_standard_table(name: str, frame: pd.DataFrame) -> None:
    """Validate a frame against one named standard schema."""

    try:
        schema = STANDARD_TABLE_SCHEMAS[name]
    except KeyError as exc:
        raise KeyError(f"unknown standard table schema: {name}") from exc
    validate_table_schema(frame, schema)
    if name == "factor":
        _validate_factor_time_column(frame)


def validate_table_schema(frame: pd.DataFrame, schema: TableSchema) -> None:
    """Validate required columns and simple dtype families."""

    missing = [column for column in schema.required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{schema.name} is missing required columns: {missing}")
    if not schema.allow_extra_columns:
        extra = [column for column in frame.columns if column not in schema.required_columns]
        if extra:
            raise ValueError(f"{schema.name} has unexpected columns: {extra}")
    for column in schema.columns:
        series = frame[column.name]
        if not column.nullable and series.isna().any():
            raise ValueError(f"{schema.name}.{column.name} must not contain nulls")
        if column.kind is not None:
            _validate_kind(series, schema=schema, column=column)


def _validate_kind(
    series: pd.Series,
    *,
    schema: TableSchema,
    column: ColumnSpec,
) -> None:
    non_null = series.dropna()
    if non_null.empty:
        return
    if column.kind == "numeric":
        numeric = pd.to_numeric(non_null, errors="coerce")
        if numeric.notna().all():
            return
    elif column.kind == "string":
        inferred = pd.api.types.infer_dtype(non_null, skipna=True)
        if inferred in {"string", "unicode", "bytes", "categorical"}:
            return
    elif column.kind == "boolean":
        inferred = pd.api.types.infer_dtype(non_null, skipna=True)
        if inferred == "boolean":
            return
    else:
        raise ValueError(f"unsupported schema column kind: {column.kind}")
    raise ValueError(f"{schema.name}.{column.name} must be {column.kind}")


def _validate_factor_time_column(frame: pd.DataFrame) -> None:
    time_columns = [column for column in ("timestamp", "bar_end_time") if column in frame]
    if not time_columns:
        raise ValueError("factor must contain either timestamp or bar_end_time")
    for column in time_columns:
        if frame[column].isna().any():
            raise ValueError(f"factor.{column} must not contain nulls")
