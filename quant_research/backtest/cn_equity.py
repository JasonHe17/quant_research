"""CN A-share execution and target-weight simulation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass(frozen=True, slots=True)
class CnEquityExecutionConstraintsConfig:
    """Configuration for deriving CN A-share tradability columns."""

    exclude_st: bool = False
    limit_up_bps: float | None = None
    limit_down_bps: float | None = None
    name_column: str = "raw_name"

    def __post_init__(self) -> None:
        if self.limit_up_bps is not None and self.limit_up_bps <= 0:
            raise ValueError("limit_up_bps must be positive")
        if self.limit_down_bps is not None and self.limit_down_bps <= 0:
            raise ValueError("limit_down_bps must be positive")
        if not self.name_column:
            raise ValueError("name_column must be non-empty")


@dataclass(frozen=True, slots=True)
class TargetWeightExecutionConfig:
    """Execution assumptions for target-weight rebalance simulation.

    The default model sizes open-price trades with open prices and only marks
    equity after trading with the bar close. Same-bar turnover based capacity is
    disabled by default because full-bar turnover is not known at the open.
    """

    initial_cash: float
    price_field: str = "open_price"
    sizing_price_field: str = "open_price"
    mark_price_field: str = "close_price"
    tradable_field: str = "tradable_bar"
    limit_up_field: str = "limit_up_open"
    limit_down_field: str = "limit_down_open"
    capacity_notional_field: str | None = "turnover"
    max_bar_turnover_participation: float | None = None
    allow_same_bar_capacity: bool = False
    commission_bps: float = 0.0
    slippage_bps: float = 0.0
    sell_stamp_tax_bps: float = 0.0
    min_commission: float = 0.0
    min_trade_weight: float = 0.0
    lot_size: int = 100

    def __post_init__(self) -> None:
        if self.initial_cash <= 0:
            raise ValueError("initial_cash must be positive")
        for name in (
            "price_field",
            "sizing_price_field",
            "mark_price_field",
            "tradable_field",
            "limit_up_field",
            "limit_down_field",
        ):
            if not getattr(self, name):
                raise ValueError(f"{name} must be non-empty")
        if self.commission_bps < 0:
            raise ValueError("commission_bps must be non-negative")
        if self.slippage_bps < 0:
            raise ValueError("slippage_bps must be non-negative")
        if self.sell_stamp_tax_bps < 0:
            raise ValueError("sell_stamp_tax_bps must be non-negative")
        if self.min_commission < 0:
            raise ValueError("min_commission must be non-negative")
        if not 0 <= self.min_trade_weight <= 1:
            raise ValueError("min_trade_weight must be in [0, 1]")
        if self.lot_size <= 0:
            raise ValueError("lot_size must be positive")
        if (
            self.max_bar_turnover_participation is not None
            and not 0 < self.max_bar_turnover_participation <= 1
        ):
            raise ValueError("max_bar_turnover_participation must be in (0, 1]")
        if (
            self.price_field == "open_price"
            and self.max_bar_turnover_participation is not None
            and self.capacity_notional_field in {"turnover", "volume"}
            and not self.allow_same_bar_capacity
        ):
            raise ValueError(
                "same-bar capacity uses full-bar turnover/volume with open-price "
                "execution; set allow_same_bar_capacity=True only when this is an "
                "explicit bar-volume execution assumption"
            )
        if self.price_field == "open_price" and self.sizing_price_field in {
            "close_price",
            "high_price",
            "low_price",
        }:
            raise ValueError(
                "open-price execution cannot size orders with same-bar OHLC fields "
                "that are unknown at the open"
            )


@dataclass(slots=True)
class TargetWeightSimulationState:
    """Mutable state for target-weight execution simulation."""

    cash: float
    lots: dict[str, list[dict[str, object]]]
    previous_date: str | None
    last_prices: dict[str, float]

    @classmethod
    def create(cls, initial_cash: float) -> "TargetWeightSimulationState":
        return cls(
            cash=float(initial_cash),
            lots={},
            previous_date=None,
            last_prices={},
        )


class TargetWeightExecutionSimulator:
    """Stateful simulator for target-weight rebalance executions."""

    def __init__(
        self,
        config: TargetWeightExecutionConfig,
        *,
        state: TargetWeightSimulationState | None = None,
    ) -> None:
        self.config = config
        self.state = state or TargetWeightSimulationState.create(config.initial_cash)
        self.last_execution_events = pd.DataFrame()

    def run(
        self, executions: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Run one execution batch and keep state for the next batch."""

        trades, equity, positions, state = simulate_target_weight_executions(
            executions,
            self.config,
            state=self.state,
        )
        self.state = state
        return trades, equity, positions

    def run_with_diagnostics(
        self, executions: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Run one batch and return standard execution diagnostics."""

        diagnostic_rows: list[dict[str, object]] = []
        trades, equity, positions, state = simulate_target_weight_executions(
            executions,
            self.config,
            state=self.state,
            diagnostics=diagnostic_rows,
        )
        self.state = state
        self.last_execution_events = pd.DataFrame(diagnostic_rows)
        diagnostics = target_weight_execution_diagnostics(
            executions,
            trades=trades,
            equity_curve=equity,
            execution_events=self.last_execution_events,
        )
        return trades, equity, positions, diagnostics

    def run_batches(
        self, batches: Iterable[pd.DataFrame]
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Run multiple execution batches while preserving simulator state."""

        trades, equity, positions, diagnostics, state = (
            simulate_target_weight_execution_batches(
                batches,
                self.config,
                state=self.state,
            )
        )
        self.state = state
        return trades, equity, positions, diagnostics


def add_cn_equity_execution_columns(
    frame: pd.DataFrame,
    config: CnEquityExecutionConstraintsConfig | None = None,
) -> pd.DataFrame:
    """Add standard CN A-share tradability and limit-state columns."""

    config = config or CnEquityExecutionConstraintsConfig()
    _require_columns(
        frame,
        (
            "instrument_id",
            "bar_end_time",
            "open_price",
            "close_price",
            "volume",
            "turnover",
        ),
        name="bars",
    )
    output = frame.copy()
    output["trade_date"] = output["bar_end_time"].astype(str).str.slice(0, 10)
    daily_close = (
        output.groupby(["instrument_id", "trade_date"], sort=False)["close_price"]
        .last()
        .rename("daily_close")
        .reset_index()
    )
    daily_close["previous_close"] = daily_close.groupby("instrument_id", sort=False)[
        "daily_close"
    ].shift(1)
    output = output.merge(
        daily_close.loc[:, ["instrument_id", "trade_date", "previous_close"]],
        on=["instrument_id", "trade_date"],
        how="left",
    )
    if config.exclude_st and config.name_column in output.columns:
        raw_name = output[config.name_column].fillna("").astype(str).str.upper()
        output["is_st"] = raw_name.str.contains("ST", regex=False)
    else:
        output["is_st"] = False
    has_valid_price = output["open_price"].astype(float).gt(0) & output[
        "close_price"
    ].astype(float).gt(0)
    has_activity = output["volume"].astype(float).gt(0) & output["turnover"].astype(
        float
    ).gt(0)
    output["suspended_bar"] = ~(has_valid_price & has_activity)
    output["tradable_bar"] = (
        has_valid_price & has_activity & ~output["is_st"].astype(bool)
    )
    previous_close = output["previous_close"].astype(float)
    has_previous = previous_close.gt(0)
    if config.limit_up_bps is None:
        output["limit_up_open"] = False
    else:
        limit_up_price = previous_close * (1.0 + config.limit_up_bps / 10_000.0)
        output["limit_up_open"] = has_previous & (
            output["open_price"].astype(float) >= limit_up_price
        )
    if config.limit_down_bps is None:
        output["limit_down_open"] = False
    else:
        limit_down_price = previous_close * (1.0 - config.limit_down_bps / 10_000.0)
        output["limit_down_open"] = has_previous & (
            output["open_price"].astype(float) <= limit_down_price
        )
    output["buyable_bar"] = output["tradable_bar"] & ~output["limit_up_open"]
    output["sellable_bar"] = output["tradable_bar"] & ~output["limit_down_open"]
    return output


def simulate_target_weight_executions(
    executions: pd.DataFrame,
    config: TargetWeightExecutionConfig,
    *,
    state: TargetWeightSimulationState | None = None,
    diagnostics: list[dict[str, object]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, TargetWeightSimulationState]:
    """Simulate target-weight rows into trades, equity, and final positions."""

    _require_columns(
        executions,
        (
            "exec_time",
            "instrument_id",
            config.price_field,
            config.sizing_price_field,
            config.mark_price_field,
            "target_weight",
        ),
        name="executions",
    )
    state = state or TargetWeightSimulationState.create(config.initial_cash)
    trades: list[dict[str, object]] = []
    equity_rows: list[dict[str, object]] = []
    shares_by_instrument = _shares_by_instrument(state.lots)
    for exec_time, group in executions.groupby("exec_time", sort=True):
        trade_date = str(exec_time)[:10]
        _roll_lots_to_sellable(state, trade_date)
        targets = _target_weights_by_instrument(group)
        relevant_instruments = set(state.lots) | set(targets)
        (
            execution_price_by_instrument,
            sizing_price_by_instrument,
            mark_price_by_instrument,
            tradable_by_instrument,
            limit_up_by_instrument,
            limit_down_by_instrument,
            capacity_by_instrument,
        ) = _execution_maps_for_group(
            group,
            relevant_instruments,
            config,
        )
        state.last_prices.update(sizing_price_by_instrument)
        equity = state.cash + _positions_value_from_shares(
            shares_by_instrument,
            state.last_prices,
        )
        if targets:
            instruments = sorted(set(state.lots) | set(targets))
            for instrument_id in instruments:
                if instrument_id not in execution_price_by_instrument:
                    _record_execution_diagnostic(
                        diagnostics,
                        exec_time=exec_time,
                        instrument_id=instrument_id,
                        reason="missing_price",
                    )
                    continue
                execution_price = execution_price_by_instrument[instrument_id]
                sizing_price = sizing_price_by_instrument.get(
                    instrument_id, execution_price
                )
                target_weight = targets.get(instrument_id, 0.0)
                target_value = equity * target_weight
                current_value = (
                    shares_by_instrument.get(instrument_id, 0) * sizing_price
                )
                delta_value = target_value - current_value
                if not tradable_by_instrument.get(instrument_id, True):
                    _record_execution_diagnostic(
                        diagnostics,
                        exec_time=exec_time,
                        instrument_id=instrument_id,
                        reason="non_tradable",
                        target_weight=target_weight,
                        desired_delta_value=delta_value,
                    )
                    continue
                if _below_min_trade_weight(delta_value, equity, config):
                    _record_execution_diagnostic(
                        diagnostics,
                        exec_time=exec_time,
                        instrument_id=instrument_id,
                        reason="below_min_trade_weight",
                        target_weight=target_weight,
                        desired_delta_value=delta_value,
                    )
                    continue
                if delta_value > execution_price * config.lot_size:
                    if limit_up_by_instrument.get(instrument_id, False):
                        _record_execution_diagnostic(
                            diagnostics,
                            exec_time=exec_time,
                            instrument_id=instrument_id,
                            reason="limit_up_buy_blocked",
                            target_weight=target_weight,
                            desired_delta_value=delta_value,
                        )
                        continue
                    desired_shares = _buy_shares_before_capacity(
                        delta_value, price=execution_price, config=config
                    )
                    shares = cap_trade_shares_by_notional(
                        desired_shares,
                        price=execution_price,
                        capacity_notional=capacity_by_instrument.get(instrument_id),
                        config=config,
                    )
                    if shares <= 0:
                        _record_execution_diagnostic(
                            diagnostics,
                            exec_time=exec_time,
                            instrument_id=instrument_id,
                            reason="capacity_zero",
                            target_weight=target_weight,
                            desired_delta_value=delta_value,
                            desired_shares=desired_shares,
                            executable_shares=0,
                            reference_price=execution_price,
                        )
                        continue
                    if shares < desired_shares:
                        _record_execution_diagnostic(
                            diagnostics,
                            exec_time=exec_time,
                            instrument_id=instrument_id,
                            reason="capacity_capped",
                            target_weight=target_weight,
                            desired_delta_value=delta_value,
                            desired_shares=desired_shares,
                            executable_shares=shares,
                            reference_price=execution_price,
                        )
                    cost_price = execution_price * (1.0 + config.slippage_bps / 10_000.0)
                    notional = shares * cost_price
                    commission = commission_cost(notional, config)
                    slippage_cost = shares * (cost_price - execution_price)
                    if notional + commission <= state.cash:
                        state.cash -= notional + commission
                        state.lots.setdefault(instrument_id, []).append(
                            {
                                "shares": shares,
                                "date": trade_date,
                                "sellable": False,
                            }
                        )
                        shares_by_instrument[instrument_id] = (
                            shares_by_instrument.get(instrument_id, 0) + shares
                        )
                        trades.append(
                            trade_row(
                                exec_time,
                                instrument_id,
                                "buy",
                                shares,
                                cost_price,
                                commission,
                                stamp_tax=0.0,
                                slippage_cost=slippage_cost,
                                reference_price=execution_price,
                            )
                        )
                    else:
                        _record_execution_diagnostic(
                            diagnostics,
                            exec_time=exec_time,
                            instrument_id=instrument_id,
                            reason="cash_insufficient",
                            target_weight=target_weight,
                            desired_delta_value=delta_value,
                            desired_shares=desired_shares,
                            executable_shares=0,
                        )
                elif delta_value < 0:
                    desired_sell_shares = int(-delta_value / execution_price)
                    if limit_down_by_instrument.get(instrument_id, False):
                        _record_execution_diagnostic(
                            diagnostics,
                            exec_time=exec_time,
                            instrument_id=instrument_id,
                            reason="limit_down_sell_blocked",
                            target_weight=target_weight,
                            desired_delta_value=delta_value,
                            desired_shares=desired_sell_shares,
                            executable_shares=0,
                        )
                        continue
                    available_sellable = sellable_shares(state.lots, instrument_id)
                    shares_to_sell = int(
                        min(
                            desired_sell_shares,
                            available_sellable,
                        )
                    )
                    if shares_to_sell < desired_sell_shares:
                        _record_execution_diagnostic(
                            diagnostics,
                            exec_time=exec_time,
                            instrument_id=instrument_id,
                            reason="t1_sell_limited",
                            target_weight=target_weight,
                            desired_delta_value=delta_value,
                            desired_shares=desired_sell_shares,
                            executable_shares=shares_to_sell,
                        )
                    before_capacity = shares_to_sell
                    shares_to_sell = cap_trade_shares_by_notional(
                        shares_to_sell,
                        price=execution_price,
                        capacity_notional=capacity_by_instrument.get(instrument_id),
                        config=config,
                    )
                    if shares_to_sell <= 0:
                        if before_capacity > 0:
                            _record_execution_diagnostic(
                                diagnostics,
                                exec_time=exec_time,
                                instrument_id=instrument_id,
                                reason="capacity_zero",
                                target_weight=target_weight,
                                desired_delta_value=delta_value,
                                desired_shares=before_capacity,
                                executable_shares=0,
                                reference_price=execution_price,
                            )
                        continue
                    if shares_to_sell < before_capacity:
                        _record_execution_diagnostic(
                            diagnostics,
                            exec_time=exec_time,
                            instrument_id=instrument_id,
                            reason="capacity_capped",
                            target_weight=target_weight,
                            desired_delta_value=delta_value,
                            desired_shares=before_capacity,
                            executable_shares=shares_to_sell,
                            reference_price=execution_price,
                        )
                    sell_price = execution_price * (
                        1.0 - config.slippage_bps / 10_000.0
                    )
                    sold = remove_sellable_shares(
                        state.lots,
                        instrument_id,
                        shares_to_sell,
                    )
                    notional = sold * sell_price
                    commission = commission_cost(notional, config)
                    stamp_tax = notional * config.sell_stamp_tax_bps / 10_000.0
                    slippage_cost = sold * (execution_price - sell_price)
                    state.cash += notional - commission - stamp_tax
                    remaining_shares = (
                        shares_by_instrument.get(instrument_id, 0) - sold
                    )
                    if remaining_shares > 0:
                        shares_by_instrument[instrument_id] = remaining_shares
                    else:
                        shares_by_instrument.pop(instrument_id, None)
                    trades.append(
                        trade_row(
                            exec_time,
                            instrument_id,
                            "sell",
                            sold,
                            sell_price,
                            commission,
                            stamp_tax=stamp_tax,
                            slippage_cost=slippage_cost,
                            reference_price=execution_price,
                        )
                    )
        state.last_prices.update(mark_price_by_instrument)
        marked_positions_value = _positions_value_from_shares(
            shares_by_instrument,
            state.last_prices,
        )
        equity_rows.append(
            {
                "timestamp": exec_time,
                "cash": state.cash,
                "positions_value": marked_positions_value,
                "equity": state.cash + marked_positions_value,
            }
        )
    return (
        pd.DataFrame(trades),
        pd.DataFrame(equity_rows),
        final_positions(state.lots),
        state,
    )


def simulate_target_weight_execution_batches(
    batches: Iterable[pd.DataFrame],
    config: TargetWeightExecutionConfig,
    *,
    state: TargetWeightSimulationState | None = None,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    TargetWeightSimulationState,
]:
    """Simulate an iterable of execution batches with one continuous state."""

    state = state or TargetWeightSimulationState.create(config.initial_cash)
    trade_frames: list[pd.DataFrame] = []
    equity_frames: list[pd.DataFrame] = []
    diagnostic_frames: list[pd.DataFrame] = []
    positions = final_positions(state.lots)
    for batch_index, batch in enumerate(batches):
        execution_events: list[dict[str, object]] = []
        trades, equity_curve, positions, state = simulate_target_weight_executions(
            batch,
            config,
            state=state,
            diagnostics=execution_events,
        )
        diagnostics = target_weight_execution_diagnostics(
            batch,
            trades=trades,
            equity_curve=equity_curve,
            execution_events=pd.DataFrame(execution_events),
        )
        diagnostics.insert(0, "batch_index", batch_index)
        trade_frames.append(trades)
        equity_frames.append(equity_curve)
        diagnostic_frames.append(diagnostics)
    return (
        _concat_frames(trade_frames),
        _concat_frames(equity_frames),
        positions,
        _concat_frames(diagnostic_frames),
        state,
    )


def target_weight_execution_diagnostics(
    executions: pd.DataFrame,
    *,
    trades: pd.DataFrame,
    equity_curve: pd.DataFrame,
    execution_events: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Summarize target-weight execution constraints and realized trading."""

    counts = execution_constraint_counts(executions)
    trade_count = int(len(trades))
    if trades.empty:
        totals = {
            "gross_traded_notional": 0.0,
            "total_commission": 0.0,
            "total_stamp_tax": 0.0,
            "total_slippage_cost": 0.0,
            "total_transaction_cost": 0.0,
        }
    else:
        totals = {
            "gross_traded_notional": float(trades["notional"].sum()),
            "total_commission": _sum_optional(trades, "commission"),
            "total_stamp_tax": _sum_optional(trades, "stamp_tax"),
            "total_slippage_cost": _sum_optional(trades, "slippage_cost"),
        }
        totals["total_transaction_cost"] = (
            totals["total_commission"]
            + totals["total_stamp_tax"]
            + totals["total_slippage_cost"]
        )
    average_equity = (
        float(equity_curve["equity"].astype(float).mean())
        if not equity_curve.empty and "equity" in equity_curve.columns
        else 0.0
    )
    row = {
        **counts,
        "trade_count": trade_count,
        **totals,
        "realized_turnover": totals["gross_traded_notional"] / average_equity
        if average_equity
        else 0.0,
    }
    row.update(execution_event_constraint_counts(execution_events))
    return pd.DataFrame([row])


def execution_event_constraint_counts(
    execution_events: pd.DataFrame | None,
) -> dict[str, int | float]:
    """Summarize diagnostic execution events emitted during simulation."""

    if execution_events is None or execution_events.empty:
        return {}
    if "reason" not in execution_events.columns:
        return {}
    reason_values = execution_events["reason"].astype("string")
    reasons = reason_values.dropna().astype(str)
    if reasons.empty:
        return {}
    counts: dict[str, int | float] = {
        f"{reason}_event_count": int(count)
        for reason, count in reasons.value_counts().items()
    }
    capacity_mask = reason_values.isin({"capacity_capped", "capacity_zero"}).fillna(
        False
    )
    if not capacity_mask.any():
        return counts
    capacity_events = execution_events.loc[capacity_mask]
    desired_shares = _numeric_event_series(capacity_events, "desired_shares").abs()
    executable_shares = _numeric_event_series(
        capacity_events, "executable_shares"
    ).abs()
    reference_price = _numeric_event_series(capacity_events, "reference_price").abs()
    unfilled_shares = (desired_shares - executable_shares).clip(lower=0.0)
    counts.update(
        {
            "capacity_limited_event_count": int(capacity_mask.sum()),
            "capacity_desired_shares": int(desired_shares.sum()),
            "capacity_executable_shares": int(executable_shares.sum()),
            "capacity_unfilled_shares": int(unfilled_shares.sum()),
            "capacity_desired_notional": float((desired_shares * reference_price).sum()),
            "capacity_executable_notional": float(
                (executable_shares * reference_price).sum()
            ),
            "capacity_unfilled_notional": float(
                (unfilled_shares * reference_price).sum()
            ),
        }
    )
    return counts


def execution_constraint_counts(frame: pd.DataFrame) -> dict[str, int]:
    """Summarize basic execution constraint states."""

    counts = empty_execution_constraint_counts()
    counts["execution_row_count"] = int(len(frame))
    if frame.empty:
        return counts
    tradable = (
        frame["tradable_bar"].fillna(False).astype(bool)
        if "tradable_bar" in frame.columns
        else pd.Series(True, index=frame.index)
    )
    limit_up = (
        frame["limit_up_open"].fillna(False).astype(bool)
        if "limit_up_open" in frame.columns
        else pd.Series(False, index=frame.index)
    )
    limit_down = (
        frame["limit_down_open"].fillna(False).astype(bool)
        if "limit_down_open" in frame.columns
        else pd.Series(False, index=frame.index)
    )
    target = (
        pd.to_numeric(frame["target_weight"], errors="coerce").fillna(0.0).gt(0)
        if "target_weight" in frame.columns
        else pd.Series(False, index=frame.index)
    )
    counts["non_tradable_row_count"] = int((~tradable).sum())
    counts["limit_up_open_row_count"] = int(limit_up.sum())
    counts["limit_down_open_row_count"] = int(limit_down.sum())
    counts["positive_target_row_count"] = int(target.sum())
    counts["positive_target_non_tradable_row_count"] = int((target & ~tradable).sum())
    counts["positive_target_limit_up_open_row_count"] = int((target & limit_up).sum())
    counts["positive_target_limit_down_open_row_count"] = int(
        (target & limit_down).sum()
    )
    return counts


def empty_execution_constraint_counts() -> dict[str, int]:
    """Return a zero-filled execution constraint count mapping."""

    return {
        "execution_row_count": 0,
        "non_tradable_row_count": 0,
        "limit_up_open_row_count": 0,
        "limit_down_open_row_count": 0,
        "positive_target_row_count": 0,
        "positive_target_non_tradable_row_count": 0,
        "positive_target_limit_up_open_row_count": 0,
        "positive_target_limit_down_open_row_count": 0,
    }


def merge_execution_constraint_counts(
    totals: dict[str, int | float],
    other: dict[str, int | float],
) -> None:
    """Add count values from ``other`` into ``totals`` in place."""

    for key, value in other.items():
        totals[key] = totals.get(key, 0) + value


def positions_value(
    lots: dict[str, list[dict[str, object]]], prices: dict[str, float]
) -> float:
    """Return marked value for all open lots."""

    return _positions_value_from_shares(_shares_by_instrument(lots), prices)


def instrument_shares(
    lots: dict[str, list[dict[str, object]]], instrument_id: str
) -> int:
    """Return total shares for one instrument."""

    return sum(int(lot["shares"]) for lot in lots.get(instrument_id, []))


def sellable_shares(
    lots: dict[str, list[dict[str, object]]], instrument_id: str
) -> int:
    """Return T+1 sellable shares for one instrument."""

    return sum(
        int(lot["shares"])
        for lot in lots.get(instrument_id, [])
        if bool(lot["sellable"])
    )


def remove_sellable_shares(
    lots: dict[str, list[dict[str, object]]], instrument_id: str, shares: int
) -> int:
    """Remove sellable shares from lots and return the actual sold shares."""

    remaining = shares
    sold = 0
    retained: list[dict[str, object]] = []
    for lot in lots.get(instrument_id, []):
        lot_shares = int(lot["shares"])
        if not bool(lot["sellable"]) or remaining <= 0:
            retained.append(lot)
            continue
        sell = min(lot_shares, remaining)
        sold += sell
        remaining -= sell
        if sell < lot_shares:
            retained.append({**lot, "shares": lot_shares - sell})
    lots[instrument_id] = retained
    return sold


def trade_row(
    timestamp: object,
    instrument_id: str,
    side: str,
    shares: int,
    price: float,
    commission: float,
    *,
    stamp_tax: float,
    slippage_cost: float,
    reference_price: float,
) -> dict[str, object]:
    """Build a standard trade row."""

    notional = shares * price
    reference_notional = shares * reference_price
    return {
        "timestamp": timestamp,
        "instrument_id": instrument_id,
        "side": side,
        "shares": shares,
        "price": price,
        "reference_price": reference_price,
        "commission": commission,
        "stamp_tax": stamp_tax,
        "slippage_cost": slippage_cost,
        "total_cost": commission + stamp_tax + slippage_cost,
        "notional": notional,
        "reference_notional": reference_notional,
    }


def commission_cost(notional: float, config: TargetWeightExecutionConfig) -> float:
    """Return commission cost for a notional trade."""

    commission = notional * config.commission_bps / 10_000.0
    if commission > 0 and config.min_commission > 0:
        return max(commission, config.min_commission)
    return commission


def cap_trade_shares_by_notional(
    shares: int,
    *,
    price: float,
    capacity_notional: float | None,
    config: TargetWeightExecutionConfig,
) -> int:
    """Cap trade shares by a notional capacity limit when configured."""

    if config.max_bar_turnover_participation is None:
        return shares
    if capacity_notional is None or capacity_notional <= 0 or price <= 0:
        return 0
    max_notional = capacity_notional * config.max_bar_turnover_participation
    max_shares = int(max_notional / price / config.lot_size) * config.lot_size
    return max(0, min(shares, max_shares))


def final_positions(lots: dict[str, list[dict[str, object]]]) -> pd.DataFrame:
    """Return final positions from open lots."""

    rows: list[dict[str, object]] = []
    for instrument_id in sorted(lots):
        shares = 0
        sellable = 0
        for lot in lots.get(instrument_id, []):
            lot_shares = int(lot["shares"])
            shares += lot_shares
            if bool(lot["sellable"]):
                sellable += lot_shares
        if shares > 0:
            rows.append(
                {
                    "instrument_id": instrument_id,
                    "shares": shares,
                    "sellable_shares": sellable,
                }
            )
    return pd.DataFrame(rows)


def _roll_lots_to_sellable(
    state: TargetWeightSimulationState,
    trade_date: str,
) -> None:
    if state.previous_date is not None and trade_date != state.previous_date:
        for instrument_lots in state.lots.values():
            for lot in instrument_lots:
                lot["sellable"] = True
    state.previous_date = trade_date


def _target_weights_by_instrument(group: pd.DataFrame) -> dict[str, float]:
    targets: dict[str, float] = {}
    instrument_values = group["instrument_id"].to_numpy(copy=False)
    target_values = group["target_weight"].to_numpy(copy=False)
    for instrument_id, target_weight in zip(instrument_values, target_values):
        if pd.notna(target_weight):
            targets[str(instrument_id)] = float(target_weight)
    return targets


def _execution_maps_for_group(
    group: pd.DataFrame,
    relevant_instruments: set[str],
    config: TargetWeightExecutionConfig,
) -> tuple[
    dict[str, float],
    dict[str, float],
    dict[str, float],
    dict[str, bool],
    dict[str, bool],
    dict[str, bool],
    dict[str, float],
]:
    execution_price_by_instrument: dict[str, float] = {}
    sizing_price_by_instrument: dict[str, float] = {}
    mark_price_by_instrument: dict[str, float] = {}
    tradable_by_instrument: dict[str, bool] = {}
    limit_up_by_instrument: dict[str, bool] = {}
    limit_down_by_instrument: dict[str, bool] = {}
    capacity_by_instrument: dict[str, float] = {}
    if not relevant_instruments:
        return (
            execution_price_by_instrument,
            sizing_price_by_instrument,
            mark_price_by_instrument,
            tradable_by_instrument,
            limit_up_by_instrument,
            limit_down_by_instrument,
            capacity_by_instrument,
        )

    instrument_values = group["instrument_id"].to_numpy(copy=False)
    execution_prices = group[config.price_field].to_numpy(copy=False)
    sizing_prices = group[config.sizing_price_field].to_numpy(copy=False)
    mark_prices = group[config.mark_price_field].to_numpy(copy=False)
    tradable_values = (
        group[config.tradable_field].to_numpy(copy=False)
        if config.tradable_field in group.columns
        else None
    )
    limit_up_values = (
        group[config.limit_up_field].to_numpy(copy=False)
        if config.limit_up_field in group.columns
        else None
    )
    limit_down_values = (
        group[config.limit_down_field].to_numpy(copy=False)
        if config.limit_down_field in group.columns
        else None
    )
    capacity_values = (
        group[config.capacity_notional_field].to_numpy(copy=False)
        if config.capacity_notional_field is not None
        and config.capacity_notional_field in group.columns
        else None
    )
    for index, instrument_value in enumerate(instrument_values):
        instrument_id = str(instrument_value)
        if instrument_id not in relevant_instruments:
            continue
        execution_price_by_instrument[instrument_id] = float(execution_prices[index])
        sizing_price_by_instrument[instrument_id] = float(sizing_prices[index])
        mark_price_by_instrument[instrument_id] = float(mark_prices[index])
        if tradable_values is not None:
            tradable_by_instrument[instrument_id] = bool(tradable_values[index])
        if limit_up_values is not None:
            limit_up_by_instrument[instrument_id] = bool(limit_up_values[index])
        if limit_down_values is not None:
            limit_down_by_instrument[instrument_id] = bool(limit_down_values[index])
        if capacity_values is not None:
            capacity_by_instrument[instrument_id] = float(capacity_values[index])
    return (
        execution_price_by_instrument,
        sizing_price_by_instrument,
        mark_price_by_instrument,
        tradable_by_instrument,
        limit_up_by_instrument,
        limit_down_by_instrument,
        capacity_by_instrument,
    )


def _shares_by_instrument(lots: dict[str, list[dict[str, object]]]) -> dict[str, int]:
    return {
        instrument_id: sum(int(lot["shares"]) for lot in instrument_lots)
        for instrument_id, instrument_lots in lots.items()
    }


def _positions_value_from_shares(
    shares_by_instrument: dict[str, int],
    prices: dict[str, float],
) -> float:
    return sum(
        shares * prices.get(instrument_id, 0.0)
        for instrument_id, shares in shares_by_instrument.items()
    )


def _below_min_trade_weight(
    delta_value: float,
    equity: float,
    config: TargetWeightExecutionConfig,
) -> bool:
    return (
        config.min_trade_weight > 0
        and equity > 0
        and abs(delta_value) / equity < config.min_trade_weight
    )


def _buy_shares_before_capacity(
    delta_value: float,
    *,
    price: float,
    config: TargetWeightExecutionConfig,
) -> int:
    return int(delta_value / price / config.lot_size) * config.lot_size


def _record_execution_diagnostic(
    diagnostics: list[dict[str, object]] | None,
    *,
    exec_time: object,
    instrument_id: str,
    reason: str,
    target_weight: float | None = None,
    desired_delta_value: float | None = None,
    desired_shares: int | None = None,
    executable_shares: int | None = None,
    reference_price: float | None = None,
) -> None:
    if diagnostics is None:
        return
    diagnostics.append(
        {
            "timestamp": exec_time,
            "instrument_id": instrument_id,
            "reason": reason,
            "target_weight": target_weight,
            "desired_delta_value": desired_delta_value,
            "desired_shares": desired_shares,
            "executable_shares": executable_shares,
            "reference_price": reference_price,
        }
    )


def _sum_optional(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns:
        return 0.0
    return float(frame[column].fillna(0.0).astype(float).sum())


def _numeric_event_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(0.0, index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)


def _concat_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _require_columns(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    *,
    name: str,
) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")
